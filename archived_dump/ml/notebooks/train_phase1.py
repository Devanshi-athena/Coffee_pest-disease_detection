# %% [markdown]
# # coffee-leaf-ai, Phase 1: healthy vs rust (experiment 1)
#
# **Data:** private Kaggle dataset `devanshigulati/coffee-leaf-ai-dataset` (built by
# `ml/scripts/07_build_phase1_dataset.py`; nothing in it is changed here).
#
# **Rules for this run**
# - Model: ConvNeXt-Tiny, ImageNet-22k pretrained (`convnext_tiny.fb_in22k_ft_in1k`).
# - Variants: `all_sources` (JMuBEN + Uganda + BRACOL) and `no_jmuben` (Uganda + BRACOL).
# - Sampling uses the manifest's `balanced_weight` exactly as built (DATASET.md); an excluded source's rows are dropped.
# - Best epoch and preferred variant are chosen on **validation macro F1 only**.
# - `test_rocole` (RoCoLe) is never used for training, epoch choice, calibration, threshold or variant choice.
# - `ood` is never used for training or any choice; it is only reported.
# - Temperature and "not sure" threshold are fitted on validation; calibration elsewhere is not guaranteed.
# - ONNX output is checked against PyTorch on random and real images; a mismatch stops the run.

# %%
import importlib.util, subprocess, sys
missing = [p for p in ("timm", "onnx", "onnxscript", "onnxruntime") if importlib.util.find_spec(p) is None]
if missing:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", *missing], check=True)
print("installed:" if missing else "all packages present", missing)

# %%
import os, sys, glob, json, hashlib
from pathlib import Path

def find_project_root():
    if os.environ.get("COFFEE_ROOT"):
        return Path(os.environ["COFFEE_ROOT"])
    for pattern in ["/kaggle/input/*", "/kaggle/input/*/*", "/kaggle/input/*/*/*", "/kaggle/input/*/*/*/*"]:
        for p in glob.glob(pattern):
            if Path(p, "data", "processed_phase1", "manifest.csv").is_file() and Path(p, "ml", "src", "crops.py").is_file():
                return Path(p)
    raise FileNotFoundError("Attach the dataset devanshigulati/coffee-leaf-ai-dataset.")

ROOT = find_project_root()
OUT = Path(os.environ.get("COFFEE_OUT") or ("/kaggle/working/phase1" if Path("/kaggle/working").exists() else ROOT / "artifacts" / "phase1"))
OUT.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(ROOT / "ml"))

# The helpers imported from the dataset must be exactly the reviewed versions.
EXPECTED_SHA256 = {
    "common.py": "6ad5c06f2fc35db20d4054ebc97e3950d0fcc1390c7b31a092404fc626c676cf",
    "crops.py": "0020042905c873654eb653190c9d2c880c7f151f103aa59dcc303739c38cd9ec",
    "evaluate.py": "5d4fb317669c7e64cd729e18550816812214266767e5422fe44c858fa182bca7",
    "export_onnx.py": "b1b761c4b41b1092779af9db54e0d0f4e6d32dc3ec027678659dca6293c90760"
}
for f, h in EXPECTED_SHA256.items():
    got = hashlib.sha256(Path(ROOT, "ml", "src", f).read_bytes()).hexdigest()
    assert got == h, f"ml/src/{f} in the dataset differs from the reviewed version"
print("project root:", ROOT, "| outputs:", OUT, "| helper checksums OK")

# %% [markdown]
# ## Training and evaluation code (reviewed copy of `ml/src/phase1.py`)

# %%
"""Phase 1 (healthy vs rust) training, evaluation and export. Used by ml/notebooks/train_phase1.ipynb.

Data comes from data/processed_phase1/ (manifest.csv + crops.csv), built by ml/scripts/07_build_phase1_dataset.py.
"""

import json
import math
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms as T

from src.crops import apply_view
from src.evaluate import (compute_metrics, expected_calibration_error, fit_temperature, plot_confusion_matrix,
                       plot_reliability, threshold_sweep)
from src.export_onnx import export_onnx, verify_onnx

DATA_SUBDIR = Path("data") / "processed_phase1"
FULL = (0.0, 0.0, 1.0, 1.0)


# ---------------------------------------------------------------- data

def find_root(candidates: list[Path]) -> Path:
    """First folder that contains data/processed_phase1/manifest.csv (searches two levels down)."""
    for c in candidates:
        for p in [c, *c.glob("*"), *c.glob("*/*"), *c.glob("*/*/*")]:
            if (p / DATA_SUBDIR / "manifest.csv").is_file():
                return p
    raise FileNotFoundError(f"{DATA_SUBDIR}/manifest.csv not found under {candidates}. Attach the dataset made "
                            "from data/phase1_kaggle.zip.")


def load_tables(root: Path) -> tuple[Path, pd.DataFrame, pd.DataFrame]:
    base = root / DATA_SUBDIR
    m = pd.read_csv(base / "manifest.csv", low_memory=False)
    m = m[m["processed_path"].notna() & m["split"].notna()].copy()
    for col in ("task_label", "ood_type"):
        m[col] = m[col].fillna("")
    crops = pd.read_csv(base / "crops.csv")
    return base, m, crops


def class_source_shares(df: pd.DataFrame, weight: str = "balanced_weight") -> pd.DataFrame:
    """Share of the total sampling weight per class and source (what training actually sees)."""
    t = df.groupby(["task_label", "source"])[weight].sum()
    return (t / t.sum()).round(3).unstack(fill_value=0)


def views_by_path(crops: pd.DataFrame, training: bool, view: str = "full") -> dict[str, list[tuple]]:
    sel = crops[crops["use_for_training"]] if training else crops[crops["view"] == view]
    out: dict[str, list[tuple]] = {}
    for r in sel.itertuples():
        out.setdefault(r.processed_path, []).append((r.x0, r.y0, r.x1, r.y1))
    return out


class ViewDataset(Dataset):
    """One item per image; in training a random one of its precomputed views is used each time."""

    def __init__(self, base: Path, df: pd.DataFrame, views: dict, classes: list[str], transform, train: bool):
        self.base, self.transform, self.train = base, transform, train
        self.paths = df["processed_path"].tolist()
        self.labels = [classes.index(x) if x in classes else -1 for x in df["task_label"]]
        self.views = [views.get(p, [FULL]) for p in self.paths]

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, i):
        with Image.open(self.base / self.paths[i]) as im:
            img = im.convert("RGB")
        boxes = self.views[i]
        img = apply_view(img, random.choice(boxes) if self.train else boxes[0])
        return self.transform(img), self.labels[i]


def transforms(size: int = 224, resize: int = 256):
    norm = T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    train = T.Compose([
        T.RandomResizedCrop(size, scale=(0.6, 1.0), ratio=(0.75, 1.333)),
        T.RandomHorizontalFlip(), T.RandomVerticalFlip(),
        T.RandomApply([T.RandomRotation(25)], p=0.5),
        T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.25, hue=0.03),
        T.RandomApply([T.GaussianBlur(5, sigma=(0.1, 1.5))], p=0.2),
        T.ToTensor(), norm,
    ])
    # Must match backend/app/inference.py: resize shorter side (bilinear) + center crop.
    evaluate = T.Compose([T.Resize(resize, interpolation=T.InterpolationMode.BILINEAR), T.CenterCrop(size),
                          T.ToTensor(), norm])
    return train, evaluate


# ---------------------------------------------------------------- model + training

def build_model(name: str, pretrained: bool, num_classes: int):
    import timm

    return timm.create_model(name, pretrained=pretrained, num_classes=num_classes, drop_path_rate=0.1)


@torch.no_grad()
def predict_logits(model, loader, device) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    ys, zs = [], []
    for x, y in loader:
        zs.append(model(x.to(device, non_blocking=True)).float().cpu().numpy())
        ys.append(np.asarray(y))
    return np.concatenate(ys), np.concatenate(zs)


def softmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


def train_variant(name: str, exclude_sources: list[str], base: Path, m: pd.DataFrame, crops: pd.DataFrame,
                  classes: list[str], out_dir: Path, hp: dict, device) -> dict:
    """Trains one variant, picks the best epoch on val macro F1, evaluates everything, exports ONNX."""
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(hp["seed"])
    random.seed(hp["seed"])
    np.random.seed(hp["seed"])
    keep = ~m["source"].isin(exclude_sources)
    tr = m[(m["split"] == "train") & keep].reset_index(drop=True)
    va = m[(m["split"] == "val") & keep & m["is_group_rep"]].reset_index(drop=True)
    tf_train, tf_eval = transforms(hp["size"], hp["resize"])
    # balanced_weight exactly as stored in the manifest (DATASET.md); excluded sources are simply dropped
    if tr["balanced_weight"].isna().any():
        raise ValueError("balanced_weight missing for some training rows")
    w = tr["balanced_weight"].to_numpy()
    shares = class_source_shares(tr)
    print(f"[{name}] sampling weight share by class/source:\n{shares}")

    train_ds = ViewDataset(base, tr, views_by_path(crops, training=True), classes, tf_train, train=True)
    sampler = WeightedRandomSampler(torch.as_tensor(w, dtype=torch.double), num_samples=hp["epoch_size"],
                                    replacement=True)
    train_dl = DataLoader(train_ds, batch_size=hp["batch_size"], sampler=sampler, num_workers=hp["workers"],
                          pin_memory=device.type == "cuda", drop_last=True)

    def eval_loader(df, view="full"):
        ds = ViewDataset(base, df, views_by_path(crops, training=False, view=view), classes, tf_eval, train=False)
        return DataLoader(ds, batch_size=hp["batch_size"] * 2, shuffle=False, num_workers=hp["workers"])

    val_dl = eval_loader(va)
    model = build_model(hp["model"], hp["pretrained"], len(classes)).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=hp["lr"], weight_decay=hp["weight_decay"])
    steps = hp["epochs"] * len(train_dl)
    warm = len(train_dl)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / warm if s < warm else 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, steps - warm))))
    crit = nn.CrossEntropyLoss(label_smoothing=hp["label_smoothing"])
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    history, best, bad_epochs = [], {"macro_f1": -1.0, "epoch": 0}, 0
    for epoch in range(1, hp["epochs"] + 1):
        model.train()
        t0, total, n = time.time(), 0.0, 0
        for x, y in train_dl:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=use_amp):
                loss = crit(model(x), y)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            total, n = total + loss.item() * len(y), n + len(y)
        yv, zv = predict_logits(model, val_dl, device)
        mv = compute_metrics(yv, softmax(zv), classes)
        row = {"epoch": epoch, "train_loss": round(total / n, 4), "val_acc": round(mv["accuracy"], 4),
               "val_macro_f1": round(mv["macro_f1"], 4), "sec": round(time.time() - t0)}
        history.append(row)
        print(f"[{name}] {row}", flush=True)
        if mv["macro_f1"] > best["macro_f1"]:
            best, bad_epochs = {"macro_f1": mv["macro_f1"], "epoch": epoch}, 0
            torch.save(model.state_dict(), out_dir / "best.pt")
        else:
            bad_epochs += 1
            if bad_epochs >= hp["patience"]:
                print(f"[{name}] early stop at epoch {epoch}")
                break

    model.load_state_dict(torch.load(out_dir / "best.pt", map_location=device))
    report = evaluate_and_export(name, exclude_sources, model, base, m, crops, classes, out_dir, hp, device,
                                 eval_loader, history, best)
    report["train_weight_share"] = {f"{c}/{src}": float(v) for (c, src), v in shares.stack().items()}
    (out_dir / "metrics.json").write_text(json.dumps(report, indent=2, default=float))
    return report


# ---------------------------------------------------------------- evaluation + export

def evaluate_and_export(name, exclude_sources, model, base, m, crops, classes, out_dir, hp, device, eval_loader,
                        history, best) -> dict:
    from sklearn.metrics import roc_auc_score

    keep = ~m["source"].isin(exclude_sources)
    sets = {
        "val": m[(m["split"] == "val") & keep & m["is_group_rep"]],
        "test": m[(m["split"] == "test") & m["is_group_rep"]],  # all sources, also the excluded one
        "test_rocole": m[m["split"] == "test_rocole"],
        "ood": m[m["split"] == "ood"],
    }
    logits, frames = {}, {}
    for s, df in sets.items():
        df = df.reset_index(drop=True)
        y, z = predict_logits(model, eval_loader(df), device)
        logits[s], frames[s] = (y, z), df
    y_leaf, z_leaf = predict_logits(model, eval_loader(frames["test_rocole"], view="leaf"), device)

    temp = fit_temperature(logits["val"][1], logits["val"][0])
    cal = {s: softmax(z / temp) for s, (_, z) in logits.items()}
    raw = {s: softmax(z) for s, (_, z) in logits.items()}
    sweep = threshold_sweep(logits["val"][0], cal["val"], np.round(np.arange(0.5, 0.99, 0.025), 3))
    ok = [r for r in sweep if r["accuracy_on_accepted"] is not None and r["accuracy_on_accepted"] >= hp["target_precision"]]
    threshold = ok[0]["threshold"] if ok else 0.9

    report = {"variant": name, "excluded_sources": exclude_sources, "model": hp["model"], "best": best,
              "history": history, "temperature": temp, "threshold": threshold,
              "threshold_rule": f"lowest confidence with >= {hp['target_precision']:.0%} val accuracy on answered images",
              "sets": {}}
    for s in ("val", "test", "test_rocole"):
        y = logits[s][0]
        mt = compute_metrics(y, cal[s], classes)
        conf = cal[s].max(1)
        answered = conf >= threshold
        mt.update(ece_raw=expected_calibration_error(y, raw[s])[0], ece_calibrated=expected_calibration_error(y, cal[s])[0],
                  answered_share=float(answered.mean()),
                  accuracy_when_answered=float((cal[s].argmax(1)[answered] == y[answered]).mean()) if answered.any() else None)
        report["sets"][s] = mt
        plot_confusion_matrix(mt["confusion_matrix"], classes, out_dir / f"cm_{s}.png", title=f"{name}: {s}")
        plot_reliability(y, cal[s], out_dir / f"reliability_{s}.png", title=f"{name}: {s}")
    p_leaf = softmax(z_leaf / temp)
    mt = compute_metrics(y_leaf, p_leaf, classes)
    ans = p_leaf.max(1) >= threshold
    mt.update(answered_share=float(ans.mean()),
              accuracy_when_answered=float((p_leaf.argmax(1)[ans] == y_leaf[ans]).mean()) if ans.any() else None)
    report["sets"]["test_rocole_leafbox"] = mt
    # per-source test results
    report["test_by_source"] = {}
    for src, idx in frames["test"].groupby("source").groups.items():
        idx = np.asarray(list(idx))
        report["test_by_source"][src] = compute_metrics(logits["test"][0][idx], cal["test"][idx], classes)
    # OOD: how well confidence separates in-scope images from out-of-scope ones
    id_conf = np.concatenate([cal["test"].max(1), cal["test_rocole"].max(1)])
    ood_conf = cal["ood"].max(1)
    rust_i = classes.index("rust")
    ood_pred = cal["ood"].argmax(1)

    def ood_stats(idx):
        rejected = ood_conf[idx] < threshold
        return {"n": int(len(idx)),
                "rejected_not_sure": float(rejected.mean()),                                 # shown as "not sure"
                "argmax_rust": float((ood_pred[idx] == rust_i).mean()),                       # top class, any confidence
                "answered_as_rust": float(((ood_pred[idx] == rust_i) & ~rejected).mean()),    # shown to the user as rust
                "answered_as_healthy": float(((ood_pred[idx] != rust_i) & ~rejected).mean())}

    groups = frames["ood"].groupby(frames["ood"]["ood_type"] + "/" + frames["ood"]["source"]).indices
    report["ood"] = {
        "auroc_in_vs_out": float(roc_auc_score(np.r_[np.ones(len(id_conf)), np.zeros(len(ood_conf))],
                                               np.r_[id_conf, ood_conf])),
        "all": ood_stats(np.arange(len(ood_conf))),
        "by_type": {t: ood_stats(np.asarray(i)) for t, i in groups.items()},
    }
    pd.DataFrame({"processed_path": frames["test_rocole"]["processed_path"], "true": frames["test_rocole"]["task_label"],
                  "pred": [classes[i] for i in cal["test_rocole"].argmax(1)], "conf": cal["test_rocole"].max(1)}
                 ).to_csv(out_dir / "predictions_test_rocole.csv", index=False)

    # export: temperature baked into the logits, so the backend's softmax gives calibrated confidence
    class Calibrated(nn.Module):
        def __init__(self, net, t):
            super().__init__()
            self.net, self.t = net, t

        def forward(self, x):
            return self.net(x) / self.t

    wrapped = Calibrated(model.eval().cpu(), temp)
    export_onnx(wrapped, out_dir / "model.onnx", hp["size"], 18)
    report["onnx_relative_diff_random"] = verify_onnx(wrapped, out_dir / "model.onnx", hp["size"])
    report["onnx_real_images"] = verify_onnx_real(wrapped, out_dir / "model.onnx", eval_loader,
                                                  pd.concat([frames["test"], frames["test_rocole"]]).head(256))
    (out_dir / "labels.json").write_text(json.dumps(classes))
    (out_dir / "preprocess.json").write_text(json.dumps({
        "resize_shorter": hp["resize"], "crop_size": hp["size"], "interpolation": "bilinear",
        "mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225], "color": "RGB", "layout": "NCHW",
        "scale": "0-1", "output": "logits (temperature-scaled)", "temperature": temp,
        "suggested_threshold": threshold}, indent=2))
    (out_dir / "metrics.json").write_text(json.dumps(report, indent=2, default=float))
    write_report(report, out_dir / "REPORT.md")
    (out_dir / "best.pt").unlink(missing_ok=True)  # keep outputs small; model.onnx is the deliverable
    model.to(device)
    return report


def verify_onnx_real(model, onnx_path: Path, eval_loader, df: pd.DataFrame) -> dict:
    """PyTorch vs onnxruntime on real images through the deployed preprocessing. Raises on any disagreement."""
    import onnxruntime as ort

    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    agree, worst, n = 0, 0.0, 0
    model = model.eval().cpu()
    with torch.no_grad():
        for x, _ in eval_loader(df.reset_index(drop=True)):
            pt = softmax(model(x).numpy())
            ox = softmax(sess.run(None, {sess.get_inputs()[0].name: x.numpy()})[0])
            agree += int((pt.argmax(1) == ox.argmax(1)).sum())
            worst = max(worst, float(np.abs(pt - ox).max()))
            n += len(x)
    if agree != n or worst > 1e-3:
        raise RuntimeError(f"ONNX disagrees with PyTorch on real images: {agree}/{n} same, max prob diff {worst:.2e}")
    return {"images": n, "same_prediction": agree, "max_probability_difference": worst}


def write_report(r: dict, path: Path) -> None:
    s = r["sets"]
    lines = [f"# Phase 1 variant `{r['variant']}`", "",
             f"Model `{r['model']}`, best epoch {r['best']['epoch']} (selected by val macro F1). "
             f"Excluded from training: {r['excluded_sources'] or 'none'}.", "",
             "**Calibration and threshold were derived from the validation set only** (temperature "
             f"T = {r['temperature']:.3f}; threshold {r['threshold']} = {r['threshold_rule']}). They are estimates; "
             "calibration on other data, including RoCoLe and real user photos, is not guaranteed.", "",
             "| set | n | accuracy | macro F1 | answered | accuracy when answered |", "|---|---|---|---|---|---|"]
    for k in ("val", "test", "test_rocole", "test_rocole_leafbox"):
        v = s[k]
        awa = v.get("accuracy_when_answered")
        lines.append(f"| {k} | {v['n']} | {v['accuracy']:.3f} | {v['macro_f1']:.3f} | "
                     f"{v.get('answered_share', float('nan')):.3f} | {'n/a' if awa is None else round(awa, 3)} |")
    lines += ["", "Test by source:", "", "| source | n | accuracy | macro F1 |", "|---|---|---|---|"]
    lines += [f"| {k} | {v['n']} | {v['accuracy']:.3f} | {v['macro_f1']:.3f} |" for k, v in r["test_by_source"].items()]
    o = r["ood"]
    lines += ["", f"OOD (never trained on, not used for any choice): AUROC in-vs-out {o['auroc_in_vs_out']:.3f}", "",
              "| OOD group | n | rejected (not sure) | answered as rust | argmax rust (any confidence) |",
              "|---|---|---|---|---|"]
    lines += [f"| {k} | {v['n']} | {v['rejected_not_sure']:.3f} | {v['answered_as_rust']:.3f} | {v['argmax_rust']:.3f} |"
              for k, v in {"ALL": o["all"], **o["by_type"]}.items()]
    oi = r["onnx_real_images"]
    lines += ["", f"ONNX vs PyTorch on {oi['images']} real images: {oi['same_prediction']} identical predictions, "
              f"max probability difference {oi['max_probability_difference']:.1e}."]
    path.write_text("\n".join(lines), encoding="utf-8")


def summary_table(reports: list[dict]) -> pd.DataFrame:
    rows = []
    for r in reports:
        s = r["sets"]
        rows.append({
            "variant": r["variant"], "best_epoch": r["best"]["epoch"],
            "val_macro_f1": s["val"]["macro_f1"], "test_macro_f1": s["test"]["macro_f1"],
            "rocole_acc": s["test_rocole"]["accuracy"], "rocole_macro_f1": s["test_rocole"]["macro_f1"],
            "rocole_leafbox_macro_f1": s["test_rocole_leafbox"]["macro_f1"],
            "rocole_ece": s["test_rocole"]["ece_calibrated"], "rocole_answered": s["test_rocole"]["answered_share"],
            "rocole_acc_when_answered": s["test_rocole"]["accuracy_when_answered"],
            "ood_auroc": r["ood"]["auroc_in_vs_out"], "ood_rejected": r["ood"]["all"]["rejected_not_sure"],
            "ood_answered_as_rust": r["ood"]["all"]["answered_as_rust"],
            "threshold": r["threshold"], "temperature": r["temperature"],
        })
    return pd.DataFrame(rows).set_index("variant").round(3)

# %% [markdown]
# ## Data checks

# %%
base, manifest, crops = load_tables(ROOT)
CLASSES = ["healthy", "rust"]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device:", DEVICE, torch.cuda.get_device_name(0) if DEVICE.type == "cuda" else "")

fit_splits = ["train", "val"]
assert not manifest.loc[manifest.split.isin(["train", "val", "test"]), "source"].eq("rocole").any(), "RoCoLe in train/val/test"
assert manifest.loc[manifest.split.isin(fit_splits), "ood_type"].eq("").all(), "OOD rows in train/val"
assert set(manifest.loc[manifest.split.isin(fit_splits), "task_label"]) == set(CLASSES)
assert manifest.loc[manifest.split.eq("train"), "balanced_weight"].notna().all()
print(manifest.groupby(["split", "source", "task_label"]).size().unstack(fill_value=0))
for name, excluded in {"all_sources": [], "no_jmuben": ["jmuben"]}.items():
    tr = manifest[manifest.split.eq("train") & ~manifest.source.isin(excluded)]
    print(f"\n{name}: share of sampling weight (balanced_weight as stored)")
    print(class_source_shares(tr))

# %%
HP = dict(
    model="convnext_tiny.fb_in22k_ft_in1k", pretrained=True,
    size=224, resize=256, batch_size=32, workers=4,
    epochs=20, patience=5, epoch_size=3000,           # images drawn per epoch (weighted, with replacement)
    lr=1e-4, weight_decay=0.05, label_smoothing=0.1, seed=42,
    target_precision=0.95,                            # "not sure" threshold: >= 95% val accuracy on answered images
)
VARIANTS = {"all_sources": [], "no_jmuben": ["jmuben"]}

# COFFEE_SMOKE=1: tiny CPU run to check the notebook end to end (not a real training run)
if os.environ.get("COFFEE_SMOKE") == "1":
    HP.update(pretrained=False, model="convnext_atto", epochs=1, epoch_size=64, batch_size=8, workers=0)
    manifest = pd.concat([g.sample(frac=1, random_state=0).groupby(["source", "task_label", "ood_type"]).head(6)
                          for _, g in manifest.groupby("split")])
print(HP)

# %% [markdown]
# ## Train and evaluate both variants

# %%
reports = []
for name, excluded in VARIANTS.items():
    print(f"\n===== {name} (excluded from training: {excluded or 'none'}) =====")
    reports.append(train_variant(name, excluded, base, manifest, crops, CLASSES, OUT / name, HP, DEVICE))

# %% [markdown]
# ## Comparison and variant choice (validation macro F1 only)

# %%
summary = summary_table(reports)
summary.to_csv(OUT / "summary.csv")
(OUT / "hparams.json").write_text(json.dumps(HP, indent=2))
preferred = summary["val_macro_f1"].idxmax()
(OUT / "selection.json").write_text(json.dumps({
    "preferred_variant": preferred, "rule": "highest validation macro F1",
    "val_macro_f1": summary["val_macro_f1"].to_dict(),
    "note": "test_rocole and ood were not used for this choice; their numbers are information only"}, indent=2))
print("Preferred variant (by validation macro F1):", preferred)
for r in reports:
    print(f"\n----- {r['variant']} -----")
    print((OUT / r["variant"] / "REPORT.md").read_text())
summary

# %% [markdown]
# ## Outputs (`/kaggle/working/phase1/`)
# - `<variant>/model.onnx` (temperature already applied), `labels.json`, `preprocess.json` (with `temperature`
#   and `suggested_threshold`), `metrics.json`, `REPORT.md`, confusion matrices, reliability plots,
#   `predictions_test_rocole.csv`
# - `summary.csv`, `selection.json`, `hparams.json`
