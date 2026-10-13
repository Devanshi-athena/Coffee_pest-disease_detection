"""Phase 2 (healthy / rust / other disease) training, evaluation, "not sure" detector and export.

Used by ml/notebooks/train_phase2.ipynb. Data: data/processed_phase2/ (ml/scripts/08_build_phase2_dataset.py).

Rules enforced here:
  - model, epoch and seed choices use the validation split only (worst per-source balanced accuracy)
  - temperature, confidence threshold and OOD threshold are fitted on validation only
  - `test_locked` (RoCoLe farms) and `ood` are only ever evaluated, never used for a choice
"""
from __future__ import annotations

import io
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

from .crops import apply_view
from .evaluate import compute_metrics, expected_calibration_error, plot_confusion_matrix, plot_reliability

DATA_SUBDIR = Path("data") / "processed_phase2"
CLASSES = ["healthy", "rust", "other"]
FULL = (0.0, 0.0, 1.0, 1.0)
SOURCE_FAMILY = {"jmuben": "jmuben", "jmuben2": "jmuben", "uganda": "uganda", "bracol": "bracol", "rocole": "rocole"}


# ---------------------------------------------------------------- data

def load_tables(root: Path) -> tuple[Path, pd.DataFrame, pd.DataFrame]:
    base = root / DATA_SUBDIR
    m = pd.read_csv(base / "manifest.csv", low_memory=False)
    m = m[m["processed_path"].notna() & m["split"].notna()].copy()
    for col in ("task_label", "ood_type"):
        m[col] = m[col].fillna("")
    return base, m, pd.read_csv(base / "crops.csv")


def sampling_weights(tr: pd.DataFrame, n_classes: int) -> np.ndarray:
    """Per-image sampling probability: every class gets the same total, every source the same share within a
    class, and inside a class/source cell the stored balanced_weight ratios (near-duplicate correction) are kept.
    Recomputed for the rows actually used, so it stays balanced when a source is left out."""
    if tr["balanced_weight"].isna().any() or (tr["balanced_weight"] <= 0).any():
        raise ValueError("balanced_weight missing or non-positive for some training rows")
    w = tr["balanced_weight"].astype(float)
    cell = w.groupby([tr["task_label"], tr["source"]]).transform("sum")
    n_src = tr.groupby("task_label")["source"].transform("nunique")
    w = w / cell / n_src / n_classes
    return (w / w.sum()).to_numpy()


def sampling_shares(tr: pd.DataFrame, p: np.ndarray) -> pd.Series:
    s = pd.Series(p).groupby([tr["task_label"].to_numpy(), tr["source"].to_numpy()]).sum()
    s.index = [f"{c}/{x}" for c, x in s.index]
    return s.sort_index()


def check_balanced(shares: pd.Series, classes: list[str], tol: float = 0.005) -> None:
    """Each class must hold 1/C of the sampling mass, split equally between its sources."""
    for c in classes:
        cells = shares[[k for k in shares.index if k.startswith(c + "/")]]
        if len(cells) == 0:
            raise AssertionError(f"class {c} has no training rows")
        if abs(cells.sum() - 1 / len(classes)) > tol or (cells - cells.mean()).abs().max() > tol:
            raise AssertionError(f"sampling for class {c} is not balanced: {cells.round(4).to_dict()}")


def views_by_path(crops: pd.DataFrame, training: bool, view: str = "full") -> dict[str, list[tuple]]:
    sel = crops[crops["use_for_training"]] if training else crops[crops["view"] == view]
    out: dict[str, list[tuple]] = {}
    for r in sel.itertuples():
        out.setdefault(r.processed_path, []).append((r.x0, r.y0, r.x1, r.y1))
    return out


class ViewDataset(Dataset):
    """One item per image; in training a random one of its precomputed views is used each time."""

    def __init__(self, base, df, views, classes, transform, train):
        self.base, self.transform, self.train = base, transform, train
        self.paths = df["processed_path"].tolist()
        self.labels = [classes.index(x) if x in classes else -1 for x in df["task_label"]]
        self.views = [views.get(p, [FULL]) for p in self.paths]

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        with Image.open(self.base / self.paths[i]) as im:
            img = im.convert("RGB")
        boxes = self.views[i]
        img = apply_view(img, random.choice(boxes) if self.train else boxes[0])
        return self.transform(img), self.labels[i]


class RandomJpeg:
    """Re-encode with a random JPEG quality (phone uploads are often heavily compressed)."""

    def __init__(self, p=0.3, quality=(30, 90)):
        self.p, self.quality = p, quality

    def __call__(self, img):
        if random.random() >= self.p:
            return img
        buf = io.BytesIO()
        img.save(buf, "JPEG", quality=random.randint(*self.quality))
        buf.seek(0)
        return Image.open(buf).convert("RGB")


class RandomDownscale:
    """Shrink and re-enlarge (low-resolution or distant photos)."""

    def __init__(self, p=0.3, factor=(0.35, 0.9)):
        self.p, self.factor = p, factor

    def __call__(self, img):
        if random.random() >= self.p:
            return img
        f = random.uniform(*self.factor)
        w, h = img.size
        return img.resize((max(16, int(w * f)), max(16, int(h * f))), Image.BILINEAR).resize((w, h), Image.BILINEAR)


def transforms(size: int = 224, resize: int = 256):
    norm = T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    train = T.Compose([
        RandomDownscale(), RandomJpeg(),
        T.RandomResizedCrop(size, scale=(0.5, 1.0), ratio=(0.75, 1.333)),
        T.RandomHorizontalFlip(), T.RandomVerticalFlip(),
        T.RandomApply([T.RandomRotation(30)], p=0.5),
        T.ColorJitter(brightness=0.35, contrast=0.35, saturation=0.3, hue=0.03),
        T.RandomApply([T.GaussianBlur(5, sigma=(0.1, 2.0))], p=0.25),
        T.ToTensor(), norm,
    ])
    # Must match backend/app/inference.py: resize shorter side (bilinear) + center crop.
    evaluate = T.Compose([T.Resize(resize, interpolation=T.InterpolationMode.BILINEAR), T.CenterCrop(size),
                          T.ToTensor(), norm])
    return train, evaluate


# ---------------------------------------------------------------- model

class Net(nn.Module):
    """timm backbone returning (logits, L2-normalised penultimate features)."""

    def __init__(self, name: str, pretrained: bool, num_classes: int, drop_path: float = 0.1):
        super().__init__()
        import timm

        self.backbone = timm.create_model(name, pretrained=pretrained, num_classes=num_classes,
                                          drop_path_rate=drop_path)

    def forward(self, x):
        f = self.backbone.forward_head(self.backbone.forward_features(x), pre_logits=True)
        return self.backbone.get_classifier()(f), nn.functional.normalize(f, dim=1)


@torch.no_grad()
def predict(model, loader, device) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    ys, zs, fs = [], [], []
    for x, y in loader:
        z, f = model(x.to(device, non_blocking=True))
        zs.append(z.float().cpu().numpy())
        fs.append(f.float().cpu().numpy())
        ys.append(np.asarray(y))
    return np.concatenate(ys), np.concatenate(zs), np.concatenate(fs)


def softmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


def balanced_accuracy_by_source(y: np.ndarray, pred: np.ndarray, sources: np.ndarray) -> dict[str, float]:
    """Mean recall over the classes present, per source family (works for single-class sources too)."""
    out = {}
    fam = np.array([SOURCE_FAMILY.get(s, s) for s in sources])
    for s in sorted(set(fam)):
        m = fam == s
        recalls = [float((pred[m & (y == c)] == c).mean()) for c in sorted(set(y[m]))]
        out[s] = float(np.mean(recalls))
    return out


def val_score(y, pred, sources) -> tuple[float, dict]:
    per = balanced_accuracy_by_source(y, pred, sources)
    return min(per.values()), per


# ---------------------------------------------------------------- training

def make_loaders(base, crops, classes, hp, tr, tf_train):
    w = sampling_weights(tr, len(classes))
    train_ds = ViewDataset(base, tr, views_by_path(crops, training=True), classes, tf_train, train=True)
    sampler = WeightedRandomSampler(torch.as_tensor(w, dtype=torch.double), num_samples=hp["epoch_size"],
                                    replacement=True)
    train_dl = DataLoader(train_ds, batch_size=hp["batch_size"], sampler=sampler, num_workers=hp["workers"],
                          pin_memory=True, drop_last=True)
    return w, train_dl


def eval_loader(base, crops, classes, hp, df, view="full", tf=None):
    ds = ViewDataset(base, df.reset_index(drop=True), views_by_path(crops, training=False, view=view), classes, tf,
                     train=False)
    return DataLoader(ds, batch_size=hp["batch_size"] * 2, shuffle=False, num_workers=hp["workers"])


def train_model(tag: str, tr: pd.DataFrame, va: pd.DataFrame, base, crops, classes, hp, device, seed: int):
    """Trains one model; keeps the EMA weights of the epoch with the best validation score."""
    from timm.utils import ModelEmaV3

    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf_train, tf_eval = transforms(hp["size"], hp["resize"])
    w, train_dl = make_loaders(base, crops, classes, hp, tr, tf_train)
    shares = sampling_shares(tr, w)
    check_balanced(shares, classes, tol=hp.get("share_tolerance", 0.005))
    val_dl = eval_loader(base, crops, classes, hp, va, tf=tf_eval)
    va_sources = va["source"].to_numpy()

    model = Net(hp["model"], hp["pretrained"], len(classes)).to(device)
    ema = ModelEmaV3(model, decay=hp["ema_decay"], use_warmup=True)
    opt = torch.optim.AdamW(model.parameters(), lr=hp["lr"], weight_decay=hp["weight_decay"])
    steps, warm = hp["epochs"] * len(train_dl), len(train_dl)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / warm if s < warm else 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, steps - warm))))
    crit = nn.CrossEntropyLoss(label_smoothing=hp["label_smoothing"])
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    history, best, bad, step = [], {"score": -1.0, "epoch": 0}, 0, 0
    best_state = None
    for epoch in range(1, hp["epochs"] + 1):
        model.train()
        t0, total, n = time.time(), 0.0, 0
        for x, y in train_dl:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=use_amp):
                loss = crit(model(x)[0], y)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            step += 1
            ema.update(model, step=step)
            total, n = total + loss.item() * len(y), n + len(y)
        yv, zv, _ = predict(ema.module, val_dl, device)
        score, per = val_score(yv, zv.argmax(1), va_sources)
        row = {"epoch": epoch, "train_loss": round(total / n, 4), "val_worst_source_bal_acc": round(score, 4),
               "val_per_source": {k: round(v, 3) for k, v in per.items()}, "sec": round(time.time() - t0)}
        history.append(row)
        print(f"[{tag}] {row}", flush=True)
        if score > best["score"]:
            best, bad = {"score": score, "epoch": epoch}, 0
            best_state = {k: v.detach().clone() for k, v in ema.module.state_dict().items()}
        else:
            bad += 1
            if bad >= hp["patience"]:
                print(f"[{tag}] early stop at epoch {epoch}")
                break
    ema.module.load_state_dict(best_state)
    return ema.module.eval(), history, best, shares


# ---------------------------------------------------------------- calibration + "not sure"

def fit_temperature_bounded(logits: np.ndarray, y: np.ndarray, lo: float = 0.5, hi: float = 5.0) -> float:
    from scipy.optimize import minimize_scalar
    from scipy.special import log_softmax

    def nll(t):
        return -float(log_softmax(logits / t, axis=1)[np.arange(len(y)), y].mean())

    return float(minimize_scalar(nll, bounds=(lo, hi), method="bounded").x)


def knn_ood_score(feats: np.ndarray, bank: np.ndarray, k: int = 10) -> np.ndarray:
    """1 - mean cosine similarity to the k nearest training features (higher = less like the training data)."""
    sims = []
    for s in range(0, len(feats), 512):
        sim = feats[s:s + 512] @ bank.T
        sims.append(np.sort(sim, axis=1)[:, -k:].mean(1))
    return 1.0 - np.concatenate(sims)


def choose_thresholds(y_val, p_val, ood_val, hp) -> dict:
    """OOD threshold: the val score at the given quantile (in-scope images above it are rejected).
    Confidence threshold: lowest value giving >= target accuracy on the val images that pass both checks."""
    t_ood = float(np.quantile(ood_val, hp["ood_val_quantile"]))
    keep = ood_val <= t_ood
    conf, pred = p_val.max(1), p_val.argmax(1)
    t_conf = None
    for t in np.round(np.arange(1 / len(CLASSES), 0.99, 0.01), 3):
        ans = keep & (conf >= t)
        if ans.sum() >= 20 and (pred[ans] == y_val[ans]).mean() >= hp["target_accuracy"]:
            t_conf = float(t)
            break
    if t_conf is None:
        t_conf = 0.99
    return {"ood": t_ood, "confidence": t_conf}


def decide(p: np.ndarray, ood: np.ndarray, th: dict) -> np.ndarray:
    """Answered = confident enough and close enough to the training data."""
    return (p.max(1) >= th["confidence"]) & (ood <= th["ood"])


def selective_report(y, p, ood, th, classes) -> dict:
    ans = decide(p, ood, th)
    pred = p.argmax(1)
    out = {"n": int(len(y)), "answered_share": float(ans.mean()),
           "accuracy_when_answered": float((pred[ans] == y[ans]).mean()) if ans.any() else None}
    if "other" in classes and "rust" in classes:
        o, r = classes.index("other"), classes.index("rust")
        if (y == o).any():
            out["other_answered_as_rust"] = float((ans & (y == o) & (pred == r)).sum() / (y == o).sum())
    return out


# ---------------------------------------------------------------- export

class Exported(nn.Module):
    """ONNX graph: temperature-scaled logits + normalised features (for the "not sure" distance check)."""

    def __init__(self, net, temperature):
        super().__init__()
        self.net, self.t = net, temperature

    def forward(self, x):
        z, f = self.net(x)
        return z / self.t, f


def export_and_verify(model, path: Path, size: int, loader) -> dict:
    import onnxruntime as ort

    model = model.eval().cpu()
    dummy = torch.randn(2, 3, size, size)  # batch 2: a size-1 dim would be specialised by torch.export
    batch = torch.export.Dim("batch", min=1, max=1024)
    torch.onnx.export(model, (dummy,), str(path), input_names=["input"], output_names=["logits", "features"],
                      dynamic_shapes=({0: batch},), dynamo=True, external_data=False, opset_version=18)
    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    agree, worst_p, worst_f, n = 0, 0.0, 0.0, 0
    with torch.no_grad():
        for x, _ in loader:
            zt, ft = model(x)
            zo, fo = sess.run(None, {"input": x.numpy()})
            pt, po = softmax(zt.numpy()), softmax(zo)
            agree += int((pt.argmax(1) == po.argmax(1)).sum())
            worst_p = max(worst_p, float(np.abs(pt - po).max()))
            worst_f = max(worst_f, float(np.abs(ft.numpy() - fo).max()))
            n += len(x)
        x1 = next(iter(loader))[0][:1]  # the backend sends batch 1
        z1, _ = model(x1)
        o1 = sess.run(None, {"input": x1.numpy()})[0]
        worst_p = max(worst_p, float(np.abs(softmax(z1.numpy()) - softmax(o1)).max()))
    if agree != n or worst_p > 1e-3 or worst_f > 1e-3:
        raise RuntimeError(f"ONNX disagrees with PyTorch: {agree}/{n} same, prob diff {worst_p:.2e}, "
                           f"feature diff {worst_f:.2e}")
    return {"images": n, "same_prediction": agree, "max_probability_difference": worst_p,
            "max_feature_difference": worst_f}


# ---------------------------------------------------------------- evaluation helpers

def metrics_block(y, p, classes, sources=None) -> dict:
    mt = compute_metrics(y, p, classes)
    mt["ece"] = expected_calibration_error(y, p)[0]
    if sources is not None:
        mt["by_source"] = {s: compute_metrics(y[sources == s], p[sources == s], classes)
                           for s in sorted(set(sources))}
    return mt


def criteria_table(results: dict, hp: dict) -> pd.DataFrame:
    c = hp["criteria"]
    lk, test, ood = results["test_locked"], results["test"], results["ood"]
    rows = [
        ("Healthy vs rust macro F1 on locked RoCoLe farms (full photo)", lk["macro_f1"], ">=", c["locked_macro_f1"]),
        ("Accuracy on answered photos, locked RoCoLe farms", lk["selective"]["accuracy_when_answered"], ">=",
         c["answered_accuracy"]),
        ("Share of locked RoCoLe photos answered", lk["selective"]["answered_share"], ">=", c["answered_share"]),
        ("Other-disease test photos answered as rust", test["selective"].get("other_answered_as_rust"), "<=",
         c["other_as_rust"]),
        ("Out-of-scope photos flagged 'not sure'", ood["rejected_share"], ">=", c["ood_rejected"]),
        ("Calibration error (ECE) on validation", results["val"]["ece"], "<=", c["val_ece"]),
    ]
    df = pd.DataFrame(rows, columns=["criterion", "value", "rule", "target"])
    df["pass"] = [(v is not None) and (v >= t if r == ">=" else v <= t) for _, v, r, t in rows]
    return df
