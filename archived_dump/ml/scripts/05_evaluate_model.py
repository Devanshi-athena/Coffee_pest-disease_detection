"""Step 5: evaluate the model in backend/models exactly as the backend runs it (ONNX + backend preprocessing).

Needs artifacts/inspect/images.csv and near_dup_pairs.csv from 01_inspect_raw.py. One image per
byte-identical group is used; groups whose copies carry different labels are dropped.

Evaluation sets:
  seen_jmuben     JMuBEN + JMuBEN2. The HF baseline was TRAINED on these: a sanity check, not validation.
  uganda_clean    Uganda images with no near-duplicate in JMuBEN            (external)
  uganda_overlap  Uganda images that near-duplicate a JMuBEN image          (may have been seen in training)
  bracol          BRACOL whole-leaf photos                                  (external)
  rocole          RoCoLe photos, healthy + rust only                        (external)
  out_of_scope    RoCoLe spider mite + BRACOL stress code 5: no valid class; tests the "uncertain" flag

Framing strategies (whole-leaf sets): center = deployed (resize + center crop), squash = whole image
resized to the input size, tiles = mean probability over a grid of input-size tiles.

Writes to artifacts/eval_<model>/: report.md, metrics.json, predictions.csv, plots/, examples/
"""
import argparse
import io
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "ml"))
sys.path.insert(0, str(ROOT / "backend"))
from src.common import die, load_config, resolve_path  # noqa: E402

TRAIN_SOURCES = ("jmuben", "jmuben2")  # what the HF baseline was trained on
WHOLE_LEAF_SETS = ("bracol", "rocole", "out_of_scope")
LABELLED_SETS = ("seen_jmuben", "uganda_clean", "uganda_overlap", "bracol", "rocole")
EXTERNAL_SETS = ("uganda_clean", "bracol", "rocole")


# ---------------------------------------------------------------- data

def build_sets(inspect_dir: Path) -> pd.DataFrame:
    images, pairs_csv = inspect_dir / "images.csv", inspect_dir / "near_dup_pairs.csv"
    if not (images.is_file() and pairs_csv.is_file()):
        die(f"{images} / {pairs_csv} not found. Run ml/scripts/01_inspect_raw.py first.")
    im = pd.read_csv(images, low_memory=False)
    im = im[im["status"] == "ok"]
    n_labels = im.groupby("md5")["label"].nunique()
    conflicting = n_labels[n_labels > 1].index
    im = im[~im["md5"].isin(conflicting)].drop_duplicates("md5").copy()

    pairs = pd.read_csv(pairs_csv, usecols=["path_a", "path_b"])
    src_a, src_b = pairs["path_a"].str.split("/").str[0], pairs["path_b"].str.split("/").str[0]
    linked = set(pairs.loc[src_a.isin(TRAIN_SOURCES) & src_b.eq("uganda"), "path_b"]) | \
        set(pairs.loc[src_b.isin(TRAIN_SOURCES) & src_a.eq("uganda"), "path_a"])

    skip = im["label"].str.startswith("SKIP")
    im["set"] = np.select(
        [im["source"].isin(TRAIN_SOURCES),
         im["source"].eq("uganda") & im["path"].isin(linked),
         im["source"].eq("uganda"),
         skip & im["source"].isin(["bracol", "rocole"]),
         im["source"].eq("bracol"),
         im["source"].eq("rocole")],
        ["seen_jmuben", "uganda_overlap", "uganda_clean", "out_of_scope", "bracol", "rocole"], "")
    im = im[im["set"] != ""]
    print(f"dropped {len(conflicting)} byte-identical groups with conflicting labels")
    return im


# ---------------------------------------------------------------- model

class OnnxModel:
    def __init__(self, model_dir: Path):
        import onnxruntime as ort

        from app.inference import PreprocessSpec

        for f in ("model.onnx", "labels.json", "preprocess.json"):
            if not (model_dir / f).is_file():
                die(f"{model_dir / f} missing. Run ml/scripts/import_hf_baseline.py (or add a trained model).")
        self.labels = json.loads((model_dir / "labels.json").read_text(encoding="utf-8"))
        self.spec = PreprocessSpec.from_dict(json.loads((model_dir / "preprocess.json").read_text(encoding="utf-8")))
        # a spec whose resize is a no-op, for inputs that are already crop_size x crop_size
        self.spec_exact = PreprocessSpec(self.spec.crop_size, self.spec.crop_size, self.spec.mean, self.spec.std,
                                         self.spec.interpolation)
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 1  # parallelism comes from the thread pool
        self.session = ort.InferenceSession(str(model_dir / "model.onnx"), opts, providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name

    def logits(self, x: np.ndarray) -> np.ndarray:
        return self.session.run(None, {self.input_name: x})[0]

    def center(self, img) -> np.ndarray:
        from app.inference import preprocess

        return preprocess(img, self.spec)

    def squash(self, img) -> np.ndarray:
        from PIL import Image

        from app.inference import preprocess

        s = self.spec.crop_size
        return preprocess(img.resize((s, s), Image.Resampling.BICUBIC), self.spec_exact)

    def tiles(self, img) -> np.ndarray:
        """Resize the shorter side to 2 tiles, then cut a grid of non-overlapping tiles."""
        from PIL import Image

        from app.inference import preprocess

        s = self.spec.crop_size
        scale = 2 * s / min(img.size)
        img = img.resize((max(2 * s, round(img.width * scale)), max(2 * s, round(img.height * scale))),
                         Image.Resampling.BICUBIC)
        nx, ny = img.width // s, img.height // s
        ox, oy = (img.width - nx * s) // 2, (img.height - ny * s) // 2
        crops = [img.crop((ox + i * s, oy + j * s, ox + (i + 1) * s, oy + (j + 1) * s))
                 for j in range(ny) for i in range(nx)]
        return np.concatenate([preprocess(c, self.spec_exact) for c in crops])


def softmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


def run_one(model: OnnxModel, path: Path, whole_leaf: bool) -> dict:
    from app.inference import load_image

    t0 = time.perf_counter()
    img = load_image(path.read_bytes())  # same decode + EXIF rotation as the API
    out = {"center": model.logits(model.center(img))[0]}
    out["latency_ms"] = (time.perf_counter() - t0) * 1000
    if whole_leaf:
        out["squash"] = model.logits(model.squash(img))[0]
        out["tiles_prob"] = softmax(model.logits(model.tiles(img))).mean(0)
    return out


# ---------------------------------------------------------------- perturbations

def perturbations():
    from PIL import ImageEnhance, ImageFilter

    def jpeg(img, q):
        buf = io.BytesIO()
        img.save(buf, "JPEG", quality=q)
        buf.seek(0)
        from PIL import Image
        return Image.open(buf).convert("RGB")

    def lowres(img, side):
        from PIL import Image
        small = img.copy()
        small.thumbnail((side, side), Image.Resampling.BILINEAR)
        return small.resize(img.size, Image.Resampling.BILINEAR)

    def color_cast(img, factors):
        from PIL import Image
        bands = [b.point(lambda v, f=f: min(255, int(v * f))) for b, f in zip(img.split(), factors)]
        return Image.merge("RGB", bands)

    def noise(img, sigma, rng=np.random.default_rng(0)):
        from PIL import Image
        a = np.asarray(img, dtype=np.float32) + rng.normal(0, sigma, (img.height, img.width, 3))
        return Image.fromarray(np.clip(a, 0, 255).astype(np.uint8))

    return {
        "clean": lambda im: im,
        "blur r=2": lambda im: im.filter(ImageFilter.GaussianBlur(2 * max(im.size) / 512)),
        "blur r=5": lambda im: im.filter(ImageFilter.GaussianBlur(5 * max(im.size) / 512)),
        "dark x0.5": lambda im: ImageEnhance.Brightness(im).enhance(0.5),
        "bright x1.6": lambda im: ImageEnhance.Brightness(im).enhance(1.6),
        "low contrast x0.5": lambda im: ImageEnhance.Contrast(im).enhance(0.5),
        "warm/yellow cast": lambda im: color_cast(im, (1.15, 1.05, 0.8)),
        "jpeg q=15": lambda im: jpeg(im, 15),
        "low-res 96px": lambda im: lowres(im, 96),
        "noise sigma=20": lambda im: noise(im, 20),
        "rotate 90": lambda im: im.rotate(90, expand=True),
    }


# ---------------------------------------------------------------- reporting helpers

def examples_sheet(df: pd.DataFrame, raw_dir: Path, out: Path, title: str) -> None:
    from src.viz import captioned_grid

    items = [(raw_dir / r.path, f"true: {r.true}\npred: {r.pred} ({r.conf:.2f})") for r in df.itertuples()]
    captioned_grid(items, out, title)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", help="path to config.yaml (default: configs/config.yaml)")
    ap.add_argument("--model-dir", type=Path, help="default: paths.model_dir")
    ap.add_argument("--name", default="baseline", help="output folder suffix: artifacts/eval_<name>")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--perturb-per-class", type=int, default=30, help="images per class per set for perturbations")
    ap.add_argument("--limit-per-set", type=int, help="random subset per set (trial runs)")
    args = ap.parse_args()

    from sklearn.metrics import f1_score

    from src.evaluate import (compute_metrics, expected_calibration_error, fit_temperature, plot_confusion_matrix,
                              plot_reliability, threshold_sweep)

    cfg = load_config(args.config)
    classes = cfg["classes"]
    threshold = cfg["inference"]["confidence_threshold"]
    raw_dir = resolve_path(cfg, "raw_dir")
    out_dir = resolve_path(cfg, "artifacts_dir") / f"eval_{args.name}"
    (out_dir / "plots").mkdir(parents=True, exist_ok=True)
    model = OnnxModel(args.model_dir or resolve_path(cfg, "model_dir"))
    perm = [model.labels.index(c) for c in classes]  # model output order -> config order
    cidx = {c: i for i, c in enumerate(classes)}

    data = build_sets(resolve_path(cfg, "artifacts_dir") / "inspect")
    if args.limit_per_set:
        data = data.sample(frac=1, random_state=0).groupby("set").head(args.limit_per_set)
    data = data.reset_index(drop=True)
    print(data.groupby("set").size().to_string())

    def work(row):
        return run_one(model, raw_dir / row.path, row.set in WHOLE_LEAF_SETS)

    t0 = time.time()
    with ThreadPoolExecutor(args.workers) as ex:
        results = list(ex.map(work, data.itertuples()))
    print(f"inference on {len(data)} images: {time.time() - t0:.0f}s")

    logits = {"center": np.stack([r["center"] for r in results])[:, perm]}
    probs = {"center": softmax(logits["center"])}
    wl = data["set"].isin(WHOLE_LEAF_SETS).to_numpy()  # squash/tiles only run on whole-leaf photos
    for strat, key in (("squash", "squash"), ("tiles", "tiles_prob")):
        arr = np.full((len(data), len(classes)), np.nan)
        if wl.any():
            vals = np.stack([r[key] for r, w in zip(results, wl) if w])[:, perm]
            arr[wl] = softmax(vals) if strat == "squash" else vals  # tiles are already probabilities
        probs[strat] = arr
    data["latency_ms"] = [r["latency_ms"] for r in results]
    data["true"] = data["label"].where(~data["label"].str.startswith("SKIP"), "")
    for s, p in probs.items():
        has = ~np.isnan(p).any(axis=1)
        data[f"pred_{s}"] = np.where(has, np.array(classes)[np.nan_to_num(p).argmax(1)], "")
        data[f"conf_{s}"] = np.where(has, np.nan_to_num(p).max(1), np.nan)
    for i, c in enumerate(classes):
        data[f"p_{c}"] = probs["center"][:, i]

    report = ["# Model evaluation\n\n",
              f"model: `{model.labels}` from `{args.model_dir or resolve_path(cfg, 'model_dir')}`  \n",
              f"preprocessing: `{model.spec}`  \n",
              f"uncertainty threshold (config): {threshold}\n"]
    metrics = {"threshold": threshold, "sets": {}}

    def section(title, table, note=""):
        text = table if isinstance(table, str) else table.to_string()
        print(f"\n== {title}\n{text}")
        report.append(f"\n## {title}\n\n{note}\n```\n{text}\n```\n")

    # 1. headline metrics per set (deployed center strategy)
    rows = []
    for name in LABELLED_SETS:
        m = data["set"].eq(name).to_numpy()
        if not m.any():
            continue
        y = data.loc[m, "true"].map(cidx).to_numpy()
        p = probs["center"][m]
        mt = compute_metrics(y, p, classes)
        ece, _ = expected_calibration_error(y, p)
        conf, correct = p.max(1), p.argmax(1) == y
        mt.update(ece=ece, mean_conf_correct=float(conf[correct].mean()) if correct.any() else None,
                  mean_conf_wrong=float(conf[~correct].mean()) if (~correct).any() else None,
                  uncertain_rate=float((conf < threshold).mean()),
                  accuracy_when_confident=float(correct[conf >= threshold].mean()) if (conf >= threshold).any() else None,
                  selective=threshold_sweep(y, p))
        metrics["sets"][name] = {"center": mt}
        rows.append({"set": name, "n": mt["n"], "accuracy": mt["accuracy"], "macro_f1": mt["macro_f1"],
                     "ECE": ece, "conf_correct": mt["mean_conf_correct"], "conf_wrong": mt["mean_conf_wrong"],
                     f"uncertain(<{threshold})": mt["uncertain_rate"],
                     "acc_if_confident": mt["accuracy_when_confident"]})
        plot_confusion_matrix(mt["confusion_matrix"], classes, out_dir / "plots" / f"cm_{name}.png", title=name)
        plot_reliability(y, p, out_dir / "plots" / f"reliability_{name}.png", title=name)
    head = pd.DataFrame(rows).set_index("set").round(3)
    section("Headline (deployed preprocessing: resize + center crop)", head,
            "macro_f1 covers the classes present in each set (rocole: healthy, rust). seen_jmuben is the model's "
            "own training source: a sanity check, not validation.")

    # 2. per-class detail and confusion matrices
    for name in LABELLED_SETS:
        if name not in metrics["sets"]:
            continue
        mt = metrics["sets"][name]["center"]
        pc = pd.DataFrame(mt["per_class"]).T[["precision", "recall", "f1-score", "support"]].round(3)
        pc["support"] = pc["support"].astype(int)
        cm = pd.DataFrame(mt["confusion_matrix"], index=[f"true {c}" for c in classes],
                          columns=[f"pred {c}" for c in classes])
        cm = cm.loc[(cm.sum(axis=1) > 0).to_numpy()]
        section(f"{name}: per class", pc.to_string() + "\n\nconfusion matrix (rows = true):\n" + cm.to_string())

    # 3. what the model says about out-of-scope images
    oos = data[data["set"] == "out_of_scope"]
    if len(oos):
        t = pd.crosstab(oos["folder"], oos["pred_center"])
        t["mean conf"] = oos.groupby("folder")["conf_center"].mean().round(3)
        t[f"flagged uncertain (<{threshold})"] = oos.groupby("folder")["conf_center"].apply(
            lambda c: f"{(c < threshold).mean():.0%}")
        section("Out-of-scope images (should ideally be flagged uncertain)", t)

    # 4. framing strategies on whole-leaf sets
    rows = []
    for name in ("bracol", "rocole"):
        m = data["set"].eq(name).to_numpy()
        if not m.any():
            continue
        y = data.loc[m, "true"].map(cidx).to_numpy()
        for strat in ("center", "squash", "tiles"):
            p = probs[strat][m]
            mt = compute_metrics(y, p, classes)
            if strat != "center":  # the center entry already holds the full headline metrics
                metrics["sets"][name][strat] = mt
            rows.append({"set": name, "strategy": strat, "accuracy": mt["accuracy"], "macro_f1": mt["macro_f1"],
                         "ECE": expected_calibration_error(y, p)[0], "mean_conf": float(p.max(1).mean())})
            if strat != "center":
                plot_confusion_matrix(mt["confusion_matrix"], classes, out_dir / "plots" / f"cm_{name}_{strat}.png",
                                      title=f"{name} ({strat})")
    if rows:
        section("Framing: crop vs whole image vs tiles (whole-leaf photos)",
                pd.DataFrame(rows).set_index(["set", "strategy"]).round(3),
                "center = deployed. squash = whole photo resized to the input size. tiles = mean over a grid "
                "of input-size tiles (closer to the close-up crops the model was trained on).")

    # 5. calibration: one temperature fitted on uganda_clean, applied elsewhere
    if "uganda_clean" in metrics["sets"]:
        m_fit = data["set"].eq("uganda_clean").to_numpy()
        temp = fit_temperature(logits["center"][m_fit], data.loc[m_fit, "true"].map(cidx).to_numpy())
        rows = []
        for name in EXTERNAL_SETS:
            m = data["set"].eq(name).to_numpy()
            if not m.any():
                continue
            y = data.loc[m, "true"].map(cidx).to_numpy()
            p_t = softmax(logits["center"][m] / temp)
            rows.append({"set": name, "ECE before": metrics["sets"][name]["center"]["ece"],
                         "ECE after": expected_calibration_error(y, p_t)[0],
                         "mean conf before": float(probs["center"][m].max(1).mean()),
                         "mean conf after": float(p_t.max(1).mean()), "accuracy": float((p_t.argmax(1) == y).mean())})
        metrics["temperature_fitted_on_uganda_clean"] = temp
        section(f"Calibration: temperature T={temp:.2f} fitted on uganda_clean",
                pd.DataFrame(rows).set_index("set").round(3),
                "ECE = expected calibration error (0 = confidence matches accuracy). Temperature scaling does not "
                "change predictions, only how confident they are.")

    # 6. selective prediction: accuracy vs share answered
    rows = []
    for name in EXTERNAL_SETS:
        if name in metrics["sets"]:
            for r in metrics["sets"][name]["center"]["selective"]:
                rows.append({"set": name, **r})
    if rows:
        sel = pd.DataFrame(rows).pivot(index="threshold", columns="set",
                                       values=["coverage", "accuracy_on_accepted"]).round(3)
        section("Confidence threshold: share answered (coverage) and accuracy on those", sel)

    # 7. natural image quality
    rows = []
    for name in EXTERNAL_SETS:
        sub = data[data["set"] == name]
        if len(sub) < 20:
            continue
        correct = sub["pred_center"] == sub["true"]
        for col in ("sharpness", "brightness"):
            q = pd.qcut(sub[col], 4, labels=["Q1 low", "Q2", "Q3", "Q4 high"], duplicates="drop")
            for b, g in correct.groupby(q, observed=True):
                rows.append({"set": name, "measure": col, "quartile": b, "n": len(g), "accuracy": g.mean(),
                             "mean_conf": sub.loc[g.index, "conf_center"].mean()})
    if rows:
        section("Accuracy by natural image quality (quartiles within each set)",
                pd.DataFrame(rows).set_index(["set", "measure", "quartile"]).round(3))

    # 8. synthetic perturbations on a fixed sample
    from app.inference import load_image

    pert = perturbations()
    rows = []
    for name in ("uganda_clean", "bracol"):
        sub = data[data["set"] == name]
        if sub.empty:
            continue
        sample = sub.sample(frac=1, random_state=0).groupby("true").head(args.perturb_per_class)
        y = sample["true"].map(cidx).to_numpy()
        imgs = [load_image((raw_dir / p).read_bytes()) for p in sample["path"]]
        clean_pred = None
        for pname, fn in pert.items():
            with ThreadPoolExecutor(args.workers) as ex:
                lg = np.stack(list(ex.map(lambda im: model.logits(model.center(fn(im)))[0], imgs)))[:, perm]
            p = softmax(lg)
            pred = p.argmax(1)
            clean_pred = pred if pname == "clean" else clean_pred
            rows.append({"set": name, "perturbation": pname, "n": len(y), "accuracy": float((pred == y).mean()),
                         "macro_f1": float(f1_score(y, pred, labels=sorted(set(y)), average="macro", zero_division=0)),
                         "mean_conf": float(p.max(1).mean()),
                         "changed_vs_clean": float((pred != clean_pred).mean())})
    if rows:
        pt = pd.DataFrame(rows).set_index(["set", "perturbation"]).round(3)
        metrics["perturbations"] = pt.reset_index().to_dict("records")
        section("Synthetic image-quality perturbations (deployed preprocessing)", pt,
                f"Stratified sample, up to {args.perturb_per_class} images per class. Blur radius is scaled to "
                "image size (radius r at 512px). changed_vs_clean = share of predictions that flip.")

    # 9. latency of the deployed path (decode + preprocess + ONNX, single image, 1 thread)
    lat = data.groupby("source")["latency_ms"].describe(percentiles=[0.5, 0.95])[["count", "50%", "95%"]].round(1)
    size = data.groupby("source")[["width", "height"]].median().astype(int)
    lat["median image"] = size["width"].astype(str) + "x" + size["height"].astype(str)
    section("Latency per image (ms, CPU, 1 thread)", lat)

    # 10. example sheets
    for name in LABELLED_SETS + ("out_of_scope",):
        sub = data[data["set"] == name].rename(columns={"pred_center": "pred", "conf_center": "conf"})
        if sub.empty:
            continue
        if name == "out_of_scope":
            sub = sub.assign(true=sub["folder"])
            examples_sheet(sub.sort_values("conf", ascending=False).head(12), raw_dir,
                           out_dir / "examples" / f"{name}_most_confident.jpg",
                           f"{name}: most confident predictions on images with no valid class")
            continue
        wrong, right = sub[sub["pred"] != sub["true"]], sub[sub["pred"] == sub["true"]]
        examples_sheet(wrong.sort_values("conf", ascending=False).head(16), raw_dir,
                       out_dir / "examples" / f"{name}_errors_most_confident.jpg",
                       f"{name}: most confident errors ({len(wrong)} errors in {len(sub)})")
        examples_sheet(right.sample(min(12, len(right)), random_state=0), raw_dir,
                       out_dir / "examples" / f"{name}_correct_random.jpg", f"{name}: random correct predictions")

    data.drop(columns=[c for c in ("md5", "phash", "error") if c in data]).to_csv(out_dir / "predictions.csv", index=False)
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, default=float), encoding="utf-8")
    (out_dir / "report.md").write_text("".join(report), encoding="utf-8")
    print(f"\nWrote {out_dir}: report.md, metrics.json, predictions.csv, plots/, examples/")


if __name__ == "__main__":
    main()
