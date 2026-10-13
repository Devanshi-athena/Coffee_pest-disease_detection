"""Step 9: evaluate the exported Phase 2 model exactly as the backend runs it, and fix the decision rule.

Uses the backend's own image loading and preprocessing (backend/app/inference.py) on the ORIGINAL photos in
data/raw, plus the ONNX model, feature bank and temperature exported by the training notebook.

Rules decided before looking at any test result:
  1. Test-time augmentation (TTA): candidates below; pick the best validation worst-source balanced accuracy,
     ties (within 0.002) go to the candidate with fewer views.
  2. "Not sure": distance threshold = 95th percentile of validation distances (as in training); confidence
     threshold = lowest value giving >= TARGET_ACCURACY on the validation photos that pass the distance check.
Validation is the only set used for any choice. test / test_locked / ood are reported, never used.

Writes artifacts/phase2_eval/: decision.json, evaluation.json, predictions.csv, REPORT.md
"""
import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageOps

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "ml"))
sys.path.insert(0, str(ROOT / "backend"))
from app.inference import PreprocessSpec, load_image, preprocess  # noqa: E402
from src.common import load_config, resolve_path  # noqa: E402

VIEWS = ["center", "hflip", "vflip", "hvflip", "scale224"]
CANDIDATES = {"center": ["center"], "flip2": ["center", "hflip"], "flip4": ["center", "hflip", "vflip", "hvflip"],
              "multiscale": ["center", "scale224"]}
TARGET_ACCURACY = 0.98
OOD_QUANTILE = 0.95
FAMILY = {"jmuben2": "jmuben"}


def view_tensors(img: Image.Image, spec: PreprocessSpec) -> np.ndarray:
    spec224 = PreprocessSpec(spec.crop_size, spec.crop_size, spec.mean, spec.std, spec.interpolation)
    views = {
        "center": preprocess(img, spec),
        "hflip": preprocess(ImageOps.mirror(img), spec),
        "vflip": preprocess(ImageOps.flip(img), spec),
        "hvflip": preprocess(ImageOps.flip(ImageOps.mirror(img)), spec),
        "scale224": preprocess(img, spec224),  # whole shorter side, less cropping
    }
    return np.concatenate([views[v] for v in VIEWS])


def softmax(z):
    z = z - z.max(-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(-1, keepdims=True)


def knn_distance(f: np.ndarray, bank: np.ndarray, k: int) -> np.ndarray:
    out = []
    for s in range(0, len(f), 512):
        sim = f[s:s + 512] @ bank.T
        out.append(1 - np.sort(sim, axis=1)[:, -k:].mean(1))
    return np.concatenate(out)


def combine(probs: np.ndarray, feats: np.ndarray, views: list[str]):
    """probs/feats: [n, len(VIEWS), ...]; average the chosen views."""
    idx = [VIEWS.index(v) for v in views]
    p = probs[:, idx].mean(1)
    f = feats[:, idx].mean(1)
    return p, f / np.linalg.norm(f, axis=1, keepdims=True)


def balanced_acc_by_source(y, pred, src):
    fam = np.array([FAMILY.get(s, s) for s in src])
    return {s: float(np.mean([(pred[(fam == s) & (y == c)] == c).mean() for c in np.unique(y[fam == s])]))
            for s in np.unique(fam)}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model-dir", type=Path, default=ROOT / "artifacts" / "phase2_run1")
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()

    import onnxruntime as ort
    from sklearn.metrics import confusion_matrix, f1_score, roc_auc_score

    cfg = load_config()
    raw = resolve_path(cfg, "raw_dir")
    out = resolve_path(cfg, "artifacts_dir") / "phase2_eval"
    out.mkdir(parents=True, exist_ok=True)
    md = args.model_dir
    classes = json.loads((md / "labels.json").read_text())
    spec = PreprocessSpec.from_dict(json.loads((md / "preprocess.json").read_text()))
    trained = json.loads((md / "decision.json").read_text())
    bank = np.load(md / "ood_bank.npz")["features"].astype(np.float32)
    sess = ort.InferenceSession(str(md / "model.onnx"), providers=["CPUExecutionProvider"])

    m = pd.read_csv(ROOT / cfg["phase2"]["output_dir"] / "manifest.csv", low_memory=False)
    m = m[m["split"].isin(["val", "test", "test_locked", "ood"])].copy()
    m = m[~m["split"].isin(["val", "test"]) | m["is_group_rep"]]  # one image per near-duplicate group
    m = m.reset_index(drop=True)
    print(m.groupby("split").size().to_dict())

    def run(path):
        x = view_tensors(load_image((raw / path).read_bytes()), spec)  # backend decode + EXIF + preprocessing
        z, f = sess.run(None, {"input": x})
        return softmax(z), f

    t0 = time.time()
    with ThreadPoolExecutor(args.workers) as ex:
        res = list(ex.map(run, m["path"]))
    print(f"inference: {len(m)} images x {len(VIEWS)} views in {time.time() - t0:.0f}s")
    probs = np.stack([r[0] for r in res])
    feats = np.stack([r[1] for r in res])
    y = m["task_label"].map({c: i for i, c in enumerate(classes)}).fillna(-1).astype(int).to_numpy()
    split, src = m["split"].to_numpy(), m["source"].to_numpy()
    val = split == "val"

    # 1. TTA choice on validation only
    tta = {}
    for name, views in CANDIDATES.items():
        p, _ = combine(probs[val], feats[val], views)
        per = balanced_acc_by_source(y[val], p.argmax(1), src[val])
        tta[name] = {"views": views, "worst_source": min(per.values()), "per_source": per,
                     "macro_f1": float(f1_score(y[val], p.argmax(1), average="macro"))}
    best = max(tta.values(), key=lambda r: r["worst_source"])["worst_source"]
    chosen = min((n for n, r in tta.items() if r["worst_source"] >= best - 0.002), key=lambda n: len(CANDIDATES[n]))
    print("TTA on validation:", {n: round(r["worst_source"], 4) for n, r in tta.items()}, "->", chosen)

    P, F = combine(probs, feats, CANDIDATES[chosen])
    D = knn_distance(F, bank, trained["knn_k"])
    conf, pred = P.max(1), P.argmax(1)

    # 2. thresholds on validation only
    t_ood = float(np.quantile(D[val], OOD_QUANTILE))
    keep = val & (D <= t_ood)
    t_conf = 0.99
    for t in np.round(np.arange(0.34, 1.0, 0.01), 2):
        a = keep & (conf >= t)
        if a.sum() >= 50 and (pred[a] == y[a]).mean() >= TARGET_ACCURACY:
            t_conf = float(t)
            break
    answered = (conf >= t_conf) & (D <= t_ood)
    decision = {"tta_views": CANDIDATES[chosen], "confidence_threshold": t_conf, "ood_threshold": t_ood,
                "knn_k": trained["knn_k"], "target_accuracy_on_answered": TARGET_ACCURACY,
                "ood_quantile": OOD_QUANTILE, "fitted_on": "validation split only (original photos, backend pipeline)",
                "rule": "answer only if max probability >= confidence_threshold and kNN distance <= ood_threshold"}
    print("decision:", decision)

    # 3. report every split (no further choices)
    rust_i, other_i = classes.index("rust"), classes.index("other")
    report = {"tta_candidates": tta, "decision": decision, "sets": {}}
    for s in ["val", "test", "test_locked"]:
        k = split == s
        ys, ps, a = y[k], pred[k], answered[k]
        present = sorted(set(ys))
        r = {"n": int(k.sum()), "accuracy": float((ps == ys).mean()),
             "macro_f1": float(f1_score(ys, ps, labels=present, average="macro")),
             "answered_share": float(a.mean()),
             "accuracy_when_answered": float((ps[a] == ys[a]).mean()) if a.any() else None,
             "confusion_matrix(rows=true)": confusion_matrix(ys, ps, labels=range(len(classes))).tolist(),
             "by_source": {}}
        if (ys == other_i).any():
            r["other_answered_as_rust"] = float((a & (ys == other_i) & (ps == rust_i)).sum() / (ys == other_i).sum())
        if (ys == rust_i).any():
            r["rust_missed_when_answered"] = float((a & (ys == rust_i) & (ps != rust_i)).sum() / max(1, (a & (ys == rust_i)).sum()))
        for so in sorted(set(src[k])):
            kk = src[k] == so
            pr = sorted(set(ys[kk]))
            r["by_source"][so] = {"n": int(kk.sum()), "accuracy": float((ps[kk] == ys[kk]).mean()),
                                  "macro_f1": float(f1_score(ys[kk], ps[kk], labels=pr, average="macro")),
                                  "answered_share": float(a[kk].mean()),
                                  "accuracy_when_answered": float((ps[kk][a[kk]] == ys[kk][a[kk]]).mean()) if a[kk].any() else None}
        report["sets"][s] = r
    k = split == "ood"
    a = answered[k]
    report["sets"]["ood"] = {
        "n": int(k.sum()), "rejected_not_sure": float((~a).mean()),
        "answered_as": {c: float((a & (pred[k] == i)).mean()) for i, c in enumerate(classes)},
        "by_folder": {f: {"n": int((m.loc[k, "folder"] == f).sum()),
                          "rejected_not_sure": float((~a[(m.loc[k, "folder"] == f).to_numpy()]).mean())}
                      for f in sorted(m.loc[k, "folder"].unique())},
        "auroc_distance": float(roc_auc_score(np.r_[np.zeros((split == "test").sum()), np.ones(k.sum())],
                                              np.r_[D[split == "test"], D[k]])),
        "auroc_confidence": float(roc_auc_score(np.r_[np.zeros((split == "test").sum()), np.ones(k.sum())],
                                                -np.r_[conf[split == "test"], conf[k]])),
    }
    (out / "decision.json").write_text(json.dumps(decision, indent=2))
    (out / "evaluation.json").write_text(json.dumps(report, indent=2, default=float))
    m.assign(pred=[classes[i] for i in pred], confidence=conf, distance=D, answered=answered).drop(
        columns=["leaf_poly"], errors="ignore").to_csv(out / "predictions.csv", index=False)
    for s, r in report["sets"].items():
        print(f"\n== {s}: " + json.dumps({k2: (round(v, 4) if isinstance(v, float) else v) for k2, v in r.items()
                                          if k2 not in ("by_source", "confusion_matrix(rows=true)")}, default=float))
        if "by_source" in r:
            print(pd.DataFrame(r["by_source"]).T.round(3).to_string())
        if "confusion_matrix(rows=true)" in r:
            print("confusion (rows=true, cols=pred", classes, "):", r["confusion_matrix(rows=true)"])
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
