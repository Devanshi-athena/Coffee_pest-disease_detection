"""Prediction, metrics and plots."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score


@torch.no_grad()
def predict(model, loader, device) -> tuple[np.ndarray, np.ndarray]:
    """Returns (y_true, probs[N, C])."""
    model.eval()
    ys, ps = [], []
    for x, y in loader:
        logits = model(x.to(device, non_blocking=True))
        ps.append(torch.softmax(logits.float(), dim=1).cpu().numpy())
        ys.append(y.numpy())
    return np.concatenate(ys), np.concatenate(ps)


def compute_metrics(y_true: np.ndarray, probs: np.ndarray, classes: list[str]) -> dict:
    """Macro-F1 and the per-class report only cover classes present in y_true
    (e.g. rocole has only healthy/rust); the confusion matrix uses all classes."""
    y_pred = probs.argmax(1)
    present = sorted(set(y_true.tolist()))
    report = classification_report(y_true, y_pred, labels=present, target_names=[classes[i] for i in present],
                                   output_dict=True, zero_division=0)
    return {
        "n": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=present, average="macro", zero_division=0)),
        "predicted_outside_present_classes": float(np.mean(~np.isin(y_pred, present))),
        "per_class": {k: v for k, v in report.items() if k in classes},
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=list(range(len(classes)))).tolist(),
    }


def threshold_sweep(y_true: np.ndarray, probs: np.ndarray, thresholds=None) -> list[dict]:
    """For each threshold: share of images answered (coverage) and accuracy on those.
    Use the val sweep to choose inference.confidence_threshold."""
    thresholds = thresholds if thresholds is not None else np.round(np.arange(0.3, 0.96, 0.05), 2)
    conf, pred = probs.max(1), probs.argmax(1)
    rows = []
    for t in thresholds:
        keep = conf >= t
        rows.append({
            "threshold": float(t),
            "coverage": float(keep.mean()),
            "accuracy_on_accepted": float((pred[keep] == y_true[keep]).mean()) if keep.any() else None,
        })
    return rows


def plot_confusion_matrix(cm, classes: list[str], path: Path, title: str = "") -> None:
    import matplotlib.pyplot as plt

    cm = np.asarray(cm)
    fig, ax = plt.subplots(figsize=(5, 4.5))
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(classes)), classes, rotation=45, ha="right")
    ax.set_yticks(range(len(classes)), classes)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.set_title(title)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def expected_calibration_error(y_true: np.ndarray, probs: np.ndarray, n_bins: int = 15) -> tuple[float, list[dict]]:
    """ECE: average |accuracy - confidence| over equal-width confidence bins, weighted by bin size."""
    conf, correct = probs.max(1), probs.argmax(1) == y_true
    edges = np.linspace(0, 1, n_bins + 1)
    bins, ece = [], 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            acc, c = float(correct[m].mean()), float(conf[m].mean())
            ece += m.mean() * abs(acc - c)
            bins.append({"lo": float(lo), "hi": float(hi), "n": int(m.sum()), "accuracy": acc, "confidence": c})
    return float(ece), bins


def plot_reliability(y_true: np.ndarray, probs: np.ndarray, path: Path, title: str = "") -> None:
    import matplotlib.pyplot as plt

    ece, bins = expected_calibration_error(y_true, probs)
    conf, correct = probs.max(1), probs.argmax(1) == y_true
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 3.8))
    ax1.plot([0, 1], [0, 1], "--", color="gray", lw=1)
    ax1.bar([b["confidence"] for b in bins], [b["accuracy"] for b in bins], width=1 / 15, alpha=0.7,
            edgecolor="black")
    ax1.set(xlim=(0, 1), ylim=(0, 1), xlabel="confidence", ylabel="accuracy", title=f"{title}  ECE={ece:.3f}")
    ax2.hist([conf[correct], conf[~correct]], bins=20, range=(0, 1), label=["correct", "wrong"], stacked=True)
    ax2.set(xlabel="confidence", ylabel="images", title="confidence of predictions")
    ax2.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def fit_temperature(logits: np.ndarray, y_true: np.ndarray) -> float:
    """Single temperature T minimizing NLL of softmax(logits / T)."""
    from scipy.optimize import minimize_scalar
    from scipy.special import log_softmax

    def nll(log_t: float) -> float:
        return -float(log_softmax(logits / np.exp(log_t), axis=1)[np.arange(len(y_true)), y_true].mean())

    return float(np.exp(minimize_scalar(nll, bounds=(-3, 3), method="bounded").x))
