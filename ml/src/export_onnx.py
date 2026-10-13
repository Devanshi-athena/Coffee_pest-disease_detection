"""Export a trained model to ONNX and write the files the backend needs."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch


def export_onnx(model: torch.nn.Module, path: Path, image_size: int, opset: int = 18) -> Path:
    """Exports with a dynamic batch dimension. Input "input" [N,3,H,W] float32, output "logits" [N,C]."""
    model = model.eval().cpu()
    # batch 2, not 1: torch.export specialises a size-1 dimension, which breaks the dynamic batch axis
    dummy = torch.randn(2, 3, image_size, image_size)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    names = {"input_names": ["input"], "output_names": ["logits"], "opset_version": opset}
    try:
        # torch.export-based exporter (needs onnxscript); the legacy one is deprecated.
        batch = torch.export.Dim("batch", min=1, max=1024)
        torch.onnx.export(model, (dummy,), str(path), dynamic_shapes=({0: batch},),
                          dynamo=True, external_data=False, **names)
    except Exception as e:
        print(f"dynamo export failed ({type(e).__name__}: {e}); falling back to the TorchScript exporter")
        torch.onnx.export(model, (dummy,), str(path), dynamo=False,
                          dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}}, **names)
    return path


def verify_onnx(model: torch.nn.Module, path: Path, image_size: int, rtol: float = 1e-3) -> float:
    """Runs the same random batch through torch and onnxruntime.

    Returns the max abs logit difference relative to the largest logit (float32 noise is ~1e-5)."""
    import onnxruntime as ort

    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    rel = 0.0
    for batch in (1, 3):  # the backend sends batch 1
        x = torch.randn(batch, 3, image_size, image_size)
        with torch.no_grad():
            expected = model.eval().cpu()(x).numpy()
        got = sess.run(None, {sess.get_inputs()[0].name: x.numpy()})[0]
        diff = float(np.abs(expected - got).max() / max(1.0, float(np.abs(expected).max())))
        if diff > rtol or (expected.argmax(1) != got.argmax(1)).any():
            raise RuntimeError(f"ONNX output differs from torch at batch {batch} (relative diff {diff:.2e})")
        rel = max(rel, diff)
    return rel


def write_artifacts(out_dir: Path, classes: list[str], preprocess: dict, metrics: dict) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "labels.json").write_text(json.dumps(classes, indent=2))
    (out_dir / "preprocess.json").write_text(json.dumps(preprocess, indent=2))
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
