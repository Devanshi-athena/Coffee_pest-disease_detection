"""Convert the pretrained Hugging Face baseline (timm efficientnet_b0) into backend files.

Source: https://huggingface.co/Huyt/arabica-coffee-leaf-disease-efficientnet-b0
Download config.json + model.safetensors into --weights first (see README).

Writes to --out (default backend/models): model.onnx, labels.json, preprocess.json, model_info.json.
Then checks that the backend's torch-free preprocessing + ONNX gives the same probabilities as
timm's canonical eval transform + the PyTorch model, on real images from data/raw.
"""
import argparse
import hashlib
import json
import math
import sys
from datetime import date
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "ml"))
sys.path.insert(0, str(ROOT / "backend"))
from src.common import die, label_lookup, list_images, load_config, resolve_path  # noqa: E402

HF_REPO = "Huyt/arabica-coffee-leaf-disease-efficientnet-b0"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", help="path to config.yaml (default: configs/config.yaml)")
    ap.add_argument("--weights", type=Path, default=ROOT / "artifacts" / "hf_model",
                    help="folder with config.json and model.safetensors")
    ap.add_argument("--out", type=Path, help="output folder (default: paths.model_dir)")
    ap.add_argument("--check-per-source", type=int, default=10, help="real images per source for the parity check")
    args = ap.parse_args()

    import timm
    import torch
    from safetensors.torch import load_file

    from src.export_onnx import export_onnx, verify_onnx

    cfg = load_config(args.config)
    out = args.out or resolve_path(cfg, "model_dir")
    hf_cfg_path, weights = args.weights / "config.json", args.weights / "model.safetensors"
    if not (hf_cfg_path.is_file() and weights.is_file()):
        die(f"Expected config.json and model.safetensors in {args.weights}. Download them from "
            f"https://huggingface.co/{HF_REPO} first.")
    hf = json.loads(hf_cfg_path.read_text(encoding="utf-8"))
    pcfg = hf["pretrained_cfg"]

    # Map the model's own class names onto ours, keeping the model's output order.
    lookup = label_lookup(cfg)
    labels = []
    for name in hf["label_names"]:
        cls = lookup.get(name.lower().replace("-", "_").replace(" ", "_"))
        if cls is None:
            die(f"Model class '{name}' has no match in config.yaml label_aliases.")
        labels.append(cls)
    if sorted(labels) != sorted(cfg["classes"]):
        die(f"Model classes {labels} do not cover config classes {cfg['classes']}.")
    print("model output order:", dict(zip(hf["label_names"], labels)))

    model = timm.create_model(hf["architecture"], pretrained=False, num_classes=hf["num_classes"])
    model.load_state_dict(load_file(str(weights)), strict=True)  # fails loudly on any key mismatch
    model.eval()
    size = pcfg["input_size"][-1]
    # timm's eval transform: resize shorter side to floor(size / crop_pct), then center-crop `size`.
    preprocess = {
        "resize_shorter": int(math.floor(size / pcfg["crop_pct"])),
        "crop_size": size,
        "interpolation": pcfg["interpolation"],
        "mean": pcfg["mean"],
        "std": pcfg["std"],
        "color": "RGB", "layout": "NCHW", "scale": "0-1", "output": "logits",
    }

    out.mkdir(parents=True, exist_ok=True)
    onnx_path = export_onnx(model, out / "model.onnx", size, cfg["train"]["onnx_opset"])
    print(f"exported {onnx_path} (relative torch/onnx diff on random input: {verify_onnx(model, onnx_path, size):.1e})")
    (out / "labels.json").write_text(json.dumps(labels, indent=2), encoding="utf-8")
    (out / "preprocess.json").write_text(json.dumps(preprocess, indent=2), encoding="utf-8")
    (out / "model_info.json").write_text(json.dumps({
        "source": f"https://huggingface.co/{HF_REPO}",
        "architecture": hf["architecture"],
        "original_label_names": hf["label_names"],
        "weights_sha256": hashlib.sha256(weights.read_bytes()).hexdigest(),
        "trained_on": "JMuBEN + JMuBEN2 (deduplicated, group split) per the model card",
        "converted": date.today().isoformat(),
    }, indent=2), encoding="utf-8")

    # Parity on real images: timm canonical pipeline vs the backend's own inference code.
    from app.inference import PreprocessSpec, load_image, preprocess as backend_preprocess, softmax

    tf = timm.data.create_transform(input_size=pcfg["input_size"], interpolation=pcfg["interpolation"],
                                    crop_pct=pcfg["crop_pct"], crop_mode=pcfg.get("crop_mode", "center"),
                                    mean=pcfg["mean"], std=pcfg["std"])
    import onnxruntime as ort

    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    spec = PreprocessSpec.from_dict(preprocess)
    input_name = session.get_inputs()[0].name
    raw = resolve_path(cfg, "raw_dir")
    rng = np.random.default_rng(0)
    files = []
    for source in cfg["sources"]:
        found = list_images(raw / source, cfg["image"]["extensions"]) if (raw / source).is_dir() else []
        files += list(rng.choice(found, min(args.check_per_source, len(found)), replace=False)) if found else []
    worst, agree = 0.0, 0
    for f in files:
        img = load_image(Path(f).read_bytes())
        with torch.no_grad():
            p_torch = model(tf(img).unsqueeze(0)).softmax(-1)[0].numpy()
        p_onnx = softmax(session.run(None, {input_name: backend_preprocess(img, spec)})[0][0])
        worst = max(worst, float(np.abs(p_torch - p_onnx).max()))
        agree += int(p_torch.argmax() == p_onnx.argmax())
    print(f"parity on {len(files)} real images: top-1 agreement {agree}/{len(files)}, "
          f"max probability difference {worst:.2e}")
    if agree != len(files) or worst > 1e-3:
        die("Backend preprocessing does not reproduce timm's transform closely enough.")
    print(f"Wrote {out}: model.onnx, labels.json, preprocess.json, model_info.json")


if __name__ == "__main__":
    main()
