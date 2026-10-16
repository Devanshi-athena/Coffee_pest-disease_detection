"""ONNX inference without torch. Preprocessing mirrors the training notebook's eval transform, driven by
preprocess.json saved next to the model. If the model folder also has decision.json (+ ood_bank.npz), the
"not sure" rule from model evaluation is applied exactly: answer only when the prediction is confident enough
AND the photo is close enough to the training data."""
from __future__ import annotations

import io
import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

log = logging.getLogger(__name__)

MODEL_FILES = ("model.onnx", "labels.json", "preprocess.json")
Image.MAX_IMAGE_PIXELS = 50_000_000  # reject decompression bombs early

_RESAMPLE = {"bilinear": Image.Resampling.BILINEAR, "bicubic": Image.Resampling.BICUBIC,
             "nearest": Image.Resampling.NEAREST}


class ModelNotAvailable(Exception):
    pass


class InvalidImage(Exception):
    pass


@dataclass(frozen=True)
class PreprocessSpec:
    resize_shorter: int
    crop_size: int
    mean: tuple[float, float, float]
    std: tuple[float, float, float]
    interpolation: str = "bilinear"

    @classmethod
    def from_dict(cls, d: dict) -> "PreprocessSpec":
        return cls(int(d["resize_shorter"]), int(d["crop_size"]), tuple(d["mean"]), tuple(d["std"]),
                   d.get("interpolation", "bilinear"))


@dataclass(frozen=True)
class Decision:
    """How a prediction becomes an answer. Fitted on validation data by ml/scripts/09_calibrate_and_evaluate.py."""
    confidence_threshold: float
    ood_threshold: float | None = None    # max kNN distance to the training features; None = no distance check
    knn_k: int = 10
    tta_views: tuple[str, ...] = ("center",)
    bank: np.ndarray | None = field(default=None, repr=False)  # L2-normalised training features


def load_image(data: bytes) -> Image.Image:
    try:
        img = Image.open(io.BytesIO(data))
        img = ImageOps.exif_transpose(img)  # phone photos are often stored rotated
        return img.convert("RGB")
    except Exception as e:
        raise InvalidImage(f"Could not read image: {e}") from e


def preprocess(img: Image.Image, spec: PreprocessSpec) -> np.ndarray:
    """RGB image -> float32 [1, 3, crop, crop]."""
    # Integer maths copied from torchvision Resize/CenterCrop (truncate the long side, round the
    # crop offset): a one-pixel shift measurably changes predictions on large photos.
    w, h = img.size
    short, long = min(w, h), max(w, h)
    new_short, new_long = spec.resize_shorter, int(spec.resize_shorter * long / short)
    new_w, new_h = (new_short, new_long) if w <= h else (new_long, new_short)
    img = img.resize((new_w, new_h), _RESAMPLE.get(spec.interpolation, Image.Resampling.BILINEAR))
    left = int(round((new_w - spec.crop_size) / 2.0))
    top = int(round((new_h - spec.crop_size) / 2.0))
    img = img.crop((left, top, left + spec.crop_size, top + spec.crop_size))
    x = np.asarray(img, dtype=np.float32) / 255.0
    x = (x - np.array(spec.mean, dtype=np.float32)) / np.array(spec.std, dtype=np.float32)
    return x.transpose(2, 0, 1)[None]


def views(img: Image.Image, spec: PreprocessSpec, names: tuple[str, ...]) -> np.ndarray:
    """Test-time augmentation views, stacked into one batch. Same definitions as the evaluation script."""
    spec224 = PreprocessSpec(spec.crop_size, spec.crop_size, spec.mean, spec.std, spec.interpolation)
    make = {
        "center": lambda: preprocess(img, spec),
        "hflip": lambda: preprocess(ImageOps.mirror(img), spec),
        "vflip": lambda: preprocess(ImageOps.flip(img), spec),
        "hvflip": lambda: preprocess(ImageOps.flip(ImageOps.mirror(img)), spec),
        "scale224": lambda: preprocess(img, spec224),
    }
    return np.concatenate([make[n]() for n in names])


def softmax(logits: np.ndarray) -> np.ndarray:
    z = logits - logits.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


class Predictor:
    """Wraps an onnxruntime session (or anything with the same run/get_inputs API).

    The model may return logits only, or (logits, features); features enable the distance check.
    """

    def __init__(self, session, labels: list[str], spec: PreprocessSpec, decision: Decision, top_k: int = 3):
        self.session = session
        self.input_name = session.get_inputs()[0].name
        self.labels = labels
        self.spec = spec
        self.decision = decision
        self.top_k = min(top_k, len(labels))

    @property
    def threshold(self) -> float:
        return self.decision.confidence_threshold

    @classmethod
    def from_dir(cls, model_dir: Path, default_threshold: float, top_k: int, expected_classes: list[str],
                 threshold_override: float | None = None) -> "Predictor":
        model_dir = Path(model_dir)
        missing = [f for f in MODEL_FILES if not (model_dir / f).is_file()]
        if missing:
            raise ModelNotAvailable(f"Missing in {model_dir}: {', '.join(missing)}")
        import onnxruntime as ort

        labels = json.loads((model_dir / "labels.json").read_text(encoding="utf-8"))
        spec = PreprocessSpec.from_dict(json.loads((model_dir / "preprocess.json").read_text(encoding="utf-8")))
        if sorted(labels) != sorted(expected_classes):  # order may differ: labels.json follows the model output
            log.warning("labels.json %s differs from config classes %s; using labels.json", labels, expected_classes)
        opts = ort.SessionOptions()
        # ORT_THREADS caps CPU threads (1 on fractional-CPU hosts, where extra threads only contend); 0 = all cores.
        opts.intra_op_num_threads = int(os.environ.get("ORT_THREADS", "0"))
        session = ort.InferenceSession(str(model_dir / "model.onnx"), opts, providers=["CPUExecutionProvider"])
        n_out = session.get_outputs()[0].shape[-1]
        if isinstance(n_out, int) and n_out != len(labels):
            raise ModelNotAvailable(f"model outputs {n_out} classes but labels.json has {len(labels)}")

        dec_file, bank_file = model_dir / "decision.json", model_dir / "ood_bank.npz"
        if dec_file.is_file():
            d = json.loads(dec_file.read_text(encoding="utf-8"))
            bank = None
            if d.get("ood_threshold") is not None:
                if not bank_file.is_file() or len(session.get_outputs()) < 2:
                    raise ModelNotAvailable("decision.json uses a distance check but ood_bank.npz or the model's "
                                            "feature output is missing")
                bank = np.load(bank_file)["features"].astype(np.float32)
            decision = Decision(confidence_threshold=float(d["confidence_threshold"]),
                                ood_threshold=d.get("ood_threshold"), knn_k=int(d.get("knn_k", 10)),
                                tta_views=tuple(d.get("tta_views", ["center"])), bank=bank)
        else:
            decision = Decision(confidence_threshold=default_threshold)
        if threshold_override is not None:
            decision = Decision(threshold_override, decision.ood_threshold, decision.knn_k, decision.tta_views,
                                decision.bank)
        return cls(session, labels, spec, decision, top_k)

    def predict(self, img: Image.Image) -> dict:
        d = self.decision
        outputs = self.session.run(None, {self.input_name: views(img, self.spec, d.tta_views)})
        probs = softmax(outputs[0].astype(np.float64)).mean(axis=0)
        distance = None
        if d.ood_threshold is not None and d.bank is not None and len(outputs) > 1:
            f = outputs[1].astype(np.float32).mean(axis=0)
            f /= np.linalg.norm(f) or 1.0
            distance = float(1.0 - np.sort(d.bank @ f)[-d.knn_k:].mean())

        order = np.argsort(probs)[::-1][: self.top_k]
        top = [{"label": self.labels[i], "confidence": round(float(probs[i]), 4)} for i in order]
        reason = None
        if distance is not None and distance > d.ood_threshold:
            reason = "unfamiliar_image"   # unlike anything the model was trained on
        elif top[0]["confidence"] < d.confidence_threshold:
            reason = "low_confidence"
        return {
            "label": top[0]["label"],
            "confidence": top[0]["confidence"],
            "top3": top,
            "uncertain": reason is not None,
            "reason": reason,
            "outcome": "not_sure" if reason else top[0]["label"],
            "threshold": d.confidence_threshold,
            "distance": None if distance is None else round(distance, 4),
        }
