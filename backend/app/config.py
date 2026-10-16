"""Backend settings: defaults from configs/config.yaml, overridable by env vars.

Env vars: CONFIG_PATH, MODEL_DIR, CONFIDENCE_THRESHOLD, MAX_UPLOAD_MB, CORS_ORIGINS (comma-separated),
STATIC_DIR (built frontend to serve at /, for single-container hosting), ORT_THREADS (read by inference.py).
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import yaml

CONFIG_REL = Path("configs") / "config.yaml"


@dataclass(frozen=True)
class Settings:
    classes: list[str]
    model_dir: Path
    confidence_threshold: float          # default when the model folder has no decision.json
    threshold_override: float | None     # CONFIDENCE_THRESHOLD env var; beats decision.json
    top_k: int
    max_upload_bytes: int
    cors_origins: list[str]
    static_dir: Path | None = None       # STATIC_DIR env var: serve the built frontend at /


def find_config() -> Path:
    if os.environ.get("CONFIG_PATH"):
        return Path(os.environ["CONFIG_PATH"])
    here = Path(__file__).resolve()
    for p in here.parents:
        if (p / CONFIG_REL).is_file():
            return p / CONFIG_REL
    raise FileNotFoundError(f"{CONFIG_REL} not found above {here}; set CONFIG_PATH.")


def load_settings() -> Settings:
    path = find_config()
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    root = path.resolve().parents[1]
    inf = cfg["inference"]
    origins = os.environ.get("CORS_ORIGINS", "http://localhost:5180")
    return Settings(
        classes=cfg.get("phase2", {}).get("classes") or cfg["classes"],
        model_dir=Path(os.environ.get("MODEL_DIR") or root / cfg["paths"]["model_dir"]),
        confidence_threshold=float(inf["confidence_threshold"]),
        threshold_override=float(os.environ["CONFIDENCE_THRESHOLD"]) if os.environ.get("CONFIDENCE_THRESHOLD") else None,
        top_k=int(inf["top_k"]),
        max_upload_bytes=int(float(os.environ.get("MAX_UPLOAD_MB", inf["max_upload_mb"])) * 1024 * 1024),
        cors_origins=[o.strip() for o in origins.split(",") if o.strip()],
        static_dir=Path(os.environ["STATIC_DIR"]) if os.environ.get("STATIC_DIR") else None,
    )
