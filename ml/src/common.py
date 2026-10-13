"""Config loading, label mapping and file helpers.

Only depends on pyyaml (+ pillow/tqdm inside functions) so it works in the
CPU prep env and in the Kaggle/Colab notebook alike.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import yaml

CONFIG_REL = Path("configs") / "config.yaml"
SPLITS = ("train", "val", "test")


def die(msg: str) -> None:
    print(f"\nERROR: {msg}", file=sys.stderr)
    sys.exit(1)


def find_repo_root(start: Path | None = None) -> Path:
    start = Path(start or __file__).resolve()
    for p in [start, *start.parents]:
        if (p / CONFIG_REL).is_file():
            return p
    raise FileNotFoundError(f"Could not find {CONFIG_REL} in {start} or any parent folder.")


def load_config(path: str | Path | None = None) -> dict:
    path = Path(path) if path else find_repo_root() / CONFIG_REL
    if not path.is_file():
        raise FileNotFoundError(f"Config not found: {path}")
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    # configs/config.yaml -> repo root is two levels up
    cfg["_root"] = str(path.resolve().parents[1])
    return cfg


def resolve_path(cfg: dict, key: str) -> Path:
    return Path(cfg["_root"]) / cfg["paths"][key]


def normalize_label(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


def label_lookup(cfg: dict) -> dict[str, str]:
    """normalized alias -> class name (each class also maps to itself)."""
    lookup = {}
    for cls in cfg["classes"]:
        lookup[normalize_label(cls)] = cls
        for alias in cfg.get("label_aliases", {}).get(cls, []):
            lookup[normalize_label(alias)] = cls
    return lookup


def map_label(raw_label: str, source: str, cfg: dict, lookup: dict[str, str]) -> tuple[str | None, str]:
    """Returns (class or None, reason). reason is "ok", "ignored", "not_used_for_source" or "unknown"."""
    norm = normalize_label(raw_label)
    if norm in {normalize_label(x) for x in cfg.get("ignore_labels", [])}:
        return None, "ignored"
    cls = lookup.get(norm)
    if cls is None:
        return None, "unknown"
    allowed = cfg["sources"][source].get("classes")
    if allowed and cls not in allowed:
        return None, "not_used_for_source"
    return cls, "ok"


def list_images(folder: Path, extensions: list[str]) -> list[Path]:
    exts = {e.lower() for e in extensions}
    return sorted(p for p in folder.rglob("*") if p.is_file() and p.suffix.lower() in exts)


def scan_raw(cfg: dict) -> tuple[dict[str, dict[str, list[Path]]], list[str], list[str]]:
    """Walks data/raw/<source>/<class>/.

    Returns ({source: {raw_class: [files]}}, missing_sources, unknown_dirs).
    """
    raw = resolve_path(cfg, "raw_dir")
    if not raw.is_dir():
        die(
            f"Raw data folder not found: {raw}\n"
            f"Create it and arrange each dataset as {raw}/<source>/<class>/<images>.\n"
            f"Configured sources: {', '.join(cfg['sources'])}"
        )
    found, missing = {}, []
    for source in cfg["sources"]:
        sdir = raw / source
        if not sdir.is_dir():
            missing.append(source)
            continue
        class_dirs = sorted(d for d in sdir.iterdir() if d.is_dir())
        found[source] = {d.name: list_images(d, cfg["image"]["extensions"]) for d in class_dirs}
    unknown = sorted(d.name for d in raw.iterdir() if d.is_dir() and d.name not in cfg["sources"])
    return found, missing, unknown


def read_image(src, draft: int | None = 256):
    """Fully decodes (so truncated files raise) and EXIF-rotates an image.

    JPEGs are decoded at a reduced scale that is still >= `draft` px, which is much
    faster for large phone photos. Returns (rgb_image, orig_width, orig_height, mode, format).
    """
    from PIL import Image, ImageOps

    with Image.open(src) as im:
        w, h, mode, fmt = im.width, im.height, im.mode, im.format
        if draft:
            im.draft("RGB", (draft, draft))
        rgb = ImageOps.exif_transpose(im).convert("RGB")
    return rgb, w, h, mode, fmt


def phash_variants(img, hash_size: int = 8) -> list[int]:
    """Perceptual hash (same algorithm as imagehash.phash) for the 8 rotations/flips of an image.

    Element 0 is the plain hash. Rotating the 32x32 thumbnail instead of the image keeps this cheap.
    Format a hash like imagehash does with f"{h:0{hash_size**2 // 4}x}".
    """
    import numpy as np
    from PIL import Image
    from scipy.fft import dct

    side = hash_size * 4
    pixels = np.asarray(img.convert("L").resize((side, side), Image.Resampling.LANCZOS), dtype=np.float64)
    weights = 1 << np.arange(hash_size * hash_size - 1, -1, -1, dtype=object)  # MSB first
    out = []
    for base in (pixels, np.fliplr(pixels)):
        for k in range(4):
            low = dct(dct(np.rot90(base, k), axis=0), axis=1)[:hash_size, :hash_size]
            bits = (low > np.median(low)).flatten()
            out.append(int(np.dot(bits.astype(object), weights)))
    return out


def run_parallel(fn, items: list, workers: int, desc: str) -> list:
    """Maps fn over items with a progress bar; keeps input order."""
    from tqdm import tqdm

    if workers <= 1:
        return [fn(x) for x in tqdm(items, desc=desc)]
    from concurrent.futures import ProcessPoolExecutor

    with ProcessPoolExecutor(max_workers=workers) as ex:
        return list(tqdm(ex.map(fn, items, chunksize=32), total=len(items), desc=desc))
