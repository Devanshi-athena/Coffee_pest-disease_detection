"""Controlled crop views for whole-leaf photos (numpy + OpenCV + PIL; no torch).

Views are boxes normalised to the image (x0, y0, x1, y1 in 0-1):
  full  the whole image (close-ups, and the deployed behaviour)
  leaf  the leaf's bounding box plus a margin
  zoom  square crops inside the leaf that are mostly leaf tissue and, for rust leaves, contain a real share of
        the leaf's rust-coloured pixels, so a "rust" crop cannot show only healthy tissue
"""
from __future__ import annotations

import hashlib

import numpy as np

FULL = (0.0, 0.0, 1.0, 1.0)


def masks(rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(leaf mask, rust-colour mask) for a leaf on a plain background."""
    import cv2

    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    tissue = ((h >= 5) & (h <= 95) & (s >= 40) & (v >= 30)).astype(np.uint8)
    tissue = cv2.morphologyEx(tissue, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    tissue = cv2.morphologyEx(tissue, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    n, lab, stats, _ = cv2.connectedComponentsWithStats(tissue)
    leaf = (lab == 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))) if n > 1 else tissue.astype(bool)
    # orange/yellow rust pustules: hue ~8-28 (OpenCV 0-180), strongly saturated, not dark
    rust = leaf & (h >= 8) & (h <= 28) & (s >= 90) & (v >= 90)
    return leaf, rust


def bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def views_for(rgb: np.ndarray, label: str, cfg: dict, key: str, leaf_box_norm=None) -> list[dict]:
    """Views for one whole-leaf image. `key` seeds the random zooms so results are reproducible.

    leaf_box_norm: optional annotated leaf box (normalised); otherwise it comes from the leaf mask.
    """
    H, W = rgb.shape[:2]
    leaf, rust = masks(rgb)
    if leaf_box_norm is not None and not np.isnan(leaf_box_norm).any():
        x0, y0, x1, y1 = (np.array(leaf_box_norm) * [W, H, W, H]).round().astype(int)
    else:
        b = bbox(leaf)
        if b is None:
            return [{"view": "full", "box": FULL, "leaf_fraction": 0.0, "rust_share": np.nan}]
        x0, y0, x1, y1 = b
    m = cfg["leaf_margin"]
    bw, bh = x1 - x0, y1 - y0
    lx0, ly0 = max(0, x0 - m * bw), max(0, y0 - m * bh)
    lx1, ly1 = min(W, x1 + m * bw), min(H, y1 + m * bh)
    out = [{"view": "leaf", "box": (lx0 / W, ly0 / H, lx1 / W, ly1 / H),
            "leaf_fraction": float(leaf[int(ly0):int(ly1), int(lx0):int(lx1)].mean()), "rust_share": np.nan}]

    total_rust = int(rust.sum())
    if label == "rust" and total_rust < cfg["min_rust_pixels"]:
        return out  # cannot locate the lesions: only the safe leaf view
    rng = np.random.default_rng(int(hashlib.sha1(key.encode()).hexdigest()[:8], 16))
    short = min(bw, bh)
    tries, zooms = 0, []
    while len(zooms) < cfg["zoom_per_image"] and tries < 60:
        tries += 1
        side = int(rng.uniform(*cfg["zoom_side"]) * short)
        if side < 32:
            break
        cx, cy = rng.uniform(x0 + side / 2, max(x0 + side / 2, x1 - side / 2)), \
            rng.uniform(y0 + side / 2, max(y0 + side / 2, y1 - side / 2))
        a0, b0 = int(max(0, cx - side / 2)), int(max(0, cy - side / 2))
        a1, b1 = min(W, a0 + side), min(H, b0 + side)
        frac = float(leaf[b0:b1, a0:a1].mean())
        if frac < cfg["min_leaf_fraction"]:
            continue
        share = float(rust[b0:b1, a0:a1].sum() / total_rust) if total_rust else 0.0
        if label == "rust" and share < cfg["min_rust_share"]:
            continue
        if any(abs(a0 - z[0]) < side * 0.25 and abs(b0 - z[1]) < side * 0.25 for z in zooms):
            continue  # too similar to a crop we already have
        zooms.append((a0, b0, a1, b1))
        out.append({"view": "zoom", "box": (a0 / W, b0 / H, a1 / W, b1 / H), "leaf_fraction": frac,
                    "rust_share": share if label == "rust" else np.nan})
    return out


def apply_view(img, box):
    """Crops a PIL image to a normalised box."""
    W, H = img.size
    x0, y0, x1, y1 = box
    return img.crop((round(x0 * W), round(y0 * H), round(x1 * W), round(y1 * H)))


# ---------------------------------------------------------------- phase 2 (3 classes, field photos)

def polygon_mask(shape: tuple[int, int], poly_norm: list[tuple[float, float]]) -> np.ndarray:
    """Boolean mask of a polygon given in normalised (x, y) coordinates."""
    import cv2

    H, W = shape
    pts = np.array([[x * W, y * H] for x, y in poly_norm], dtype=np.int32)
    m = np.zeros((H, W), np.uint8)
    cv2.fillPoly(m, [pts], 1)
    return m.astype(bool)


def symptom_mask(rgb: np.ndarray, leaf: np.ndarray, label: str) -> np.ndarray:
    """Pixels inside the leaf that look like the labelled symptom.

    rust:  orange/yellow pustules (as in masks()).
    other: any clearly non-green tissue (brown, yellow, grey, pale): lesions, mines, spots.
    """
    import cv2

    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    if label == "rust":
        return leaf & (h >= 8) & (h <= 28) & (s >= 90) & (v >= 90)
    green = (h >= 30) & (h <= 90) & (s >= 50)
    return leaf & ~green & (v >= 25)


def views_for_v2(rgb: np.ndarray, label: str, cfg: dict, key: str, leaf_poly=None) -> list[dict]:
    """Leaf view + symptom-aware zooms. Healthy zooms only need to be mostly leaf; rust/other zooms must also
    contain a real share of the leaf's symptom pixels. leaf_poly (normalised polygon) overrides the colour mask,
    which is required for field photos where the background is also green."""
    H, W = rgb.shape[:2]
    if leaf_poly:
        leaf = polygon_mask((H, W), leaf_poly)
    else:
        leaf, _ = masks(rgb)
    b = bbox(leaf)
    if b is None:
        return [{"view": "full", "box": FULL, "leaf_fraction": 0.0, "symptom_share": np.nan}]
    x0, y0, x1, y1 = b
    m = cfg["leaf_margin"]
    bw, bh = x1 - x0, y1 - y0
    lx0, ly0 = max(0, x0 - m * bw), max(0, y0 - m * bh)
    lx1, ly1 = min(W, x1 + m * bw), min(H, y1 + m * bh)
    out = [{"view": "leaf", "box": (lx0 / W, ly0 / H, lx1 / W, ly1 / H),
            "leaf_fraction": float(leaf[int(ly0):int(ly1), int(lx0):int(lx1)].mean()), "symptom_share": np.nan}]

    sym = symptom_mask(rgb, leaf, label) if label != "healthy" else None
    total = int(sym.sum()) if sym is not None else 0
    if sym is not None and total < cfg["min_symptom_pixels"]:
        return out  # cannot locate the lesions: only the safe leaf view
    rng = np.random.default_rng(int(hashlib.sha1(key.encode()).hexdigest()[:8], 16))
    short = min(bw, bh)
    tries, zooms = 0, []
    while len(zooms) < cfg["zoom_per_image"] and tries < 80:
        tries += 1
        side = int(rng.uniform(*cfg["zoom_side"]) * short)
        if side < 32:
            break
        cx = rng.uniform(x0 + side / 2, max(x0 + side / 2, x1 - side / 2))
        cy = rng.uniform(y0 + side / 2, max(y0 + side / 2, y1 - side / 2))
        a0, b0 = int(max(0, cx - side / 2)), int(max(0, cy - side / 2))
        a1, b1 = min(W, a0 + side), min(H, b0 + side)
        frac = float(leaf[b0:b1, a0:a1].mean())
        if frac < cfg["min_leaf_fraction"]:
            continue
        share = float(sym[b0:b1, a0:a1].sum() / total) if sym is not None else np.nan
        if sym is not None and share < cfg["min_symptom_share"]:
            continue
        if any(abs(a0 - z[0]) < side * 0.25 and abs(b0 - z[1]) < side * 0.25 for z in zooms):
            continue
        zooms.append((a0, b0, a1, b1))
        out.append({"view": "zoom", "box": (a0 / W, b0 / H, a1 / W, b1 / H), "leaf_fraction": frac,
                    "symptom_share": share})
    return out
