"""Thumbnail sheets for looking at images (PIL only)."""
from __future__ import annotations

from pathlib import Path


def thumb(path: Path, size: int):
    """Square thumbnail on a white background; a red tile if the file cannot be read."""
    from PIL import Image

    from .common import read_image

    try:
        img = read_image(path, draft=size)[0]
        img.thumbnail((size, size))
    except Exception:
        img = Image.new("RGB", (size, size), (200, 60, 60))
    canvas = Image.new("RGB", (size, size), (255, 255, 255))
    canvas.paste(img, ((size - img.width) // 2, (size - img.height) // 2))
    return canvas


def captioned_grid(items: list[tuple[Path, str]], out: Path, title: str, cols: int = 4, size: int = 180) -> None:
    """items: (image path, caption). Captions may contain newlines."""
    from PIL import Image, ImageDraw

    if not items:
        return
    cap_h = 46
    rows = (len(items) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * (size + 8), 24 + rows * (size + cap_h + 8)), (255, 255, 255))
    draw = ImageDraw.Draw(sheet)
    draw.text((6, 6), title, fill=(0, 0, 0))
    for k, (path, caption) in enumerate(items):
        x, y = (k % cols) * (size + 8), 24 + (k // cols) * (size + cap_h + 8)
        sheet.paste(thumb(path, size), (x, y))
        draw.multiline_text((x + 2, y + size + 3), caption, fill=(0, 0, 0))
    out.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out, quality=85)
