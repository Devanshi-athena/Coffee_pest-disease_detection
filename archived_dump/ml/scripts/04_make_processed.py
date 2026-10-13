"""Step 4: resize images into data/processed/<split>/<label>/ and zip everything Kaggle/Colab needs.

Each image: EXIF-rotate, RGB, resize shorter side to image.resize, center-crop to a square, save JPEG.
The zip contains data/processed/ (incl. its metadata.csv), configs/config.yaml and ml/src/,
so the notebook needs only this one upload.
"""
import argparse
import hashlib
import os
import sys
import zipfile
from functools import partial
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.common import CONFIG_REL, die, load_config, resolve_path, run_parallel  # noqa: E402


def process_one(job: tuple[str, str], size: int, quality: int, overwrite: bool) -> str:
    from PIL import Image, ImageOps

    src, dst = Path(job[0]), Path(job[1])
    if dst.exists() and not overwrite:
        return "exists"
    try:
        with Image.open(src) as im:
            im = ImageOps.exif_transpose(im).convert("RGB")
            w, h = im.size
            scale = size / min(w, h)
            im = im.resize((max(size, round(w * scale)), max(size, round(h * scale))), Image.Resampling.BILINEAR)
            left, top = (im.width - size) // 2, (im.height - size) // 2
            im = im.crop((left, top, left + size, top + size))
            dst.parent.mkdir(parents=True, exist_ok=True)
            im.save(dst, "JPEG", quality=quality)
    except Exception as e:
        return f"error: {e}"
    return "ok"


def build_zip(cfg: dict, processed_dir: Path, image_paths: list[str], zip_path: Path) -> None:
    root = Path(cfg["_root"])
    # Only files in the current metadata, so stale images from an earlier split are left out.
    files = [processed_dir / p for p in image_paths] + [processed_dir / "metadata.csv", root / CONFIG_REL]
    files += sorted((root / "ml" / "src").glob("*.py"))
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    # JPEGs are already compressed, so store them; deflate only the text files.
    with zipfile.ZipFile(zip_path, "w") as zf:
        for p in files:
            method = zipfile.ZIP_STORED if p.suffix == ".jpg" else zipfile.ZIP_DEFLATED
            zf.write(p, p.relative_to(root).as_posix(), compress_type=method)
    print(f"Wrote {zip_path} ({zip_path.stat().st_size / 1e6:.1f} MB, {len(files)} files)")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", help="path to config.yaml (default: configs/config.yaml)")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--overwrite", action="store_true", help="re-process images that already exist")
    ap.add_argument("--no-zip", action="store_true", help="skip building the Kaggle/Colab zip")
    args = ap.parse_args()

    cfg = load_config(args.config)
    raw_dir = resolve_path(cfg, "raw_dir")
    processed_dir = resolve_path(cfg, "processed_dir")
    csv_path = resolve_path(cfg, "metadata_csv")
    if not csv_path.is_file():
        die(f"{csv_path} not found. Run 02_build_metadata.py and 03_cluster_split.py first.")
    df = pd.read_csv(csv_path, dtype={"hash": str})
    if "split" not in df or df["split"].isna().all():
        die(f"{csv_path} has no splits yet. Run 03_cluster_split.py first.")

    df = df[~df["split"].fillna("excluded").str.startswith("excluded")].copy()
    if df.empty:
        die("No rows to process after removing excluded rows.")

    # Name by a hash of the raw path so files from different sources never collide.
    df["out_path"] = [
        f"{split}/{label}/{source}_{hashlib.sha1(p.encode()).hexdigest()[:12]}.jpg"
        for split, label, source, p in zip(df["split"], df["label"], df["source"], df["path"])
    ]
    jobs = [(str(raw_dir / p), str(processed_dir / o)) for p, o in zip(df["path"], df["out_path"])]
    fn = partial(process_one, size=cfg["image"]["resize"], quality=cfg["image"]["jpeg_quality"],
                 overwrite=args.overwrite)
    results = run_parallel(fn, jobs, args.workers, desc="resizing")

    errors = [(j[0], r) for j, r in zip(jobs, results) if r.startswith("error")]
    for src, msg in errors[:10]:
        print(f"  {src}: {msg}")
    ok = [not r.startswith("error") for r in results]
    out = df.loc[ok, ["out_path", "label", "source", "split", "cluster_id", "hash"]].rename(columns={"out_path": "path"})
    out.to_csv(processed_dir / "metadata.csv", index=False)
    print(f"processed {sum(ok)} images ({results.count('exists')} already existed), {len(errors)} errors")
    print(f"Wrote {processed_dir / 'metadata.csv'}")

    if not args.no_zip:
        build_zip(cfg, processed_dir, out["path"].tolist(), resolve_path(cfg, "kaggle_zip"))
        print("Next: upload the zip as a Kaggle dataset (or to Google Drive for Colab) and open "
              "ml/notebooks/train_kaggle.ipynb.")


if __name__ == "__main__":
    main()
