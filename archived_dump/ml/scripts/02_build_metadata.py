"""Step 2: map raw folders to the 5 classes, compute perceptual hashes, write data/metadata.csv.

Columns: path,label,source,hash,cluster_id,split,raw_label
(cluster_id and split are left empty; 03_cluster_split.py fills them).
"""
import argparse
import os
import sys
from functools import partial
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.common import (  # noqa: E402
    die, label_lookup, load_config, map_label, phash_variants, read_image, resolve_path, run_parallel, scan_raw,
)

COLUMNS = ["path", "label", "source", "hash", "cluster_id", "split", "raw_label"]


def hash_image(path: Path, hash_size: int, min_side: int) -> tuple[str, str]:
    """Returns (status, hex_hash). Same image reading and hash as 01_inspect_raw.py."""
    try:
        img, w, h, _, _ = read_image(path)
    except Exception:
        return "corrupt", ""
    if min(w, h) < min_side:
        return "tiny", ""
    return "ok", f"{phash_variants(img, hash_size)[0]:0{hash_size * hash_size // 4}x}"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", help="path to config.yaml (default: configs/config.yaml)")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    args = ap.parse_args()

    cfg = load_config(args.config)
    raw_dir = resolve_path(cfg, "raw_dir")
    out_csv = resolve_path(cfg, "metadata_csv")
    found, missing, _ = scan_raw(cfg)
    lookup = label_lookup(cfg)
    for s in missing:
        print(f"WARNING: source '{s}' not found, skipping")

    rows, skipped_dirs = [], []
    for source, classes in found.items():
        for raw_cls, files in classes.items():
            cls, reason = map_label(raw_cls, source, cfg, lookup)
            if cls is None:
                skipped_dirs.append((source, raw_cls, reason, len(files)))
                continue
            for f in files:
                rows.append({"path": f.relative_to(raw_dir).as_posix(), "label": cls,
                             "source": source, "raw_label": raw_cls})
    if not rows:
        die("No images matched any class. Run 01_inspect_raw.py to see how folders were mapped.")

    fn = partial(hash_image, hash_size=cfg["split"]["hash_size"], min_side=cfg["image"]["min_side"])
    results = run_parallel(fn, [raw_dir / r["path"] for r in rows], args.workers, desc="hashing")

    bad = {"corrupt": 0, "tiny": 0}
    kept = []
    for r, (status, h) in zip(rows, results):
        if status != "ok":
            bad[status] += 1
            continue
        kept.append({**r, "hash": h, "cluster_id": "", "split": ""})

    df = pd.DataFrame(kept, columns=COLUMNS).sort_values(["source", "label", "path"])
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)

    for source, raw_cls, reason, n in skipped_dirs:
        print(f"skipped folder {source}/{raw_cls} ({reason}, {n} files)")
    print(f"skipped files: {bad['corrupt']} corrupt, {bad['tiny']} tiny")
    print("\nImages per source/class:")
    print(df.pivot_table(index="source", columns="label", values="path", aggfunc="count", fill_value=0)
            .reindex(columns=cfg["classes"], fill_value=0).to_string())
    print(f"\nWrote {len(df)} rows to {out_csv}")


if __name__ == "__main__":
    main()
