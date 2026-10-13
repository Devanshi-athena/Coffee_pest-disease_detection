"""Step 1: analyse the raw data before anything is labelled or split. Read-only on data/raw.

Answers two questions with numbers and pictures:
  1. Does each raw folder map to the right class?  -> per-folder stats + sample sheets
  2. Which split.max_hamming separates "same image" from "different image"?
     -> pair counts per hash distance, cluster outcomes per threshold, pictures of pairs

Writes to artifacts/inspect/ (or --out):
  summary.md              all tables
  images.csv              one row per file (size, mode, bytes, brightness, sharpness, status, phash)
  flagged.csv             corrupt / tiny files
  near_dup_pairs.csv      pairs within 6 bits (plain or rotation/flip-aware), with paths and labels
  pairs/*.jpg             side-by-side pairs per distance band, label conflicts, rotation/flip-only copies
  samples/<source>__<folder>.jpg   random thumbnails per raw folder
"""
import argparse
import hashlib
import io
import os
import sys
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.viz import thumb  # noqa: E402
from src.common import (  # noqa: E402
    label_lookup, load_config, map_label, phash_variants, read_image, resolve_path, run_parallel, scan_raw,
)

THRESHOLDS = [0, 2, 4, 6, 8, 10, 12]
BANDS = [(0, 0), (1, 2), (3, 4), (5, 6), (7, 8), (9, 10), (11, 12)]


# ---------------------------------------------------------------- per-image analysis

def analyse_one(path: str, hash_size: int, min_side: int) -> dict:
    import cv2
    from PIL import Image

    data = Path(path).read_bytes()
    rec = {"kb": round(len(data) / 1024, 1), "md5": hashlib.md5(data).hexdigest()}
    try:
        img, w, h, mode, fmt = read_image(io.BytesIO(data))
    except Exception as e:
        return {**rec, "status": "corrupt", "error": str(e)[:200]}
    rec.update(width=w, height=h, mode=mode, format=fmt)
    if min(w, h) < min_side:
        return {**rec, "status": "tiny"}
    gray = np.asarray(img.convert("L").resize((256, 256), Image.Resampling.BILINEAR), dtype=np.float32)
    rec["brightness"] = float(gray.mean())
    rec["sharpness"] = float(cv2.Laplacian(gray, cv2.CV_32F).var())  # low = blurry
    rec["hashes"] = phash_variants(img, hash_size)
    return {**rec, "status": "ok"}


# ---------------------------------------------------------------- near-duplicate search

def find_pairs(plain: np.ndarray, variants: np.ndarray | None, max_dist: int, block: int = 128):
    """Exact all-pairs search (vectorised, chunked). Returns arrays i, j (i < j), d_plain, d_any.

    d_any is the smallest distance between i and any rotation/flip of j (== d_plain without variants).
    """
    from tqdm import tqdm

    n = len(plain)
    out = {"i": [], "j": [], "d": [], "d_any": []}
    for s in tqdm(range(0, n, block), desc="pair search"):
        e = min(n, s + block)
        a = plain[s:e, None]
        d = np.bitwise_count(a ^ plain[None, s:]).astype(np.uint8)
        d_any = d.copy()
        if variants is not None:
            for v in variants[1:]:
                np.minimum(d_any, np.bitwise_count(a ^ v[None, s:]).astype(np.uint8), out=d_any)
        rows, cols = np.nonzero(d_any <= max_dist)
        keep = cols > rows  # both are offsets from s, so this keeps j > i and drops self-pairs
        rows, cols = rows[keep], cols[keep]
        out["i"].append(rows + s)
        out["j"].append(cols + s)
        out["d"].append(d[rows, cols])
        out["d_any"].append(d_any[rows, cols])
    return {k: np.concatenate(v) if v else np.array([], dtype=int) for k, v in out.items()}


def components(n: int, i: np.ndarray, j: np.ndarray) -> np.ndarray:
    """Group id per image, where i[k]--j[k] are edges (union of all linked pairs)."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components

    graph = coo_matrix((np.ones(len(i), dtype=np.int8), (i, j)), shape=(n, n))
    return connected_components(graph, directed=False)[1]


def cluster_stats(comp: np.ndarray, labels: np.ndarray, is_ext: np.ndarray) -> dict:
    df = pd.DataFrame({"c": comp, "label": labels, "ext": is_ext})
    g = df.groupby("c").agg(size=("label", "size"), labels=("label", "nunique"), ext=("ext", "sum"))
    multi = g[g["size"] > 1]
    mixed = multi[multi["labels"] > 1]
    leak = multi[(multi["ext"] > 0) & (multi["ext"] < multi["size"])]
    return {
        "distinct_after_grouping": len(g),
        "dup_groups": len(multi),
        "images_in_groups": int(multi["size"].sum()),
        "largest_group": int(g["size"].max()),
        "mixed_label_groups": len(mixed),
        "images_in_mixed": int(mixed["size"].sum()),
        "train_vs_external_groups": len(leak),
        "would_drop_from_train": int((leak["size"] - leak["ext"]).sum()),
    }


# ---------------------------------------------------------------- pictures

def pair_sheet(rows: pd.DataFrame, root: Path, out: Path, title: str, size: int = 150) -> None:
    from PIL import Image, ImageDraw

    if rows.empty:
        return
    text_w = 430
    sheet = Image.new("RGB", (size * 2 + text_w, 24 + len(rows) * (size + 6)), (255, 255, 255))
    draw = ImageDraw.Draw(sheet)
    draw.text((6, 6), title, fill=(0, 0, 0))
    for k, r in enumerate(rows.itertuples()):
        y = 24 + k * (size + 6)
        sheet.paste(thumb(root / r.path_a, size), (0, y))
        sheet.paste(thumb(root / r.path_b, size), (size + 2, y))
        info = (f"plain d={r.d}   rot/flip d={r.d_any}\n\nA [{r.label_a}] {r.path_a}\n\n"
                f"B [{r.label_b}] {r.path_b}")
        draw.multiline_text((size * 2 + 10, y + 8), info, fill=(0, 0, 0))
    out.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out, quality=85)


def sample_sheet(paths: list[Path], out: Path, title: str, cols: int = 6, size: int = 128) -> None:
    from PIL import Image, ImageDraw

    rows = (len(paths) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * (size + 4), 22 + rows * (size + 4)), (255, 255, 255))
    ImageDraw.Draw(sheet).text((4, 5), title, fill=(0, 0, 0))
    for k, p in enumerate(paths):
        sheet.paste(thumb(p, size), ((k % cols) * (size + 4), 22 + (k // cols) * (size + 4)))
    out.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out, quality=85)


# ---------------------------------------------------------------- main

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", help="path to config.yaml (default: configs/config.yaml)")
    ap.add_argument("--out", type=Path, help="output folder (default: <artifacts_dir>/inspect)")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--quick", action="store_true", help="only count files per folder, open nothing")
    ap.add_argument("--max-dist", type=int, default=12, help="largest hash distance to record pairs for")
    ap.add_argument("--no-rotations", action="store_true", help="skip the rotation/flip-aware pair search")
    ap.add_argument("--limit-per-folder", type=int, help="analyse a random sample per folder (trial runs)")
    ap.add_argument("--samples", type=int, default=24, help="thumbnails per folder in samples/")
    args = ap.parse_args()

    cfg = load_config(args.config)
    raw_dir = resolve_path(cfg, "raw_dir")
    out_dir = args.out or resolve_path(cfg, "artifacts_dir") / "inspect"
    seed, hash_size = cfg["split"]["seed"], cfg["split"]["hash_size"]
    found, missing, unknown = scan_raw(cfg)
    lookup = label_lookup(cfg)
    roles = {s: v["role"] for s, v in cfg["sources"].items()}

    for s in missing:
        print(f"WARNING: configured source '{s}' not found in {raw_dir}")
    for d in unknown:
        print(f"WARNING: folder '{d}' in raw dir is not a configured source; it will be ignored")
    if not found:
        sys.exit("\nERROR: none of the configured sources were found. Nothing to inspect.")

    rng = np.random.default_rng(seed)
    rows = []
    for source, folders in found.items():
        for folder, files in folders.items():
            cls, reason = map_label(folder, source, cfg, lookup)
            if args.limit_per_folder and len(files) > args.limit_per_folder:
                files = sorted(rng.choice(files, args.limit_per_folder, replace=False))
            for f in files:
                rows.append({"source": source, "folder": folder, "label": cls or f"SKIP ({reason})",
                             "role": roles[source], "path": Path(f).relative_to(raw_dir).as_posix()})
    df = pd.DataFrame(rows)

    counts = df.groupby(["source", "folder", "label"], sort=False).size().rename("files").reset_index()
    if args.quick:
        print("\n" + counts.to_string(index=False))
        return

    fn = partial(analyse_one, hash_size=hash_size, min_side=cfg["image"]["min_side"])
    results = run_parallel(fn, [str(raw_dir / p) for p in df["path"]], args.workers, desc="analysing")
    df = pd.concat([df, pd.DataFrame(results)], axis=1)
    ok = df["status"].eq("ok")
    nbits_hex = hash_size * hash_size // 4
    df["phash"] = [f"{h[0]:0{nbits_hex}x}" if isinstance(h, list) else "" for h in df["hashes"]]

    out_dir.mkdir(parents=True, exist_ok=True)
    report = [f"# Raw data inspection\n\nraw dir: `{raw_dir}`  \nimages: {len(df)}  \n"]

    def section(title: str, table, note: str = "") -> None:
        text = table if isinstance(table, str) else table.to_string()
        print(f"\n== {title}\n{text}")
        report.append(f"\n## {title}\n\n{note}\n```\n{text}\n```\n")

    # 1. per-folder overview
    per = df.groupby(["source", "folder", "label"], sort=False).agg(
        files=("path", "size"),
        corrupt=("status", lambda s: int((s == "corrupt").sum())),
        tiny=("status", lambda s: int((s == "tiny").sum())),
        formats=("format", lambda s: ",".join(sorted(s.dropna().unique()))),
        non_rgb=("mode", lambda s: int(s.notna().sum() - (s == "RGB").sum())),
        w_med=("width", "median"), h_med=("height", "median"),
        side_min=("width", lambda s: int(np.minimum(s, df.loc[s.index, "height"]).min())),
        side_max=("width", lambda s: int(np.maximum(s, df.loc[s.index, "height"]).max())),
        kb_med=("kb", "median"),
        bright_med=("brightness", "median"),
        sharp_med=("sharpness", "median"),
    ).round(1)
    section("Per folder", per.reset_index().to_string(index=False),
            "w/h_med: median size in px. side_min/max: smallest/largest image side. "
            "bright_med: mean gray 0-255. sharp_med: Laplacian variance at 256px (low = blurry).")

    # 2. class balance of usable images
    usable = df[ok & ~df["label"].str.startswith("SKIP")]
    bal = usable.pivot_table(index="label", columns="source", values="path", aggfunc="count", fill_value=0)
    bal = bal.reindex(index=[c for c in cfg["classes"] if c in bal.index])
    train_cols = [s for s in bal.columns if roles[s] == "train"]
    bal["TRAIN POOL"] = bal[train_cols].sum(axis=1)
    section("Usable images per class (before de-duplication)", bal)

    # 3. image size buckets
    sizes = df[ok].assign(min_side=np.minimum(df["width"], df["height"]))
    size_tab = pd.crosstab([sizes["source"], sizes["folder"]],
                           pd.cut(sizes["min_side"], [0, 128, 224, 256, 512, 1024, np.inf], right=False,
                                  labels=["<128", "128-223", "224-255", "256-511", "512-1023", ">=1024"]))
    section("Shorter image side (px)", size_tab, f"Training uses {cfg['image']['size']}px crops "
            f"from a {cfg['image']['resize']}px resize, so small images get upscaled.")

    # 4. exact duplicates
    dup = df[ok].groupby("md5").agg(n=("path", "size"), labels=("label", "nunique"),
                                    sources=("source", "nunique"), folders=("folder", "nunique"))
    dup = dup[dup["n"] > 1]
    exact = pd.Series({
        "groups of identical files": len(dup),
        "redundant copies": int((dup["n"] - 1).sum()),
        "groups across different classes": int((dup["labels"] > 1).sum()),
        "groups across different sources": int((dup["sources"] > 1).sum()),
    })
    section("Exact (byte-identical) duplicates", exact.to_string())

    # 5. near-duplicates. Byte-identical copies are collapsed first (one representative per md5), so the
    # pair search and pair counts are about distinct files; groups are then mapped back to every image.
    okdf = df[ok].reset_index(drop=True)
    n = len(okdf)
    reps = okdf.drop_duplicates("md5")
    rep_pos = reps.index.to_numpy()
    rep_of = okdf["md5"].map(pd.Series(rep_pos, index=reps["md5"].to_numpy())).to_numpy()
    copy_i = np.nonzero(rep_of != np.arange(n))[0]
    exact_edges = (copy_i, rep_of[copy_i])

    hashes = np.array(reps["hashes"].tolist(), dtype=np.uint64)  # (distinct files, 8)
    print(f"\nSearching near-duplicate pairs among {len(reps)} distinct files "
          f"({n} images; distance <= {args.max_dist})...")
    pr = find_pairs(hashes[:, 0], None if args.no_rotations else hashes.T, args.max_dist)
    a, b = rep_pos[pr["i"]], rep_pos[pr["j"]]  # positions in okdf
    label_c = okdf["label"].astype("category").cat.codes.to_numpy()
    source_c = okdf["source"].astype("category").cat.codes.to_numpy()
    folder_c = (okdf["source"] + "/" + okdf["folder"]).astype("category").cat.codes.to_numpy()
    is_ext = okdf["role"].eq("external_test").to_numpy()
    relation = np.select(
        [folder_c[a] == folder_c[b], source_c[a] == source_c[b], is_ext[a] != is_ext[b]],
        ["same folder", "same source, other folder", "train vs external"], "train source vs train source")
    pairs = pd.DataFrame({"a": a, "b": b, "d": pr["d"], "d_any": pr["d_any"],
                          "relation": pd.Categorical(relation),
                          "cross_label": label_c[a] != label_c[b], "cross_source": source_c[a] != source_c[b]})
    print(f"{len(pairs)} pairs of distinct files within distance {args.max_dist}")

    def with_paths(sub: pd.DataFrame) -> pd.DataFrame:
        sub = sub.copy()
        for side in ("a", "b"):
            for col in ("path", "label"):
                sub[f"{col}_{side}"] = okdf[col].to_numpy()[sub[side]]
        return sub

    def grouped(sel: pd.DataFrame) -> np.ndarray:
        return components(n, np.concatenate([exact_edges[0], sel["a"].to_numpy()]),
                          np.concatenate([exact_edges[1], sel["b"].to_numpy()]))

    csv_dist = 6
    close = pairs[np.minimum(pairs["d"], pairs["d_any"]) <= csv_dist]
    with_paths(close).drop(columns=["a", "b"]).to_csv(out_dir / "near_dup_pairs.csv", index=False)

    dist_index = range(0, args.max_dist + 1, 2)  # this hash always has 32 bits set, so distances are even
    plain = pairs[pairs["d"] <= args.max_dist]
    hist = pd.crosstab(plain["d"], plain["relation"]).reindex(dist_index, fill_value=0)
    hist["ALL"] = hist.sum(axis=1)
    hist["of which different class"] = plain[plain["cross_label"]].groupby("d").size().reindex(dist_index).fillna(0)
    hist = hist.astype(int)
    hist.index.name = "distance"
    section("Near-duplicate pairs (distinct files) by plain hash distance", hist,
            f"{hash_size * hash_size}-bit perceptual hash; unrelated images sit ~32 bits apart. Distances are always "
            "even (every hash has exactly half its bits set). Check pairs/*.jpg to see what each band looks like.")

    if not args.no_rotations:
        rot = pairs[pairs["d"] > pairs["d_any"]]
        rot_hist = pd.crosstab(rot["d_any"], rot["relation"]).reindex(dist_index, fill_value=0)
        rot_hist["ALL"] = rot_hist.sum(axis=1)
        rot_hist["of which different class"] = (rot[rot["cross_label"]].groupby("d_any").size()
                                                .reindex(dist_index).fillna(0).astype(int))
        rot_hist.index.name = "rot/flip distance"
        section("Extra pairs found only when rotations/flips are compared", rot_hist,
                "Rotated or mirrored copies that the plain hash misses (their plain distance is larger).")

    labels = okdf["label"].to_numpy()
    cfg_t = cfg["split"]["max_hamming"]
    sim, kept = [], {}  # kept: group ids at thresholds 0 and the configured one, for the per-folder table
    exact_comp = grouped(pairs.iloc[:0])
    sim.append({"compare": "exact copies only", "max_hamming": "-", **cluster_stats(exact_comp, labels, is_ext)})
    for mode, col in [("plain", "d")] + ([] if args.no_rotations else [("rot/flip", "d_any")]):
        for t in [t for t in THRESHOLDS if t <= args.max_dist]:
            comp = grouped(pairs[pairs[col] <= t])
            sim.append({"compare": mode, "max_hamming": t, **cluster_stats(comp, labels, is_ext)})
            if t in (0, cfg_t):
                kept[f"{mode} <={t}"] = comp
    sim = pd.DataFrame(sim).set_index(["compare", "max_hamming"])
    section("What grouping would do at each threshold (all images)", sim,
            "distinct_after_grouping: number of groups = effectively distinct images. dup_groups: groups with >1 "
            "image (each kept in one split). mixed_label_groups: groups whose images have different labels. "
            "would_drop_from_train: train-pool images grouped with an external test image.")

    # Distinct images per folder: what is really there once copies are collapsed.
    per_folder = okdf[["source", "folder", "label"]].copy()
    per_folder["exact"] = exact_comp
    for name, comp in kept.items():
        per_folder[name] = comp
    cols = ["exact", *kept]
    distinct = per_folder.groupby(["source", "folder", "label"], sort=False).agg(
        readable=("label", "size"), **{f"distinct ({c})": (c, "nunique") for c in cols})
    section("Distinct images per folder", distinct.reset_index().to_string(index=False),
            "readable: decodable, non-tiny files. distinct (exact): after collapsing byte-identical copies. "
            "distinct (rot/flip <=t): after also grouping images within t bits, rotations/flips included. "
            f"split.max_hamming in config.yaml is currently {cfg_t}.")

    group_col = f"rot/flip <={cfg_t}" if f"rot/flip <={cfg_t}" in kept else "exact"
    usable = per_folder[~per_folder["label"].str.startswith("SKIP")]
    cls_tab = usable.pivot_table(index="label", columns="source", values=group_col, aggfunc="nunique", fill_value=0)
    pool = usable[usable["source"].map(roles).eq("train")]
    cls_tab["TRAIN POOL"] = pool.groupby("label")[group_col].nunique()  # a group shared by sources counts once
    cls_tab = cls_tab.reindex(index=[c for c in cfg["classes"] if c in cls_tab.index]).fillna(0).astype(int)
    section(f"Distinct images per class (grouped at {group_col})", cls_tab,
            "This, not the raw file count, is how much independent data each class has.")

    # 6. pictures
    rs = np.random.RandomState(seed)

    def pick(frame: pd.DataFrame, k: int = 10) -> pd.DataFrame:
        return with_paths(frame.sample(min(k, len(frame)), random_state=rs).sort_values(["d", "d_any"]))

    for lo, hi in BANDS:
        if hi <= args.max_dist:
            band = plain[(plain["d"] >= lo) & (plain["d"] <= hi)]
            pair_sheet(pick(band), raw_dir, out_dir / "pairs" / f"plain_d{lo:02d}-{hi:02d}.jpg",
                       f"plain distance {lo}-{hi}: {len(band)} pairs (10 random)")
            cross = band[band["cross_source"]]
            pair_sheet(pick(cross), raw_dir, out_dir / "pairs" / f"cross_source_d{lo:02d}-{hi:02d}.jpg",
                       f"different sources, plain distance {lo}-{hi}: {len(cross)} pairs (10 random)")
    conflicts = close[close["cross_label"]]
    pair_sheet(pick(conflicts, 14), raw_dir, out_dir / "pairs" / "label_conflicts.jpg",
               f"near-identical (d<={csv_dist}) but different labels: {len(conflicts)} pairs (14 random)")
    if not args.no_rotations:
        rot_only = pairs[(pairs["d_any"] <= 4) & (pairs["d"] > args.max_dist)]
        pair_sheet(pick(rot_only, 12), raw_dir, out_dir / "pairs" / "rotation_flip_only.jpg",
                   f"rot/flip distance <=4 but plain distance >{args.max_dist}: {len(rot_only)} pairs (12 random)")
    for (source, folder), g in df[ok].groupby(["source", "folder"]):
        paths = [raw_dir / p for p in g["path"].sample(min(args.samples, len(g)), random_state=rs)]
        sample_sheet(paths, out_dir / "samples" / f"{source}__{folder}.jpg", f"{source}/{folder}  ({len(g)} images)")

    # 7. files
    df.drop(columns=["hashes"]).to_csv(out_dir / "images.csv", index=False)
    df[df["status"] != "ok"][["path", "status", "width", "height", "kb", "error"]
                             if "error" in df else ["path", "status", "width", "height", "kb"]].to_csv(
        out_dir / "flagged.csv", index=False)
    (out_dir / "summary.md").write_text("".join(report), encoding="utf-8")
    print(f"\nWrote {out_dir}/: summary.md, images.csv, flagged.csv, near_dup_pairs.csv, pairs/, samples/")


if __name__ == "__main__":
    main()
