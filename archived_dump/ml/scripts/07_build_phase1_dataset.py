"""Step 7: build the Phase 1 dataset (healthy vs rust) from the audit outputs.

Needs artifacts/inspect/images.csv (step 1), artifacts/audit/audit_images.csv + label_review.csv (step 6), and
Data_dump/RoCoLe*.zip (leaf annotations). Every file under data/raw appears in the manifest exactly once: kept
(with a split) or excluded (with a reason). Byte-identical copies are collapsed into one row (`copies`).

Splits (config.yaml -> phase1):
  train / val / test   JMuBEN, Uganda, BRACOL; grouped so a near-duplicate group never straddles splits,
                       stratified by source x label
  test_rocole          RoCoLe, completely unseen: phone photos of whole leaves in the field
  ood                  never trained on: other diseases (phoma, miner, cercospora) and out-of-scope images
                       (spider mite, BRACOL mixed-stress), for testing the "not sure" behaviour

Preprocessing per image: EXIF rotation -> RGB -> shrink so the shorter side <= store_shorter_side (never
enlarge) -> JPEG. The 224x224 model input is made at training/inference time.

Writes <phase1.output_dir>/: images/<split>/<label>/*.jpg, manifest.csv, crops.csv, DATASET.md, checks/*.jpg,
and <phase1.kaggle_zip>.
"""
import argparse
import hashlib
import io
import json
import os
import re
import sys
import zipfile
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "ml"))
from src.common import CONFIG_REL, die, load_config, resolve_path, run_parallel  # noqa: E402
from src.splits import assign_group_splits  # noqa: E402

ROCOLE_NAME = re.compile(r"^C(\d+)P(\d+)([HE])(\d+)$")
TRAIN_SPLITS = ("train", "val", "test")
ORDER = ["train", "val", "test", "test_rocole", "ood"]


# ---------------------------------------------------------------- RoCoLe leaf boxes

def rocole_leaf_boxes(raw_dir: Path) -> pd.DataFrame:
    """Leaf bounding box per RoCoLe photo, normalised to the EXIF-rotated image (what we store).

    The annotation polygons are in the stored pixel frame (before EXIF rotation), so they are rotated here
    with the same orientation the image gets.
    """
    from PIL import Image

    zf = next(iter((ROOT / "Data_dump").glob("RoCoLe*.zip")), None)
    if zf is None:
        print("WARNING: RoCoLe zip not found in Data_dump/; leaf boxes skipped")
        return pd.DataFrame(columns=["file", "leaf_x0", "leaf_y0", "leaf_x1", "leaf_y1"])
    z = zipfile.ZipFile(zf)
    ann = pd.read_csv(io.BytesIO(z.read(next(n for n in z.namelist() if n.endswith("RoCoLE-csv.csv")))))
    files = {p.name: p for p in (raw_dir / "rocole").rglob("*.jpg")}
    rows = []
    for _, r in ann.iterrows():
        f = files.get(r["External ID"])
        try:
            pts = json.loads(r["Label"])["Leaf"][0]["geometry"]
        except Exception:
            continue
        if f is None:
            continue
        with Image.open(f) as im:
            w, h = im.size
            orient = im.getexif().get(274, 1)
        x, y = np.array([p["x"] for p in pts], float), np.array([p["y"] for p in pts], float)
        if orient == 3:
            x, y = w - x, h - y
        elif orient == 6:      # displayed rotated 90 deg clockwise
            x, y, w, h = h - y, x, h, w
        elif orient == 8:      # displayed rotated 90 deg counter-clockwise
            x, y, w, h = y, w - x, h, w
        elif orient != 1:
            continue
        rows.append({"file": f.name, "leaf_x0": max(0, x.min() / w), "leaf_y0": max(0, y.min() / h),
                     "leaf_x1": min(1, x.max() / w), "leaf_y1": min(1, y.max() / h)})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- image processing

def process_one(job: tuple[str, str], shorter: int, quality: int, overwrite: bool) -> str:
    from PIL import Image, ImageOps

    src, dst = Path(job[0]), Path(job[1])
    if dst.exists() and not overwrite:
        return "exists"
    try:
        with Image.open(src) as im:
            im = ImageOps.exif_transpose(im).convert("RGB")
            scale = shorter / min(im.size)
            if scale < 1:  # never upscale
                im = im.resize((round(im.width * scale), round(im.height * scale)), Image.Resampling.LANCZOS)
            dst.parent.mkdir(parents=True, exist_ok=True)
            im.save(dst, "JPEG", quality=quality)
    except Exception as e:
        return f"error: {e}"
    return "ok"


def crop_views(job: tuple[str, str, str, tuple], ccfg: dict) -> list[dict]:
    """Views for one processed whole-leaf image: (path, processed file, label, annotated leaf box or None)."""
    from PIL import Image

    from src.crops import views_for

    key, file, label, box = job
    rgb = np.asarray(Image.open(file).convert("RGB"))
    return [{"processed_path": key, **v} for v in views_for(rgb, label, ccfg, key, box)]


# ---------------------------------------------------------------- main

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", help="path to config.yaml (default: configs/config.yaml)")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--overwrite", action="store_true", help="re-process images that already exist")
    ap.add_argument("--manifest-only", action="store_true", help="write manifest + card only (no images/crops/zip)")
    ap.add_argument("--no-zip", action="store_true")
    args = ap.parse_args()

    cfg = load_config(args.config)
    p1 = cfg["phase1"]
    raw = resolve_path(cfg, "raw_dir")
    art = resolve_path(cfg, "artifacts_dir")
    out = Path(cfg["_root"]) / p1["output_dir"]
    need = [art / "inspect" / "images.csv", art / "audit" / "audit_images.csv", art / "audit" / "label_review.csv"]
    for f in need:
        if not f.is_file():
            die(f"{f} not found. Run ml/scripts/01_inspect_raw.py and ml/scripts/06_audit_data.py first.")
    classes = p1["classes"]
    ext_sources = set(p1["external_test_sources"])

    allfiles = pd.read_csv(need[0], low_memory=False)
    audit = pd.read_csv(need[1], low_memory=False)
    review = pd.read_csv(need[2])

    # ---- one row per distinct readable file, plus one row per unreadable file
    keep_cols = ["path", "source", "folder", "label", "md5", "copies", "label_conflict", "group", "leaf_id",
                 "width", "height", "sharpness", "brightness"]
    m = audit[keep_cols].copy()
    m["leaf_id"] = m["leaf_id"].fillna("")
    bad = allfiles[allfiles["status"] != "ok"][["path", "source", "folder", "label", "status"]]
    m = pd.concat([m, bad.assign(copies=1, label_conflict=False, group=-1, leaf_id="")], ignore_index=True)
    m["status"] = m["status"].fillna("ok")

    # ---- labels: task label for healthy/rust, ood type for everything to be tested but never trained on
    m["label5"] = m["label"].where(~m["label"].str.startswith("SKIP"), "")
    m["task_label"] = m["label5"].map(p1["label_map"]).fillna("")
    m["label_basis"] = np.where(m["task_label"] != "", "folder", "")
    if p1.get("bracol_rust_from_symptoms"):
        bcsv = Path(cfg["_root"]) / "data" / "bracol_dataset.csv"
        if not bcsv.is_file():
            die(f"{bcsv} not found (needed for bracol_rust_from_symptoms).")
        b = pd.read_csv(bcsv).set_index("id")
        isb = m["source"].eq("bracol") & m["label5"].isin(["miner", "phoma", "cercospora"])
        has_rust = m.loc[isb, "path"].map(lambda p: int(Path(p).stem)).map(b["rust"]).fillna(0).astype(int).eq(1)
        idx = has_rust[has_rust].index
        m.loc[idx, ["task_label", "label_basis"]] = ["rust", "bracol symptom column: rust present"]
        print(f"BRACOL: {len(idx)} leaves with another predominant stress but rust present -> rust")
    m["ood_type"] = ""
    m.loc[m["task_label"].eq("") & m["label5"].isin(p1["ood"]["other_disease"]), "ood_type"] = "other_disease"
    m.loc[m["folder"].isin(p1["ood"]["out_of_scope"]), "ood_type"] = "out_of_scope"

    # ---- exclusions (first reason wins)
    m["exclude_reason"] = ""

    def exclude(mask, reason):
        m.loc[mask & m["exclude_reason"].eq(""), "exclude_reason"] = reason

    exclude(m["status"].eq("corrupt"), "unreadable or empty file")
    exclude(m["status"].eq("tiny"), "image too small")
    exclude(m["label_conflict"].astype(bool), "identical file filed under two labels")
    exclude(m["task_label"].eq("") & m["ood_type"].eq(""), "not a task class")
    for rule in p1.get("exclude_source_labels", []):
        exclude(m["source"].eq(rule["source"]) & m["label5"].eq(rule["label"]), rule["reason"])
    if p1.get("exclude_flagged_for_review"):
        exclude(m["path"].isin(set(review["path"])), "flagged for label review (artifacts/audit/label_review.csv)")
    # a near-duplicate group may only carry one label (task label or ood type), otherwise drop all of it
    live = m["exclude_reason"].eq("")
    m["any_label"] = np.where(m["task_label"] != "", m["task_label"], "ood:" + m["ood_type"])
    n_lab = m[live].groupby("group")["any_label"].nunique()
    exclude(live & m["group"].isin(n_lab[n_lab > 1].index), "near-identical images with different labels")
    if p1.get("exclude_rocole_side_disagreement"):
        rc = m["source"].eq("rocole") & m["leaf_id"].ne("") & m["status"].eq("ok")
        n_side = m[rc].groupby("leaf_id")["any_label"].nunique()
        exclude(rc & m["leaf_id"].isin(n_side[n_side > 1].index), "RoCoLe leaf: top and underside photos disagree")

    # ---- splits
    kept = m["exclude_reason"].eq("")
    plant = m["path"].map(lambda p: (lambda r: f"C{r[1]}P{r[2]}" if r else "")(ROCOLE_NAME.match(Path(p).stem)))
    m["plant_id"] = np.where(m["source"].eq("rocole"), plant, "")
    m["farm_id"] = m["plant_id"].str.extract(r"^(C\d+)")[0].fillna("")
    m["split_group"] = np.where(m["plant_id"] != "", "plant:" + m["plant_id"], "img:" + m["group"].astype(str))
    m["split"] = ""
    m.loc[kept & m["ood_type"].ne(""), "split"] = "ood"
    m.loc[kept & m["split"].eq("") & m["source"].isin(ext_sources), "split"] = "test_rocole"
    pool = kept & m["split"].eq("")
    m.loc[pool, "split"] = assign_group_splits(m.loc[pool, "split_group"],
                                               m.loc[pool, "source"] + "|" + m.loc[pool, "task_label"],
                                               p1["split"]["ratios"], p1["split"]["seed"])

    # ---- weights
    used = m["split"].ne("")
    m.loc[used, "group_size"] = m[used].groupby(["split", "group"])["path"].transform("size")
    m.loc[used, "sample_weight"] = 1.0 / m.loc[used, "group_size"]
    m["is_group_rep"] = False
    m.loc[m[used].drop_duplicates(["split", "group"]).index, "is_group_rep"] = True
    # balanced_weight: within each train/val/test split every class gets the same total weight, and within a
    # class every source gets the same share (groups still count once).
    m["balanced_weight"] = np.nan
    tv = m["split"].isin(TRAIN_SPLITS)
    cell = m[tv].groupby(["split", "task_label", "source"])["sample_weight"].transform("sum")
    n_src = m[tv].groupby(["split", "task_label"])["source"].transform("nunique")
    m.loc[tv, "balanced_weight"] = m.loc[tv, "sample_weight"] / cell / n_src / len(classes)

    # ---- RoCoLe leaf boxes (normalised, valid for the processed copy too since it is only scaled)
    boxes = rocole_leaf_boxes(raw)
    m["file"] = m["path"].map(lambda p: Path(p).name)
    m = m.merge(boxes, on="file", how="left")
    m.loc[~m["source"].eq("rocole"), ["leaf_x0", "leaf_y0", "leaf_x1", "leaf_y1"]] = np.nan

    # ---- integrity checks
    s = m[m["split"].isin(TRAIN_SPLITS)]
    for col in ("split_group", "group"):
        straddle = s.groupby(col)["split"].nunique()
        if (straddle > 1).any():
            die(f"{int((straddle > 1).sum())} {col}s straddle splits; split logic is broken.")
    shared = set(m.loc[m["split"].isin(TRAIN_SPLITS), "group"]) & set(m.loc[m["split"].isin(["test_rocole", "ood"]), "group"])
    if shared:
        die(f"{len(shared)} near-duplicate groups appear both in train/val/test and in test_rocole/ood.")
    if m.loc[m["split"].isin(TRAIN_SPLITS), "source"].isin(ext_sources).any():
        die("An external test source leaked into train/val/test.")
    if (m.loc[m["split"].isin(TRAIN_SPLITS), "task_label"] == "").any():
        die("A train/val/test row has no task label.")
    if m["path"].duplicated().any():
        die("Duplicate paths in manifest.")
    if int(m["copies"].sum()) != len(allfiles):
        die(f"Manifest covers {int(m['copies'].sum())} files but data/raw has {len(allfiles)}.")

    # ---- output paths
    m["processed_path"] = ""
    use = m["split"].ne("")
    m.loc[use, "processed_path"] = [
        f"images/{sp}/{lab or ood}/{src}_{hashlib.sha1(p.encode()).hexdigest()[:12]}.jpg"
        for sp, lab, ood, src, p in zip(m.loc[use, "split"], m.loc[use, "task_label"], m.loc[use, "ood_type"],
                                        m.loc[use, "source"], m.loc[use, "path"])]
    cols = ["processed_path", "split", "task_label", "ood_type", "label5", "label_basis", "source", "folder", "path",
            "exclude_reason", "copies", "group", "group_size", "sample_weight", "balanced_weight", "is_group_rep",
            "split_group", "leaf_id", "plant_id", "farm_id", "width", "height", "sharpness", "brightness",
            "leaf_x0", "leaf_y0", "leaf_x1", "leaf_y1", "md5"]
    m = m[cols].sort_values(["split", "task_label", "ood_type", "source", "path"]).reset_index(drop=True)
    out.mkdir(parents=True, exist_ok=True)

    crops = pd.DataFrame()
    if not args.manifest_only:
        use = m["processed_path"].ne("")  # recomputed after the re-sort above
        jobs = [(str(raw / p), str(out / q)) for p, q in zip(m.loc[use, "path"], m.loc[use, "processed_path"])]
        res = run_parallel(partial(process_one, shorter=p1["store_shorter_side"], quality=p1["jpeg_quality"],
                                   overwrite=args.overwrite), jobs, args.workers, desc="processing")
        errs = [(j[0], r) for j, r in zip(jobs, res) if r.startswith("error")]
        if errs:
            die(f"{len(errs)} images failed to process, e.g. {errs[:3]}")
        print(f"processed {len(jobs)} images ({res.count('exists')} already existed)")
        from src.common import phash_variants, read_image

        smp = m[use].sample(min(300, int(use.sum())), random_state=0)
        far = [p for p, q in zip(smp["path"], smp["processed_path"])
               if bin(phash_variants(read_image(raw / p)[0])[0] ^ phash_variants(read_image(out / q)[0])[0])
               .count("1") > 8]
        if far:
            die(f"{len(far)} of {len(smp)} processed images do not match their raw file, e.g. {far[:3]}")
        print(f"alignment check passed on {len(smp)} random images")

        # ---- views: whole-leaf images get controlled crops; everything else is used whole
        wl = use & m["source"].isin(p1["whole_leaf_sources"]) & m["split"].ne("ood")
        cjobs = [(q, str(out / q), lab, None if np.isnan(x0) else (x0, y0, x1, y1))
                 for q, lab, x0, y0, x1, y1 in zip(m.loc[wl, "processed_path"], m.loc[wl, "task_label"],
                                                   m.loc[wl, "leaf_x0"], m.loc[wl, "leaf_y0"],
                                                   m.loc[wl, "leaf_x1"], m.loc[wl, "leaf_y1"])]
        views = [v for vs in run_parallel(partial(crop_views, ccfg=p1["crops"]), cjobs, args.workers, desc="crops")
                 for v in vs]
        full = m.loc[use, ["processed_path"]].assign(view="full", box=[(0.0, 0.0, 1.0, 1.0)] * int(use.sum()),
                                                     leaf_fraction=np.nan, rust_share=np.nan)
        crops = pd.concat([full, pd.DataFrame(views)], ignore_index=True)
        crops[["x0", "y0", "x1", "y1"]] = pd.DataFrame(crops["box"].tolist(), index=crops.index).round(4)
        crops = crops.drop(columns="box").merge(m[["processed_path", "split", "task_label", "source"]],
                                                on="processed_path")
        # Train/val use leaf+zoom views for whole-leaf images; test splits are scored on the full image (deployed).
        crops["use_for_training"] = crops["split"].isin(["train", "val"]) & (
            ~crops["source"].isin(p1["whole_leaf_sources"]) | crops["view"].ne("full"))
        crops.to_csv(out / "crops.csv", index=False)
    m.to_csv(out / "manifest.csv", index=False)

    # ---- tables for the card / console
    u = m[m["split"].ne("")].assign(label=lambda d: np.where(d["task_label"] != "", d["task_label"],
                                                              "ood:" + d["ood_type"]))
    cnt = u.pivot_table(index=["split", "source"], columns="label", values="path", aggfunc="count", fill_value=0)
    cnt = cnt.reindex([sp for sp in ORDER if sp in u["split"].unique()], level=0)
    indep = u[u["is_group_rep"]].pivot_table(index="split", columns="label", values="path", aggfunc="count",
                                             fill_value=0).reindex([sp for sp in ORDER if sp in u["split"].unique()])
    tr = m[m["split"].eq("train")]
    bal = tr.groupby(["task_label", "source"]).agg(images=("path", "size"), independent=("is_group_rep", "sum"),
                                                   weight=("sample_weight", "sum"),
                                                   bweight=("balanced_weight", "sum"))
    bal["share of class (sample_weight)"] = (bal["weight"] / bal.groupby(level=0)["weight"].transform("sum")).round(2)
    bal["share of class (balanced_weight)"] = (bal["bweight"] / bal.groupby(level=0)["bweight"].transform("sum")).round(2)
    bal["class share of all (sample_weight)"] = (bal.groupby(level=0)["weight"].transform("sum")
                                                 / bal["weight"].sum()).round(2)
    bal = bal.drop(columns=["weight", "bweight"])
    src_classes = tr.groupby("source")["task_label"].agg(lambda s: ", ".join(sorted(s.unique())))
    excl = m[m["exclude_reason"].ne("")].groupby(["exclude_reason", "source"])["copies"].agg(["size", "sum"])
    excl.columns = ["distinct files", "files incl. copies"]
    relab = m[m["label_basis"].str.startswith("bracol")].groupby(["split", "label5"]).size()
    view_tab = (crops[crops["use_for_training"]].groupby(["split", "source", "task_label", "view"]).size()
                .unstack(fill_value=0) if len(crops) else pd.DataFrame())
    no_zoom = (crops[crops["split"].isin(["train", "val"]) & crops["source"].isin(p1["whole_leaf_sources"])]
               .groupby(["task_label", "processed_path"])["view"].apply(lambda v: "zoom" not in set(v))
               .groupby(level=0).mean().round(2) if len(crops) else pd.Series(dtype=float))

    card = f"""# Phase 1 dataset: {p1['name']}

Built by `ml/scripts/07_build_phase1_dataset.py` from the data audit (see `docs/DATA_AUDIT.md`).

**Task.** {' vs '.join(classes)}. Nothing else is trained on.

**Splits.**
- `train` / `val` / `test`: JMuBEN, Uganda and BRACOL, split {p1['split']['ratios']} by near-duplicate group,
  stratified by source x label.
- `test_rocole`: RoCoLe, completely unseen (phone photos of whole leaves in the field).
- `ood`: never trained on. `other_disease` = phoma, miner, cercospora (all sources); `out_of_scope` = spider
  mite and BRACOL mixed-stress leaves. For testing the "not sure" behaviour.

Checks passed: no near-duplicate group appears in two of train/val/test, none is shared between
train/val/test and test_rocole/ood, no RoCoLe image is in train/val/test, every raw file is accounted for,
and 300 random processed images match their raw file.

**Preprocessing.** EXIF rotation -> RGB -> shorter side shrunk to <= {p1['store_shorter_side']} px (never enlarged)
-> JPEG q{p1['jpeg_quality']}. The 224x224 model input is made at training/inference time. No intensity changes.

**Weights.** `sample_weight` = 1 / (images of the same visual group in that split): augmented copies count once.
`balanced_weight` additionally gives every class the same total and every source the same share within a class.

**Views (`crops.csv`).** Close-ups (JMuBEN, Uganda) are used whole (`full`). Whole-leaf images (BRACOL) are
trained on precomputed views instead: `leaf` (leaf box + {p1['crops']['leaf_margin']:.0%} margin) and up to
{p1['crops']['zoom_per_image']} square `zoom` crops that are >= {p1['crops']['min_leaf_fraction']:.0%} leaf tissue;
for rust leaves a zoom must also hold >= {p1['crops']['min_rust_share']:.0%} of the leaf's rust-coloured pixels.
Draw an image by weight, then one of its views, so views do not add weight. Test splits are scored on `full`
(what the app does); RoCoLe also has annotated `leaf` views for a secondary score.

**Labels.** `task_label` = healthy/rust, `ood_type` for OOD rows, `label5` = original 5-class label,
`label_basis` = where the label came from. BRACOL leaves whose predominant stress is not rust but which have
rust present are labelled rust:

```
{relab.to_string() if len(relab) else 'none'}
```

## Images per split, source and label

```
{cnt.to_string()}
```

## Independent images (one per near-duplicate group)

```
{indep.to_string()}
```

## Source balance in train

```
{bal.to_string()}
```

Classes each training source contributes: {'; '.join(f'{k}: {v}' for k, v in src_classes.items())}.

## Training views (train/val)

```
{view_tab.to_string() if len(view_tab) else 'n/a'}
```

Share of whole-leaf train/val images with no zoom crop (leaf view only): {no_zoom.to_dict() if len(no_zoom) else 'n/a'}

## Excluded

```
{excl.to_string()}
```
"""
    (out / "DATASET.md").write_text(card, encoding="utf-8")

    # ---- visual checks
    if not args.manifest_only:
        from PIL import Image, ImageDraw

        from src.viz import captioned_grid

        (out / "checks").mkdir(exist_ok=True)
        rs = np.random.RandomState(0)
        colours = {"leaf": "blue", "zoom": "red"}
        for lab in classes:
            pick = crops[crops["source"].eq("bracol") & crops["split"].eq("train") & crops["task_label"].eq(lab)
                         & crops["view"].ne("full")]["processed_path"].drop_duplicates()
            tiles = []
            for q in pick.sample(min(8, len(pick)), random_state=rs):
                im = Image.open(out / q).convert("RGB")
                W, H = im.size
                d = ImageDraw.Draw(im)
                for r in crops[crops["processed_path"].eq(q) & crops["view"].ne("full")].itertuples():
                    d.rectangle([r.x0 * W, r.y0 * H, r.x1 * W, r.y1 * H], outline=colours[r.view], width=5)
                im.thumbnail((360, 360))
                tiles.append(im)
            if tiles:
                cols_n = 4
                sheet = Image.new("RGB", (cols_n * 366, ((len(tiles) + cols_n - 1) // cols_n) * 186), "white")
                for k, t in enumerate(tiles):
                    sheet.paste(t, ((k % cols_n) * 366, (k // cols_n) * 186))
                sheet.save(out / "checks" / f"bracol_views_{lab}.jpg", quality=85)
        for sp in ORDER:
            sub = u[u["split"].eq(sp)]
            if sub.empty:
                continue
            smp = sub.sample(frac=1, random_state=0).groupby("label").head(8)
            captioned_grid([(out / r.processed_path, f"{r.label}\n{r.source}/{r.folder}"[:40])
                            for r in smp.itertuples()], out / "checks" / f"samples_{sp}.jpg",
                           f"{sp}: random examples per label", cols=8, size=130)

    # ---- zip for Kaggle/Colab
    if not args.manifest_only and not args.no_zip:
        zpath = Path(cfg["_root"]) / p1["kaggle_zip"]
        files = [out / q for q in m.loc[m["processed_path"].ne(""), "processed_path"]]
        files += [out / "manifest.csv", out / "crops.csv", out / "DATASET.md", Path(cfg["_root"]) / CONFIG_REL]
        files += sorted((Path(cfg["_root"]) / "ml" / "src").glob("*.py"))
        with zipfile.ZipFile(zpath, "w") as zf:
            for f in files:
                zf.write(f, f.relative_to(Path(cfg["_root"])).as_posix(),
                         compress_type=zipfile.ZIP_STORED if f.suffix == ".jpg" else zipfile.ZIP_DEFLATED)
        print(f"Wrote {zpath} ({zpath.stat().st_size / 1e6:.0f} MB, {len(files)} files)")

    print("\n" + cnt.to_string() + "\n\nindependent:\n" + indep.to_string() + "\n\ntrain source balance:\n"
          + bal.to_string() + "\n\ntraining views:\n" + (view_tab.to_string() if len(view_tab) else "n/a")
          + f"\nleaf-view-only share: {no_zoom.to_dict() if len(no_zoom) else 'n/a'}"
          + "\n\nexcluded:\n" + excl.to_string())
    print(f"\nWrote {out}: manifest.csv, DATASET.md" + ("" if args.manifest_only else ", crops.csv, images/, checks/"))


if __name__ == "__main__":
    main()
