"""Step 8: build the Phase 2 dataset (healthy / rust / other disease) from the audit outputs.

Differences from Phase 1 (07_build_phase1_dataset.py):
  - three classes: other = phoma, miner or cercospora (a leaf disease that is not rust)
  - RoCoLe split by whole farm: some farms train, some validate, the rest form `test_locked`, which is meant to be
    evaluated once at the end and never used for any choice
  - RoCoLe leaf polygons (from the dataset's annotations) give the leaf mask for its field photos
  - zoom crops of rust/other leaves must contain a real share of the leaf's symptom pixels

Splits:
  train / val / test   JMuBEN, JMuBEN2, Uganda, BRACOL (grouped by near-duplicate group, stratified source x label)
                       + RoCoLe train/val farms in train/val
  test_locked          RoCoLe farms held out completely
  ood                  never trained on: RoCoLe spider mite, BRACOL mixed-stress (tests "not sure")

Writes <phase2.output_dir>/: images/, manifest.csv, crops.csv, DATASET.md, checks/, and <phase2.kaggle_zip>.
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
FIT_SPLITS = ("train", "val")
INDOMAIN_SPLITS = ("train", "val", "test")
ORDER = ["train", "val", "test", "test_locked", "ood"]


def rocole_leaf_polygons(raw_dir: Path) -> pd.DataFrame:
    """Leaf polygon + box per RoCoLe photo, normalised to the EXIF-rotated image (what we store).

    The annotation polygons are in the stored pixel frame (before EXIF rotation); they are rotated with the image.
    """
    from PIL import Image

    zf = next(iter((ROOT / "Data_dump").glob("RoCoLe*.zip")), None)
    if zf is None:
        die("RoCoLe zip not found in Data_dump/ (needed for the leaf polygons).")
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
        elif orient == 6:
            x, y, w, h = h - y, x, h, w
        elif orient == 8:
            x, y, w, h = y, w - x, h, w
        elif orient != 1:
            continue
        xn, yn = np.clip(x / w, 0, 1), np.clip(y / h, 0, 1)
        rows.append({"file": f.name, "leaf_x0": xn.min(), "leaf_y0": yn.min(), "leaf_x1": xn.max(),
                     "leaf_y1": yn.max(), "leaf_poly": json.dumps([[round(a, 4), round(b, 4)] for a, b in zip(xn, yn)])})
    return pd.DataFrame(rows)


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


def crop_views(job: tuple, ccfg: dict) -> list[dict]:
    from PIL import Image

    from src.crops import views_for_v2

    key, file, label, poly = job
    rgb = np.asarray(Image.open(file).convert("RGB"))
    return [{"processed_path": key, **v} for v in views_for_v2(rgb, label, ccfg, key, poly)]


def farm_units(rows: pd.DataFrame) -> dict:
    """Farms that share a near-duplicate photo must land in the same split; returns {farm: unit id}."""
    parent = {f: f for f in rows["farm_id"].unique()}

    def find(f):
        while parent[f] != f:
            parent[f] = parent[parent[f]]
            f = parent[f]
        return f

    for _, farms in rows.groupby("group")["farm_id"]:
        fs = sorted(set(farms))
        for f in fs[1:]:
            parent[find(f)] = find(fs[0])
    return {f: find(f) for f in parent}


def split_rocole_farms(farm_stats: pd.DataFrame, units: dict, n: dict, seed: int, tries: int = 20000) -> dict:
    """Assigns linked-farm units to splits so that each split's healthy/rust mix, image share and farm count stay
    close to the targets (n = farm counts per split). Returns {farm: split}."""
    farms = farm_stats.index.to_list()
    if sum(n.values()) != len(farms):
        die(f"rocole_farm_split {n} needs {sum(n.values())} farms, found {len(farms)}: {farms}")
    unit_ids = sorted(set(units[f] for f in farms))
    members = {u: [f for f in farms if units[f] == u] for u in unit_ids}
    total = farm_stats.sum()
    rust_share = total["rust"] / total.sum()
    splits = list(n)
    target_img = {s: n[s] / len(farms) for s in splits}
    rng = np.random.default_rng(seed)
    best, best_score = None, np.inf
    for _ in range(tries):
        choice = rng.integers(0, len(splits), len(unit_ids))
        assign = {f: splits[c] for u, c in zip(unit_ids, choice) for f in members[u]}
        score = 0.0
        for s in splits:
            fs = [f for f, v in assign.items() if v == s]
            if not fs:
                score = np.inf
                break
            sub = farm_stats.loc[fs].sum()
            if (sub == 0).any():  # every split needs both classes
                score = np.inf
                break
            score += (abs(sub["rust"] / sub.sum() - rust_share) + abs(sub.sum() / total.sum() - target_img[s])
                      + 0.02 * abs(len(fs) - n[s]))
        if score < best_score:
            best, best_score = assign, score
    if best is None:
        die("Could not find a valid RoCoLe farm split.")
    return best


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", help="path to config.yaml (default: configs/config.yaml)")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--manifest-only", action="store_true", help="write manifest + card only (no images/crops/zip)")
    ap.add_argument("--no-zip", action="store_true")
    args = ap.parse_args()

    cfg = load_config(args.config)
    p2 = cfg["phase2"]
    raw = resolve_path(cfg, "raw_dir")
    art = resolve_path(cfg, "artifacts_dir")
    out = Path(cfg["_root"]) / p2["output_dir"]
    need = [art / "inspect" / "images.csv", art / "audit" / "audit_images.csv", art / "audit" / "label_review.csv"]
    for f in need:
        if not f.is_file():
            die(f"{f} not found. Run ml/scripts/01_inspect_raw.py and ml/scripts/06_audit_data.py first.")
    classes = p2["classes"]

    allfiles = pd.read_csv(need[0], low_memory=False)
    audit = pd.read_csv(need[1], low_memory=False)
    review = pd.read_csv(need[2])

    keep_cols = ["path", "source", "folder", "label", "md5", "copies", "label_conflict", "group", "leaf_id",
                 "width", "height", "sharpness", "brightness"]
    m = audit[keep_cols].copy()
    m["leaf_id"] = m["leaf_id"].fillna("")
    bad = allfiles[allfiles["status"] != "ok"][["path", "source", "folder", "label", "status"]]
    m = pd.concat([m, bad.assign(copies=1, label_conflict=False, group=-1, leaf_id="")], ignore_index=True)
    m["status"] = m["status"].fillna("ok")

    # ---- labels
    m["label5"] = m["label"].where(~m["label"].str.startswith("SKIP"), "")
    m["task_label"] = m["label5"].map(p2["label_map"]).fillna("")
    m["label_basis"] = np.where(m["task_label"] != "", "folder", "")
    if p2.get("bracol_rust_from_symptoms"):
        b = pd.read_csv(Path(cfg["_root"]) / "data" / "bracol_dataset.csv").set_index("id")
        isb = m["source"].eq("bracol") & m["label5"].isin(["miner", "phoma", "cercospora"])
        has_rust = m.loc[isb, "path"].map(lambda p: int(Path(p).stem)).map(b["rust"]).fillna(0).astype(int).eq(1)
        idx = has_rust[has_rust].index
        m.loc[idx, ["task_label", "label_basis"]] = ["rust", "bracol symptom column: rust present"]
        print(f"BRACOL: {len(idx)} leaves with another predominant stress but rust present -> rust")
    m["ood_type"] = np.where(m["folder"].isin(p2["out_of_scope"]), "out_of_scope", "")

    # ---- exclusions (first reason wins)
    m["exclude_reason"] = ""

    def exclude(mask, reason):
        m.loc[mask & m["exclude_reason"].eq(""), "exclude_reason"] = reason

    exclude(m["status"].eq("corrupt"), "unreadable or empty file")
    exclude(m["status"].eq("tiny"), "image too small")
    exclude(m["label_conflict"].astype(bool), "identical file filed under two labels")
    exclude(m["task_label"].eq("") & m["ood_type"].eq(""), "not a task class")
    for rule in p2.get("exclude_source_labels", []):
        exclude(m["source"].eq(rule["source"]) & m["label5"].eq(rule["label"]), rule["reason"])
    if p2.get("exclude_flagged_for_review"):
        exclude(m["path"].isin(set(review["path"])), "flagged for label review (artifacts/audit/label_review.csv)")
    live = m["exclude_reason"].eq("")
    m["any_label"] = np.where(m["task_label"] != "", m["task_label"], "ood:" + m["ood_type"])
    n_lab = m[live].groupby("group")["any_label"].nunique()
    exclude(live & m["group"].isin(n_lab[n_lab > 1].index), "near-identical images with different labels")
    if p2.get("exclude_rocole_side_disagreement"):
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

    rc_kept = kept & m["source"].eq("rocole") & m["split"].eq("")
    farm_stats = m[rc_kept].pivot_table(index="farm_id", columns="task_label", values="path", aggfunc="count",
                                        fill_value=0)[["healthy", "rust"]]
    # every RoCoLe row with a group (incl. OOD and excluded ones) links farms, so no photo can cross splits
    units = farm_units(m[m["source"].eq("rocole") & m["farm_id"].ne("") & m["group"].ge(0)])
    linked = {u: sorted(f for f, v in units.items() if v == u) for u in set(units.values())}
    print("RoCoLe farm units (farms sharing a near-duplicate photo stay together):",
          sorted(v for v in linked.values() if len(v) > 1))
    farm_split = split_rocole_farms(farm_stats, units, p2["rocole_farm_split"], p2["split"]["seed"])
    m.loc[rc_kept, "split"] = m.loc[rc_kept, "farm_id"].map(farm_split)

    pool = kept & m["split"].eq("")
    m.loc[pool, "split"] = assign_group_splits(m.loc[pool, "split_group"],
                                               m.loc[pool, "source"] + "|" + m.loc[pool, "task_label"],
                                               p2["split"]["ratios"], p2["split"]["seed"])

    # ---- weights
    used = m["split"].ne("")
    m.loc[used, "group_size"] = m[used].groupby(["split", "group"])["path"].transform("size")
    m.loc[used, "sample_weight"] = 1.0 / m.loc[used, "group_size"]
    m["is_group_rep"] = False
    m.loc[m[used].drop_duplicates(["split", "group"]).index, "is_group_rep"] = True
    m["balanced_weight"] = np.nan
    tv = m["split"].isin(INDOMAIN_SPLITS)
    cell = m[tv].groupby(["split", "task_label", "source"])["sample_weight"].transform("sum")
    n_src = m[tv].groupby(["split", "task_label"])["source"].transform("nunique")
    m.loc[tv, "balanced_weight"] = m.loc[tv, "sample_weight"] / cell / n_src / len(classes)

    # ---- RoCoLe polygons
    polys = rocole_leaf_polygons(raw)
    m["file"] = m["path"].map(lambda p: Path(p).name)
    m = m.merge(polys, on="file", how="left")
    m.loc[~m["source"].eq("rocole"), ["leaf_x0", "leaf_y0", "leaf_x1", "leaf_y1", "leaf_poly"]] = np.nan

    # ---- integrity checks
    s = m[m["split"].isin(INDOMAIN_SPLITS)]
    for col in ("split_group", "group"):
        straddle = s.groupby(col)["split"].nunique()
        if (straddle > 1).any():
            die(f"{int((straddle > 1).sum())} {col}s straddle train/val/test.")
    farms = m[m["source"].eq("rocole") & m["split"].isin(["train", "val", "test_locked"])].groupby("farm_id")["split"].nunique()
    if (farms > 1).any():
        die("A RoCoLe farm appears in more than one split.")
    if m.loc[m["split"].eq("test_locked"), "source"].ne("rocole").any():
        die("test_locked must contain RoCoLe only.")
    fit_groups = set(m.loc[m["split"].isin(INDOMAIN_SPLITS), "group"])
    for held in ("test_locked", "ood"):
        shared = fit_groups & set(m.loc[m["split"].eq(held), "group"])
        if shared:
            die(f"{len(shared)} near-duplicate groups appear in train/val/test and in {held}.")
    if (m.loc[m["split"].isin(INDOMAIN_SPLITS + ("test_locked",)), "task_label"] == "").any():
        die("A labelled split has rows without a task label.")
    if m["path"].duplicated().any():
        die("Duplicate paths in manifest.")
    if int(m["copies"].sum()) != len(allfiles):
        die(f"Manifest covers {int(m['copies'].sum())} files but data/raw has {len(allfiles)}.")

    # ---- outputs
    m["processed_path"] = ""
    use = m["split"].ne("")
    m.loc[use, "processed_path"] = [
        f"images/{sp}/{lab or ood}/{src}_{hashlib.sha1(p.encode()).hexdigest()[:12]}.jpg"
        for sp, lab, ood, src, p in zip(m.loc[use, "split"], m.loc[use, "task_label"], m.loc[use, "ood_type"],
                                        m.loc[use, "source"], m.loc[use, "path"])]
    cols = ["processed_path", "split", "task_label", "ood_type", "label5", "label_basis", "source", "folder", "path",
            "exclude_reason", "copies", "group", "group_size", "sample_weight", "balanced_weight", "is_group_rep",
            "split_group", "leaf_id", "plant_id", "farm_id", "width", "height", "sharpness", "brightness",
            "leaf_x0", "leaf_y0", "leaf_x1", "leaf_y1", "leaf_poly", "md5"]
    m = m[cols].sort_values(["split", "task_label", "ood_type", "source", "path"]).reset_index(drop=True)
    out.mkdir(parents=True, exist_ok=True)

    crops = pd.DataFrame()
    if not args.manifest_only:
        use = m["processed_path"].ne("")  # recomputed after the re-sort above
        jobs = [(str(raw / p), str(out / q)) for p, q in zip(m.loc[use, "path"], m.loc[use, "processed_path"])]
        res = run_parallel(partial(process_one, shorter=p2["store_shorter_side"], quality=p2["jpeg_quality"],
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

        wl = use & m["source"].isin(p2["whole_leaf_sources"]) & m["split"].ne("ood")
        cjobs = [(q, str(out / q), lab, json.loads(poly) if isinstance(poly, str) else None)
                 for q, lab, poly in zip(m.loc[wl, "processed_path"], m.loc[wl, "task_label"], m.loc[wl, "leaf_poly"])]
        views = [v for vs in run_parallel(partial(crop_views, ccfg=p2["crops"]), cjobs, args.workers, desc="crops")
                 for v in vs]
        full = m.loc[use, ["processed_path"]].assign(view="full", box=[(0.0, 0.0, 1.0, 1.0)] * int(use.sum()),
                                                     leaf_fraction=np.nan, symptom_share=np.nan)
        crops = pd.concat([full, pd.DataFrame(views)], ignore_index=True)
        crops[["x0", "y0", "x1", "y1"]] = pd.DataFrame(crops["box"].tolist(), index=crops.index).round(4)
        crops = crops.drop(columns="box").merge(m[["processed_path", "split", "task_label", "source"]],
                                                on="processed_path")
        # Training views: close-ups whole; BRACOL leaf + zooms; RoCoLe full field photo + leaf + zooms
        # (the app receives full field photos, so those are trained on too).
        fit = crops["split"].isin(FIT_SPLITS)
        crops["use_for_training"] = fit & (
            ~crops["source"].isin(p2["whole_leaf_sources"]) | crops["view"].ne("full") | crops["source"].eq("rocole"))
        crops.to_csv(out / "crops.csv", index=False)
    m.to_csv(out / "manifest.csv", index=False)

    # ---- tables
    u = m[m["split"].ne("")].assign(label=lambda d: np.where(d["task_label"] != "", d["task_label"],
                                                              "ood:" + d["ood_type"]))
    cnt = u.pivot_table(index=["split", "source"], columns="label", values="path", aggfunc="count", fill_value=0)
    cnt = cnt.reindex([sp for sp in ORDER if sp in u["split"].unique()], level=0)
    indep = u[u["is_group_rep"]].pivot_table(index="split", columns="label", values="path", aggfunc="count",
                                             fill_value=0).reindex([sp for sp in ORDER if sp in u["split"].unique()])
    tr = m[m["split"].eq("train")]
    bal = tr.groupby(["task_label", "source"]).agg(images=("path", "size"), independent=("is_group_rep", "sum"),
                                                   bweight=("balanced_weight", "sum"))
    bal["share of all sampling (balanced_weight)"] = (bal["bweight"] / bal["bweight"].sum()).round(3)
    bal = bal.drop(columns="bweight")
    farm_tab = m[m["source"].eq("rocole") & m["split"].isin(["train", "val", "test_locked"])].pivot_table(
        index=["split", "farm_id"], columns="task_label", values="path", aggfunc="count", fill_value=0)
    excl = m[m["exclude_reason"].ne("")].groupby(["exclude_reason", "source"])["copies"].agg(["size", "sum"])
    excl.columns = ["distinct files", "files incl. copies"]
    view_tab = (crops[crops["use_for_training"]].groupby(["split", "source", "task_label", "view"]).size()
                .unstack(fill_value=0) if len(crops) else pd.DataFrame())
    no_zoom = (crops[crops["split"].isin(FIT_SPLITS) & crops["source"].isin(p2["whole_leaf_sources"])]
               .groupby(["source", "task_label", "processed_path"])["view"].apply(lambda v: "zoom" not in set(v))
               .groupby(level=[0, 1]).mean().round(2) if len(crops) else pd.Series(dtype=float))

    card = f"""# Phase 2 dataset: {p2['name']}

Built by `ml/scripts/08_build_phase2_dataset.py` from the data audit (`docs/DATA_AUDIT.md`).

**Task.** {', '.join(classes)}. `other` = phoma, miner or cercospora: a leaf disease that is not rust.
JMuBEN2 Healthy is excluded (2 independent images). Out-of-scope images (spider mite, BRACOL mixed stress) are
kept as `ood` and never trained on.

**Splits.**
- `train` / `val` / `test`: JMuBEN, JMuBEN2, Uganda and BRACOL, split {p2['split']['ratios']} by near-duplicate
  group, stratified by source x label; plus RoCoLe training farms in `train` and validation farms in `val`.
- `test_locked`: RoCoLe farms {sorted(f for f, s in farm_split.items() if s == 'test_locked')}, held out completely.
  Evaluate once, at the end; never use for choosing models, epochs, thresholds or calibration.
- `ood`: spider mite (RoCoLe, all farms) and BRACOL mixed-stress leaves; never trained on.

RoCoLe farm assignment (whole farms, chosen so each split keeps the overall healthy/rust mix):
{json.dumps(dict(sorted(farm_split.items())), indent=0)}

Checks passed: no near-duplicate group or RoCoLe plant straddles train/val/test; no RoCoLe farm is in two splits;
`test_locked` is RoCoLe only; no near-duplicate group is shared between the training splits and `test_locked`/`ood`;
every raw file is accounted for; 300 random processed images match their raw file.

**Preprocessing.** EXIF rotation -> RGB -> shorter side shrunk to <= {p2['store_shorter_side']} px (never enlarged)
-> JPEG q{p2['jpeg_quality']}. The model input is made at training/inference time. No intensity changes.

**Weights.** `sample_weight` = 1 / near-duplicate group size in the split. `balanced_weight` gives each class the
same total and each source the same share within a class (per split).

**Views (`crops.csv`).** Close-ups (JMuBEN, JMuBEN2, Uganda) are used whole. BRACOL: `leaf` + up to
{p2['crops']['zoom_per_image']} `zoom` crops. RoCoLe: the `full` field photo + `leaf` (from the annotated polygon)
+ zooms inside the polygon. Rust/other zooms must hold >= {p2['crops']['min_symptom_share']:.0%} of the leaf's
symptom pixels (rust: orange/yellow pustules; other: non-green lesion tissue).

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

## RoCoLe farms

```
{farm_tab.to_string()}
```

## Training views (train/val)

```
{view_tab.to_string() if len(view_tab) else 'n/a'}
```

Share of whole-leaf images with no zoom crop (leaf/full views only):
```
{no_zoom.to_string() if len(no_zoom) else 'n/a'}
```

## Excluded

```
{excl.to_string()}
```
"""
    (out / "DATASET.md").write_text(card, encoding="utf-8")

    if not args.manifest_only:
        from PIL import Image, ImageDraw

        from src.viz import captioned_grid

        (out / "checks").mkdir(exist_ok=True)
        rs = np.random.RandomState(0)
        colours = {"leaf": "blue", "zoom": "red"}
        for src, lab in [("rocole", "rust"), ("rocole", "healthy"), ("bracol", "other"), ("bracol", "rust")]:
            pick = crops[crops["source"].eq(src) & crops["split"].eq("train") & crops["task_label"].eq(lab)
                         & crops["view"].ne("full")]["processed_path"].drop_duplicates()
            tiles = []
            for q in pick.sample(min(8, len(pick)), random_state=rs):
                im = Image.open(out / q).convert("RGB")
                W, H = im.size
                d = ImageDraw.Draw(im)
                poly = m.loc[m["processed_path"].eq(q), "leaf_poly"].iloc[0]
                if isinstance(poly, str):
                    d.line([(x * W, y * H) for x, y in json.loads(poly)] + [(json.loads(poly)[0][0] * W,
                           json.loads(poly)[0][1] * H)], fill="yellow", width=4)
                for r in crops[crops["processed_path"].eq(q) & crops["view"].ne("full")].itertuples():
                    d.rectangle([r.x0 * W, r.y0 * H, r.x1 * W, r.y1 * H], outline=colours[r.view], width=5)
                im.thumbnail((360, 360))
                tiles.append(im)
            if tiles:
                cols_n = 4
                rows_n = (len(tiles) + cols_n - 1) // cols_n
                sheet = Image.new("RGB", (cols_n * 366, rows_n * 366), "white")
                for k, t in enumerate(tiles):
                    sheet.paste(t, ((k % cols_n) * 366, (k // cols_n) * 366))
                sheet.save(out / "checks" / f"views_{src}_{lab}.jpg", quality=85)
        for sp in ORDER:
            sub = u[u["split"].eq(sp)]
            if sub.empty:
                continue
            smp = sub.sample(frac=1, random_state=0).groupby(["label", "source"]).head(3)
            captioned_grid([(out / r.processed_path, f"{r.label}\n{r.source}/{r.folder}"[:40])
                            for r in smp.itertuples()], out / "checks" / f"samples_{sp}.jpg",
                           f"{sp}: random examples per label and source", cols=8, size=130)

    if not args.manifest_only and not args.no_zip:
        zpath = Path(cfg["_root"]) / p2["kaggle_zip"]
        files = [out / q for q in m.loc[m["processed_path"].ne(""), "processed_path"]]
        files += [out / "manifest.csv", out / "crops.csv", out / "DATASET.md", Path(cfg["_root"]) / CONFIG_REL]
        files += sorted((Path(cfg["_root"]) / "ml" / "src").glob("*.py"))
        with zipfile.ZipFile(zpath, "w") as zf:
            for f in files:
                zf.write(f, f.relative_to(Path(cfg["_root"])).as_posix(),
                         compress_type=zipfile.ZIP_STORED if f.suffix == ".jpg" else zipfile.ZIP_DEFLATED)
        print(f"Wrote {zpath} ({zpath.stat().st_size / 1e6:.0f} MB, {len(files)} files)")

    print("\n" + cnt.to_string() + "\n\nindependent:\n" + indep.to_string() + "\n\ntrain source balance:\n"
          + bal.to_string() + "\n\nRoCoLe farms:\n" + farm_tab.to_string() + "\n\ntraining views:\n"
          + (view_tab.to_string() if len(view_tab) else "n/a") + "\n\nno-zoom share:\n"
          + (no_zoom.to_string() if len(no_zoom) else "n/a") + "\n\nexcluded:\n" + excl.to_string())
    print(f"\nWrote {out}: manifest.csv, DATASET.md" + ("" if args.manifest_only else ", crops.csv, images/, checks/"))


if __name__ == "__main__":
    main()
