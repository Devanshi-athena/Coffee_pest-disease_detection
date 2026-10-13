"""Step 6: deep data audit. Read-only on data/raw; needs 01_inspect_raw.py outputs.

Questions answered (artifacts/audit/AUDIT.md):
  JMuBEN/JMuBEN2  how many truly independent images/crops; what augmentation produced the copies
  Uganda          which images/labels look unreliable; duplicates and augmentation artifacts
  BRACOL          how its stress codes map to our classes; multi-stress leaves; missing files
  RoCoLe          which images belong to our task, which are out of scope; leaf/plant structure
  All             resolution, framing, background, camera, coffee type, class balance, source shortcuts,
                  and a cheap estimate of cross-dataset performance (linear probe on frozen embeddings)

Embeddings: DINOv2 ViT-S/14 (general-purpose, not trained on any coffee data), cached in
artifacts/audit/embeddings.npz. Needs ml/.venv-eval.
"""
import argparse
import io
import json
import os
import re
import sys
import zipfile
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "ml"))
from src.common import die, load_config, read_image, resolve_path  # noqa: E402
from src.viz import captioned_grid  # noqa: E402

EMBED_MODEL = "vit_small_patch14_dinov2.lvd142m"
SIM_BANDS = [(0.98, 1.01), (0.95, 0.98), (0.92, 0.95), (0.88, 0.92)]
WHOLE_LEAF_SOURCES = {"bracol", "rocole"}  # whole leaves look alike to the embedder; need a stricter threshold
FRAMING = {  # from reviewing sheets/cluster_*.jpg and sample sheets; colour heuristics alone were unreliable
    "jmuben": "close-up crop (128px) of part of a leaf/lesion; many pale, blurry or upscaled",
    "jmuben2": "close-up crop (128px); Healthy = a few leaf crops rotated/recoloured thousands of times",
    "uganda": "close-up of part of a leaf in the field (256px), often blurry and upscaled; some branch shots",
    "bracol": "one whole detached leaf, upper side, on a white/grey studio background (2048x1024)",
    "rocole": "one whole leaf still on the plant, field background, leaf ~26% of the photo; top + underside",
}
ROCOLE_NAME = re.compile(r"^C(\d+)P(\d+)([HE])(\d+)$")
COFFEE_TYPE = {  # from the dataset publications; not derivable from the images
    "jmuben": "Arabica (Kenya), per JMuBEN paper / model card",
    "jmuben2": "Arabica (Kenya), per JMuBEN2 paper / model card",
    "uganda": "not stated in the download (Uganda grows mostly Robusta; check the paper)",
    "bracol": "Arabica (Brazil), per BRACOL paper",
    "rocole": "Robusta (Ecuador), per RoCoLe title/paper",
}


# ---------------------------------------------------------------- per-image features

def image_features(path: Path) -> dict:
    """Framing/background/resolution measurements on a <=512px working copy."""
    import cv2

    img = read_image(path, draft=512)[0]
    w, h = img.size
    scale = 256 / min(w, h)
    work = np.asarray(img.resize((max(1, round(w * scale)), max(1, round(h * scale)))))
    hsv = cv2.cvtColor(work, cv2.COLOR_RGB2HSV)
    H, S, V = hsv[..., 0].astype(int), hsv[..., 1].astype(int), hsv[..., 2].astype(int)
    plant = (H >= 5) & (H <= 95) & (S >= 50) & (V >= 30)      # green/yellow/brown tissue
    green = (H >= 30) & (H <= 90) & (S >= 50) & (V >= 30)
    plain_bg = (S < 35) & (V > 120)                           # white/grey paper or table
    ring = np.zeros(plant.shape, bool)
    m = max(2, int(0.08 * min(plant.shape)))
    ring[:m], ring[-m:], ring[:, :m], ring[:, -m:] = True, True, True, True
    gray = cv2.cvtColor(work, cv2.COLOR_RGB2GRAY).astype(np.float32)
    feats = {
        "plant_frac": float(plant.mean()), "green_frac": float(green.mean()),
        "border_plant_frac": float(plant[ring].mean()), "border_plain_frac": float(plain_bg[ring].mean()),
        "border_gray_std": float(gray[ring].std()),
        "mean_r": float(work[..., 0].mean()), "mean_g": float(work[..., 1].mean()), "mean_b": float(work[..., 2].mean()),
    }
    # Upscaling check at native resolution (small images only): an image that was enlarged by >=2x
    # barely changes when shrunk 2x and enlarged again.
    if max(w, h) <= 512:
        native = np.asarray(read_image(path, draft=None)[0].convert("L"), dtype=np.float32)
        nh, nw = native.shape
        down_up = cv2.resize(cv2.resize(native, (nw // 2, nh // 2), interpolation=cv2.INTER_AREA), (nw, nh),
                             interpolation=cv2.INTER_LINEAR)
        feats["detail_loss_2x"] = float(np.abs(native - down_up).mean())
    else:
        feats["detail_loss_2x"] = np.nan
    return feats


def exif_of(path: Path) -> dict:
    from PIL import Image

    try:
        e = Image.open(path).getexif()
    except Exception:
        return {"camera": "", "exif_date": ""}
    clean = lambda v: str(v).replace("\x00", "").strip() if v else ""  # noqa: E731
    cam = f"{clean(e.get(271))} {clean(e.get(272))}".strip()
    return {"camera": cam, "exif_date": clean(e.get(306))[:7].replace(":", "-")}


# ---------------------------------------------------------------- embeddings

def embed_all(paths: list[Path], cache: Path, workers: int) -> np.ndarray:
    keys = [p.as_posix() for p in paths]
    if cache.is_file():
        z = np.load(cache, allow_pickle=False)
        if list(z["keys"]) == keys:
            print(f"using cached embeddings {cache}")
            return z["emb"]
    import timm
    import torch
    from tqdm import tqdm

    model = timm.create_model(EMBED_MODEL, pretrained=True, num_classes=0, img_size=224).eval()
    cfg = timm.data.resolve_data_config({}, model=model)
    tf = timm.data.create_transform(**{**cfg, "input_size": (3, 224, 224), "crop_pct": 1.0})
    torch.set_num_threads(max(1, workers))

    def load(p):
        return tf(read_image(p, draft=448)[0])

    out = []
    with ThreadPoolExecutor(workers) as ex:
        for s in tqdm(range(0, len(paths), 32), desc="embedding"):
            batch = torch.stack(list(ex.map(load, paths[s:s + 32])))
            with torch.no_grad():
                out.append(torch.nn.functional.normalize(model(batch), dim=1).numpy())
    emb = np.concatenate(out).astype(np.float32)
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache, keys=np.array(keys), emb=emb)
    return emb


def similar_pairs(emb: np.ndarray, min_sim: float, block: int = 512):
    """All pairs i<j with cosine similarity >= min_sim."""
    ii, jj, ss = [], [], []
    for s in range(0, len(emb), block):
        sim = emb[s:s + block] @ emb[s:].T
        r, c = np.nonzero(sim >= min_sim)
        keep = c > r
        ii.append(r[keep] + s)
        jj.append(c[keep] + s)
        ss.append(sim[r[keep], c[keep]])
    return np.concatenate(ii), np.concatenate(jj), np.concatenate(ss)


def components(n: int, i: np.ndarray, j: np.ndarray) -> np.ndarray:
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components

    return connected_components(coo_matrix((np.ones(len(i)), (i, j)), shape=(n, n)), directed=False)[1]


# ---------------------------------------------------------------- linear probes

def probe(X_tr, y_tr, X_te, y_te, classes) -> dict:
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, f1_score

    clf = LogisticRegression(max_iter=3000, class_weight="balanced", C=1.0)
    clf.fit(X_tr, y_tr)
    pred = clf.predict(X_te)
    common = sorted(set(y_te) & set(y_tr))
    keep = np.isin(y_te, common)
    return {"n_test": int(keep.sum()), "classes": ",".join(classes[c] for c in common),
            "accuracy": float(accuracy_score(y_te[keep], pred[keep])),
            "macro_f1": float(f1_score(y_te[keep], pred[keep], labels=common, average="macro", zero_division=0))}


def grouped_cv(X, y, groups, n_splits=5, seed=42):
    """Out-of-fold probabilities with StratifiedGroupKFold (groups never straddle folds)."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedGroupKFold

    proba = np.zeros((len(y), int(y.max()) + 1))
    for tr, te in StratifiedGroupKFold(n_splits, shuffle=True, random_state=seed).split(X, y, groups):
        clf = LogisticRegression(max_iter=3000, class_weight="balanced").fit(X[tr], y[tr])
        proba[np.ix_(te, clf.classes_)] = clf.predict_proba(X[te])
    return proba


# ---------------------------------------------------------------- main

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", help="path to config.yaml (default: configs/config.yaml)")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--sim-closeup", type=float, default=0.95,
                    help="embedding similarity at/above which two close-up crops count as the same image")
    ap.add_argument("--sim-wholeleaf", type=float, default=0.98,
                    help="the same for pairs involving whole-leaf sources (BRACOL, RoCoLe)")
    args = ap.parse_args()

    from sklearn.metrics import accuracy_score, f1_score

    cfg = load_config(args.config)
    classes = cfg["classes"]
    cidx = {c: i for i, c in enumerate(classes)}
    raw = resolve_path(cfg, "raw_dir")
    art = resolve_path(cfg, "artifacts_dir")
    out = art / "audit"
    (out / "sheets").mkdir(parents=True, exist_ok=True)
    inspect = art / "inspect"
    if not (inspect / "images.csv").is_file():
        die("artifacts/inspect/images.csv not found. Run ml/scripts/01_inspect_raw.py first.")

    report = ["# Data audit\n\n",
              f"Embeddings: `{EMBED_MODEL}`. Same image = byte-identical, or rotation/flip-aware hash distance <= 4 "
              f"(close-ups only), or embedding similarity >= {args.sim_closeup} (close-ups) / "
              f">= {args.sim_wholeleaf} (any pair involving BRACOL/RoCoLe). Thresholds chosen from "
              "sheets/emb_pairs_*.jpg: at 0.95-0.98 close-up pairs are the same crop but whole-leaf pairs are "
              "different leaves.\n"]

    def section(title, body, note=""):
        text = body if isinstance(body, str) else body.to_string()
        print(f"\n== {title}\n{text}")
        report.append(f"\n## {title}\n\n{note}\n```\n{text}\n```\n")

    # ---- load distinct files
    im = pd.read_csv(inspect / "images.csv", low_memory=False)
    status = im.groupby(["source", "folder"])["status"].value_counts().unstack(fill_value=0)
    ok = im[im["status"] == "ok"].copy()
    ok["copies"] = ok.groupby("md5")["md5"].transform("size")
    md5_labels = ok.groupby("md5")["label"].nunique()
    ok["label_conflict"] = ok["md5"].map(md5_labels) > 1
    df = ok.drop_duplicates("md5").reset_index(drop=True)
    df["in_scope"] = ~df["label"].str.startswith("SKIP") & ~df["label_conflict"]
    print(f"{len(df)} distinct files ({len(ok)} readable, {len(im)} total)")

    # ---- per-image features, EXIF, embeddings
    paths = [raw / p for p in df["path"]]
    fcache = out / "features.csv"
    cached = pd.read_csv(fcache) if fcache.is_file() else None
    if cached is not None and cached["path"].tolist() == df["path"].tolist():
        print(f"using cached features {fcache}")
        fx = cached.drop(columns=["path"])
    else:
        with ThreadPoolExecutor(args.workers) as ex:
            fx = pd.concat([pd.DataFrame(list(ex.map(image_features, paths))),
                            pd.DataFrame(list(ex.map(exif_of, paths)))], axis=1)
        fx.assign(path=df["path"]).to_csv(fcache, index=False)
    fx[["camera", "exif_date"]] = fx[["camera", "exif_date"]].fillna("").astype(str)
    df = pd.concat([df, fx], axis=1)
    emb = embed_all(paths, out / "embeddings.npz", args.workers)

    # ---- embedding near-duplicate groups (on top of byte-identical merging)
    pi, pj, ps = similar_pairs(emb, min(b[0] for b in SIM_BANDS))
    pairs = pd.DataFrame({"a": pi, "b": pj, "sim": ps})
    for k in ("source", "folder", "label", "path"):
        pairs[f"{k}_a"], pairs[f"{k}_b"] = df[k].to_numpy()[pi], df[k].to_numpy()[pj]
    whole = pairs["source_a"].isin(WHOLE_LEAF_SOURCES) | pairs["source_b"].isin(WHOLE_LEAF_SOURCES)
    pairs["same_image"] = pairs["sim"] >= np.where(whole, args.sim_wholeleaf, args.sim_closeup)
    strong = pairs[pairs["same_image"]]
    # rotated/flipped copies found by the step-1 hash search, close-up sources only
    pos = pd.Series(np.arange(len(df)), index=df["path"])
    nd0 = pd.read_csv(inspect / "near_dup_pairs.csv", usecols=["path_a", "path_b", "d", "d_any"])
    closeup = (~nd0["path_a"].str.split("/").str[0].isin(WHOLE_LEAF_SOURCES)
               & ~nd0["path_b"].str.split("/").str[0].isin(WHOLE_LEAF_SOURCES)
               & (nd0[["d", "d_any"]].min(axis=1) <= 4))
    ha = pos.reindex(nd0.loc[closeup, "path_a"]).to_numpy()
    hb = pos.reindex(nd0.loc[closeup, "path_b"]).to_numpy()
    okh = ~(np.isnan(ha) | np.isnan(hb))
    df["group"] = components(len(df), np.concatenate([strong["a"].to_numpy(), ha[okh].astype(int)]),
                             np.concatenate([strong["b"].to_numpy(), hb[okh].astype(int)]))
    # RoCoLe: the two photos of one leaf (top + underside) are one physical sample
    rc = df["source"].eq("rocole")
    leaf = df.loc[rc, "path"].map(lambda p: (lambda m: f"C{m[1]}P{m[2]}L{m[4]}" if m else p)(
        ROCOLE_NAME.match(Path(p).stem)))
    df["leaf_id"] = ""
    df.loc[rc, "leaf_id"] = leaf
    df.to_csv(out / "audit_images.csv", index=False)

    rs = np.random.RandomState(42)
    for lo, hi in SIM_BANDS:
        band = pairs[(pairs["sim"] >= lo) & (pairs["sim"] < hi)]
        smp = band.sample(min(12, len(band)), random_state=rs)
        items = []
        for r in smp.itertuples():
            items += [(raw / r.path_a, f"{r.source_a}/{r.label_a}\nsim {r.sim:.3f}"),
                      (raw / r.path_b, f"{r.source_b}/{r.label_b}")]
        captioned_grid(items, out / "sheets" / f"emb_pairs_{lo:.2f}-{min(hi, 1):.2f}.jpg",
                       f"embedding similarity {lo:.2f}-{min(hi, 1):.2f}: {len(band)} pairs (pairs side by side)",
                       cols=4, size=160)

    # =============================================================== JMuBEN / JMuBEN2
    def ladder(sub_mask):
        sub_ok = ok[ok["source"].isin(df.loc[sub_mask, "source"].unique())]
        rows = []
        for (s, f), g in df[sub_mask].groupby(["source", "folder"]):
            rows.append({"source": s, "folder": f, "label": g["label"].iloc[0],
                         "files": int((sub_ok["source"].eq(s) & sub_ok["folder"].eq(f)).sum()),
                         "byte-unique": len(g),
                         "independent images": g["group"].nunique()})
        return pd.DataFrame(rows).set_index(["source", "folder"])

    jm = df["source"].isin(["jmuben", "jmuben2"])
    jl = ladder(jm)
    section("JMuBEN/JMuBEN2: files -> byte-unique -> independent images", jl,
            "Perceptual-hash grouping from step 1 sits between these (see artifacts/inspect/summary.md).")
    # How were copies produced? Classify near-duplicate pairs within JMuBEN.
    nd = pd.read_csv(inspect / "near_dup_pairs.csv")
    njm = nd[nd["path_a"].str.match(r"jmuben2?/") & nd["path_b"].str.match(r"jmuben2?/")]
    col = df.set_index("path")[["mean_r", "mean_g", "mean_b"]]
    ca, cb = col.reindex(njm["path_a"]).to_numpy(), col.reindex(njm["path_b"]).to_numpy()
    color_shift = np.abs(ca - cb).max(axis=1)
    kind = np.where(njm["d_any"] < njm["d"], "rotated/flipped copy",
                    np.where(color_shift > 12, "colour-shifted copy", "near-identical (crop/compression)"))
    copies = ok[ok["source"].isin(["jmuben", "jmuben2"])]
    aug = pd.Series({
        "byte-identical extra copies": int((copies.groupby("md5").size() - 1).sum()),
        **pd.Series(kind).value_counts().rename(lambda k: f"near-dup pairs: {k}").to_dict(),
        "max copies of one file": int(copies["copies"].max()),
    })
    section("JMuBEN/JMuBEN2: how the copies were made", aug.to_string(),
            "Pairs from step 1 (hash distance <= 6). Colour shift = largest per-channel mean difference > 12/255.")
    big = df[jm].groupby("group").agg(n=("path", "size"), label=("label", "first"),
                                      path=("path", "first")).sort_values("n", ascending=False)
    captioned_grid([(raw / r.path, f"{r.label}: {r.n} files") for r in big.head(12).itertuples()],
                   out / "sheets" / "jmuben_largest_groups.jpg", "JMuBEN: largest embedding groups", cols=4)

    # =============================================================== Uganda
    ug = df["source"].eq("uganda")
    ul = ladder(ug)
    ul["empty/corrupt files"] = [int(status.loc[i, "corrupt"]) if i in status.index and "corrupt" in status
                                 else 0 for i in ul.index]
    section("Uganda: files -> byte-unique -> independent images", ul)
    u_conf = ok[ok["source"].eq("uganda") & ok["label_conflict"]]
    to_jm = pairs[(pairs["source_a"].isin(["jmuben", "jmuben2"]) & pairs["source_b"].eq("uganda")) |
                  (pairs["source_b"].isin(["jmuben", "jmuben2"]) & pairs["source_a"].eq("uganda"))]
    to_jm_strong = to_jm[to_jm["same_image"]]
    ug_paths_jm = set(to_jm_strong["path_a"]) | set(to_jm_strong["path_b"])
    ug_jm = df[ug & df["path"].isin(ug_paths_jm)]
    section("Uganda: duplicates, cross-label copies, overlap with JMuBEN", pd.Series({
        "files that are byte-identical copies within Uganda": int((ok[ok["source"].eq("uganda")]
                                                                  .groupby("md5").size() - 1).sum()),
        "files whose identical copy carries another label": len(u_conf),
        f"Uganda images ~identical to a JMuBEN image (sim>={args.sim_closeup})": len(ug_jm),
        "  ...by Uganda label": dict(Counter(ug_jm["label"])),
        "Uganda images upscaled >=2x (detail_loss_2x < 1.5)": int((df.loc[ug, "detail_loss_2x"] < 1.5).sum()),
    }).to_string())
    captioned_grid([(raw / p, f"{lab}\n(also in another class)") for p, lab in
                    u_conf[["path", "label"]].head(16).itertuples(index=False)],
                   out / "sheets" / "uganda_cross_label_copies.jpg", "Uganda: identical files under two labels")

    # =============================================================== label reliability (all labelled sources)
    rel_rows, flagged_all = [], []
    for s in ["jmuben", "jmuben2", "uganda", "bracol", "rocole"]:
        m = (df["source"] == s) & df["in_scope"]
        sub = df[m]
        labs = sorted(sub["label"].unique())
        if len(labs) < 2 and s not in ("jmuben", "jmuben2"):
            continue
        if s in ("jmuben", "jmuben2"):  # each has <=3 classes; audit them together
            if s == "jmuben2":
                continue
            m = df["source"].isin(["jmuben", "jmuben2"]) & df["in_scope"]
            sub, s = df[m], "jmuben+jmuben2"
        X = emb[m.to_numpy()]
        y = sub["label"].map(cidx).to_numpy()
        groups = np.where(sub["leaf_id"] != "", sub["leaf_id"], sub["group"].astype(str))
        proba = grouped_cv(X, y, groups)
        given = proba[np.arange(len(y)), y]
        other = np.where(np.arange(proba.shape[1])[None, :] == y[:, None], -1, proba)
        flag = (given < 0.2) & (other.max(1) >= 0.6)
        pred = proba.argmax(1)
        rel_rows.append({"source": s, "n": len(y), "probe accuracy (grouped 5-fold)": accuracy_score(y, pred),
                         "probe macro F1": f1_score(y, pred, average="macro"),
                         "flagged as likely mislabelled": int(flag.sum()),
                         "flagged %": flag.mean()})
        fl = sub[flag].assign(suggested=[classes[i] for i in other[flag].argmax(1)],
                              p_suggested=other[flag].max(1), p_given=given[flag], audit_source=s)
        flagged_all.append(fl)
        for lab, g in fl.groupby("label"):
            g = g.sort_values("p_suggested", ascending=False).head(16)
            captioned_grid([(raw / r.path, f"labelled: {r.label}\nlooks like: {r.suggested} ({r.p_suggested:.2f})")
                            for r in g.itertuples()],
                           out / "sheets" / f"review_{s}_{lab}.jpg",
                           f"{s}: '{lab}' images that look like another class ({len(fl[fl.label == lab])} flagged)")
    rel = pd.DataFrame(rel_rows).set_index("source").round(3)
    section("Label reliability (cross-validated linear probe on embeddings, within each source)", rel,
            "Flagged = given label gets < 0.2 probability while another class gets >= 0.6, from a classifier that "
            "never saw that image (or its duplicates/other leaf side). Flags are candidates for human review, not "
            "verdicts. Review sheets: sheets/review_*.jpg; full list: label_review.csv.")
    if flagged_all:
        pd.concat(flagged_all)[["audit_source", "path", "label", "suggested", "p_given", "p_suggested"]].to_csv(
            out / "label_review.csv", index=False)

    # =============================================================== BRACOL
    bcsv = ROOT / "data" / "bracol_dataset.csv"
    if bcsv.is_file():
        b = pd.read_csv(bcsv)
        code_map = b.groupby("predominant_stress")[["miner", "rust", "phoma", "cercospora"]].mean().round(2)
        code_map["n in csv"] = b["predominant_stress"].value_counts().sort_index()
        names = {0: "healthy", 1: "miner", 2: "rust", 3: "phoma", 4: "cercospora", 5: "unmapped_stress_code_5"}
        present = df[df["source"].eq("bracol")]["folder"].value_counts()
        code_map["our class"] = [names.get(c, "?") for c in code_map.index]
        code_map["images present"] = [int(present.get(names.get(c, ""), 0)) for c in code_map.index]
        b["n_stress"] = b[["miner", "rust", "phoma", "cercospora"]].sum(axis=1)
        multi = b[b["predominant_stress"].between(1, 4)].groupby("predominant_stress")["n_stress"].apply(
            lambda s: f"{(s > 1).mean():.0%}")
        code_map["leaves with >1 stress"] = multi
        sev = pd.crosstab(b["predominant_stress"], b["severity"])
        section("BRACOL: stress code -> our class", code_map.to_string() + "\n\nseverity (0 none .. 4 high) by code:\n"
                + sev.to_string(),
                "Each code's own symptom column is 1.00, which confirms the code order. Code 5 = several stresses "
                "with none predominant (not one of our classes). 'images present' < 'n in csv' because the "
                "published archive is truncated.")

    # =============================================================== RoCoLe
    rm = df["source"].eq("rocole")
    r = df[rm].copy()
    parts = r["path"].map(lambda p: ROCOLE_NAME.match(Path(p).stem))
    r["plant"] = [f"C{m[1]}P{m[2]}" if m else "?" for m in parts]
    r["side"] = [("top" if m[3] == "H" else "underside") if m else "?" for m in parts]
    r["task_label"] = np.where(r["label"].str.startswith("SKIP"), "out of scope (spider mite)", r["label"])
    pv = r.pivot_table(index="leaf_id", columns="side", values="task_label", aggfunc="first").dropna()
    agree = (pv["top"] == pv["underside"]).mean() if len(pv) else np.nan
    disagree = pd.crosstab(pv["top"], pv["underside"]) if len(pv) else pd.DataFrame()
    zf = next(iter((ROOT / "Data_dump").glob("RoCoLe*.zip")), None)
    leaf_frac = None
    if zf:
        z = zipfile.ZipFile(zf)
        ann = pd.read_csv(io.BytesIO(z.read(next(n for n in z.namelist() if n.endswith("RoCoLE-csv.csv")))))
        sizes = df[rm].assign(f=df.loc[rm, "path"].map(lambda p: Path(p).name)).set_index("f")[["width", "height"]]

        def poly_frac(row):
            try:
                pts = json.loads(row["Label"])["Leaf"][0]["geometry"]
                x, y = np.array([p["x"] for p in pts], float), np.array([p["y"] for p in pts], float)
                wh = sizes.loc[row["External ID"]]
                return 0.5 * abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1))) / (wh["width"] * wh["height"])
            except Exception:
                return np.nan
        leaf_frac = ann.apply(poly_frac, axis=1).describe(percentiles=[0.1, 0.5, 0.9]).round(3)
    section("RoCoLe: structure, task membership, label consistency", pd.Series({
        "farms (C)": r["plant"].str.extract(r"C(\d+)")[0].nunique(), "plants": r["plant"].nunique(),
        "leaves (2 per plant, each shot top + underside)": r["leaf_id"].nunique(),
        "photos by task label": dict(Counter(r["task_label"])),
        "rust photos by level": dict(Counter(r.loc[r["label"].eq("rust"), "folder"])),
        "leaves whose two sides get the same task label": f"{agree:.1%}",
        "leaf area / photo area (from annotation polygons)": dict(leaf_frac) if leaf_frac is not None else "n/a",
        "camera models": dict(Counter(r["camera"]).most_common(6)),
    }).to_string() + "\n\ntop side label (rows) vs underside label (columns):\n" + disagree.to_string(),
            "In scope: healthy and rust (levels 1-4 merged). Out of scope: red spider mite (a pest). "
            "Splits must keep both photos of a leaf (and ideally a whole plant) together.")
    # rust severity separability from healthy, within RoCoLe
    sev_rows = []
    for lvl in ["rust_level_1", "rust_level_2", "rust_level_3", "rust_level_4"]:
        m2 = rm & df["folder"].isin(["healthy", lvl])
        if df[m2]["folder"].nunique() < 2:
            continue
        y = (df.loc[m2, "folder"] != "healthy").astype(int).to_numpy()
        proba = grouped_cv(emb[m2.to_numpy()], y, df.loc[m2, "leaf_id"].to_numpy())
        sev_rows.append({"healthy vs": lvl, "n": int(m2.sum()),
                         "probe accuracy": accuracy_score(y, proba.argmax(1)),
                         f"recall of {lvl}": float((proba.argmax(1)[y == 1] == 1).mean())})
    if sev_rows:
        section("RoCoLe: how separable is each rust level from healthy?", pd.DataFrame(sev_rows)
                .set_index("healthy vs").round(3), "Grouped by leaf. Low numbers mean the label is visually subtle.")

    # =============================================================== all sources
    by_src = df.groupby("source")
    res = by_src.apply(lambda g: pd.Series({
        "distinct files": len(g),
        "median size": f"{int(g['width'].median())}x{int(g['height'].median())}",
        "min side (p5/p50/p95)": "/".join(str(int(v)) for v in np.percentile(np.minimum(g["width"], g["height"]),
                                                                            [5, 50, 95])),
        "aspect w/h (median)": round(float((g["width"] / g["height"]).median()), 2),
        "upscaled >=2x": f"{(g['detail_loss_2x'] < 1.5).mean():.0%}" if g["detail_loss_2x"].notna().any() else "n/a",
        "median KB": round(float(g["kb"].median()), 1),
    }), include_groups=False)
    section("All: resolution", res, "upscaled >=2x: small images that lose almost no detail when shrunk 2x "
            "(detail_loss_2x < 1.5 grey levels) were most likely enlarged from a smaller original.")
    bg = by_src.apply(lambda g: pd.Series({
        "framing (visual review)": FRAMING.get(g.name, "?"),
        "plain background at edges": f"{(g['border_plain_frac'] >= 0.6).mean():.0%}",
        "median brightness": round(float(g["brightness"].median()), 1),
        "median sharpness": round(float(g["sharpness"].median()), 1),
    }), include_groups=False)
    section("All: framing and background", bg.to_string(),
            "Framing comes from reviewing the sample and cluster sheets. plain background = >=60% of an 8% edge "
            "frame is low-saturation bright pixels (white/grey paper). Sharpness = Laplacian variance at 256px.")
    cams = df.groupby("source").apply(lambda g: pd.Series({
        "with EXIF camera": f"{(g['camera'] != '').mean():.0%}",
        "camera models": dict(Counter(g.loc[g["camera"] != "", "camera"]).most_common(4)),
        "capture months": dict(Counter(g.loc[g["exif_date"] != "", "exif_date"]).most_common(3)),
        "coffee type": COFFEE_TYPE.get(g.name, "?"),
    }), include_groups=False)
    section("All: camera and coffee type", cams.to_string())

    inscope = df[df["in_scope"]]
    bal = inscope.pivot_table(index="label", columns="source", values="group", aggfunc="nunique", fill_value=0)
    bal = bal.reindex([c for c in classes if c in bal.index])
    bal_files = ok[~ok["label"].str.startswith("SKIP")].pivot_table(index="label", columns="source", values="path",
                                                                     aggfunc="count", fill_value=0)
    section("All: class balance (independent images = embedding groups; RoCoLe photos, 2 per leaf)",
            bal.to_string() + "\n\nfor comparison, raw readable files:\n" +
            bal_files.reindex(bal.index).to_string())

    # ---- domain structure: clusters of all images
    from sklearn.cluster import KMeans

    km = KMeans(n_clusters=10, n_init=4, random_state=42).fit(emb)
    df["cluster"] = km.labels_
    comp = pd.crosstab(df["cluster"], df["source"])
    comp["dominant labels"] = df.groupby("cluster")["label"].agg(lambda s: dict(Counter(s).most_common(3)))
    for c, g in df.groupby("cluster"):
        smp = g.sample(min(12, len(g)), random_state=rs)
        captioned_grid([(raw / p, f"{s}/{lab}") for p, s, lab in smp[["path", "source", "label"]].itertuples(index=False)],
                       out / "sheets" / f"cluster_{c:02d}.jpg", f"cluster {c}: {len(g)} images", cols=4, size=150)
    section("All: what kinds of images exist (k-means on embeddings, 10 clusters)", comp,
            "Clusters that contain a single source are visual domains no other dataset covers. See sheets/cluster_*.jpg.")

    # ---- source shortcut risk
    rows = []
    for lab in classes:
        m3 = df["in_scope"] & df["label"].eq(lab)
        srcs = df.loc[m3, "source"].value_counts()
        srcs = srcs[srcs >= 20]
        if len(srcs) < 2:
            rows.append({"class": lab, "sources (>=20 imgs)": ",".join(srcs.index), "source predictable from image": "n/a"})
            continue
        m3 = m3 & df["source"].isin(srcs.index)
        ys = df.loc[m3, "source"].astype("category").cat.codes.to_numpy()
        proba = grouped_cv(emb[m3.to_numpy()], ys, df.loc[m3, "group"].to_numpy())
        rows.append({"class": lab, "sources (>=20 imgs)": ",".join(srcs.index),
                     "source predictable from image": f"{accuracy_score(ys, proba.argmax(1)):.1%}",
                     "chance": f"{srcs.max() / srcs.sum():.1%}"})
    pool = df["in_scope"] & df["source"].map(lambda s: cfg["sources"].get(s, {}).get("role") == "train")
    conf_tab = pd.crosstab(df.loc[pool, "label"], df.loc[pool, "source"], normalize="index").round(2)
    section("All: source shortcut risk", pd.DataFrame(rows).set_index("class").to_string() +
            "\n\nplanned train pool: share of each class coming from each source:\n" + conf_tab.to_string(),
            "If the source can be read off an image almost perfectly, and some classes come from only one source, "
            "a model can learn 'which dataset' instead of 'which disease'.")

    # ---- expected performance: linear probe across datasets
    X_all, y_all = emb[df["in_scope"].to_numpy()], df.loc[df["in_scope"], "label"].map(cidx).to_numpy()
    src_all = df.loc[df["in_scope"], "source"].replace({"jmuben2": "jmuben"}).to_numpy()
    grp_all = np.where(df.loc[df["in_scope"], "leaf_id"] != "", df.loc[df["in_scope"], "leaf_id"],
                       df.loc[df["in_scope"], "group"].astype(str))
    # one sample per independent group for training, so 520 copies don't count 520 times
    first = ~pd.Series(grp_all).duplicated().to_numpy()
    train_sets = {"jmuben (JMuBEN+2)": ["jmuben"], "uganda": ["uganda"], "jmuben + uganda (planned pool)":
                  ["jmuben", "uganda"], "bracol": ["bracol"], "rocole": ["rocole"]}
    rows = []
    for tname, tsrc in train_sets.items():
        tr = np.isin(src_all, tsrc) & first
        for test in ["jmuben", "uganda", "bracol", "rocole"]:
            te = src_all == test
            if test in tsrc:
                m4 = np.isin(src_all, [test])
                proba = grouped_cv(X_all[m4], y_all[m4], grp_all[m4])
                common = sorted(set(y_all[m4]))
                yy, pp = y_all[m4], proba.argmax(1)
                rows.append({"train on": tname, "test on": test, "how": "grouped 5-fold CV (in-domain)",
                             "accuracy": accuracy_score(yy, pp),
                             "macro_f1": f1_score(yy, pp, labels=common, average="macro"),
                             "classes": ",".join(classes[c] for c in common)})
            else:
                rr = probe(X_all[tr], y_all[tr], X_all[te], y_all[te], classes)
                rows.append({"train on": tname, "test on": test, "how": "train/test on different datasets",
                             "accuracy": rr["accuracy"], "macro_f1": rr["macro_f1"], "classes": rr["classes"]})
    for test in ["uganda", "bracol", "rocole"]:
        tr, te = (src_all != test) & first, src_all == test
        rr = probe(X_all[tr], y_all[tr], X_all[te], y_all[te], classes)
        rows.append({"train on": f"all except {test}", "test on": test, "how": "leave-one-dataset-out",
                     "accuracy": rr["accuracy"], "macro_f1": rr["macro_f1"], "classes": rr["classes"]})
    perf = pd.DataFrame(rows).set_index(["train on", "test on"]).round(3)
    perf.to_csv(out / "expected_performance.csv")
    section("Expected performance (linear probe on frozen DINOv2 embeddings)", perf,
            "A cheap proxy for training: a logistic regression on general-purpose image embeddings. Fine-tuning "
            "usually beats it in-domain, but the gap between in-domain and cross-dataset rows is the part that "
            "transfers. Macro F1 is over the classes both sides share.")

    (out / "AUDIT.md").write_text("".join(report), encoding="utf-8")
    print(f"\nWrote {out}: AUDIT.md, audit_images.csv, label_review.csv, expected_performance.csv, "
          "embeddings.npz, sheets/")


if __name__ == "__main__":
    main()
