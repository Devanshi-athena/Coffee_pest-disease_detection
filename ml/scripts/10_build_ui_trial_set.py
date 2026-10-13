"""Step 10: a small set of photos for trying the app by hand, covering every pass and fail scenario.

Picks photos the model never trained on (test, test_locked, out-of-scope) from artifacts/phase2_eval/predictions.csv,
saves them the way the browser uploads them (EXIF-rotated, longest side <= 1280, JPEG under 2 MB, so the frontend
sends them unchanged), re-runs the installed backend model on the SAVED files and keeps a photo only if the app's
answer still matches its scenario. Writes trial_images/: <ID>.jpg, trial_cases.csv, README.md, contact sheets.

Usage: ml/.venv-eval/Scripts/python ml/scripts/10_build_ui_trial_set.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageOps

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))
sys.path.insert(0, str(ROOT / "ml"))
from app.inference import Predictor, load_image  # noqa: E402

OUT = ROOT / "trial_images"
MAX_SIDE = 1280  # frontend/src/image.ts
SEED = 7
STYLE = {
    "jmuben": "close-up tile of part of a leaf (lab, Kenya)",
    "jmuben2": "close-up tile of part of a leaf (lab, Kenya)",
    "uganda": "single leaf photo (Uganda)",
    "bracol": "whole leaf on a white sheet (Brazil)",
    "rocole": "whole leaf on the plant, phone photo in the field (Ecuador)",
}
STRESS = {0: "healthy", 1: "miner", 2: "rust", 3: "phoma", 4: "cercospora"}

# (scenario, how many to keep). Order = ID order.
PLAN = [
    ("PASS: healthy recognised", 6),
    ("PASS: rust recognised", 8),
    ("PASS: other disease recognised as not rust", 6),
    ("PASS: unknown problem (mite) answered Not sure", 3),
    ("FAIL: other disease called rust", 3),
    ("FAIL: other disease called healthy", 3),
    ("FAIL: rust called another problem", 3),
    ("FAIL: rust missed (called healthy)", 2),
    ("FAIL: healthy called rust", 1),
    ("FAIL: unknown problem (mite) called rust", 3),
    ("FAIL: unknown problem (mite) called healthy", 2),
    ("SAFE FAIL: known condition but Not sure (low confidence)", 4),
    ("SAFE FAIL: known condition but Not sure (unfamiliar photo)", 3),
    ("MIXED: BRACOL leaf with several stresses (code 5)", 4),
]


def scenario(r) -> str | None:
    truth, out = r.truth, r.out
    if r.folder == "unmapped_stress_code_5":
        return "MIXED: BRACOL leaf with several stresses (code 5)"
    if truth == "out_of_scope":
        return {"not_sure": "PASS: unknown problem (mite) answered Not sure",
                "rust": "FAIL: unknown problem (mite) called rust",
                "healthy": "FAIL: unknown problem (mite) called healthy"}.get(out)
    if out == truth:
        return {"healthy": "PASS: healthy recognised", "rust": "PASS: rust recognised",
                "other": "PASS: other disease recognised as not rust"}[truth]
    if out == "not_sure":
        return ("SAFE FAIL: known condition but Not sure (unfamiliar photo)" if r.reason == "unfamiliar_image"
                else "SAFE FAIL: known condition but Not sure (low confidence)")
    return {("other", "rust"): "FAIL: other disease called rust", ("other", "healthy"): "FAIL: other disease called healthy",
            ("rust", "other"): "FAIL: rust called another problem", ("rust", "healthy"): "FAIL: rust missed (called healthy)",
            ("healthy", "rust"): "FAIL: healthy called rust"}.get((truth, out))


def bracol_symptoms(path: str, table: pd.DataFrame) -> str:
    if not path.startswith("bracol/"):
        return ""
    row = table.loc[int(Path(path).stem)] if int(Path(path).stem) in table.index else None
    if row is None:
        return ""
    present = [d for d in ("rust", "phoma", "miner", "cercospora") if row[d] > 0]
    return f"annotated: {', '.join(present) or 'none'}; predominant={STRESS.get(int(row.predominant_stress), 'code ' + str(int(row.predominant_stress)))}; severity={int(row.severity)}"


def save_like_browser(src: Path, dst: Path) -> tuple[int, int]:
    img = ImageOps.exif_transpose(Image.open(src)).convert("RGB")
    s = min(1.0, MAX_SIDE / max(img.size))
    if s < 1:
        img = img.resize((round(img.width * s), round(img.height * s)), Image.BILINEAR)
    for q in (92, 85, 75):
        img.save(dst, "JPEG", quality=q)
        if dst.stat().st_size < 2_000_000:
            break
    return img.size


def explain(r, d) -> tuple[str, str]:
    """(result, why) in plain words, from the measured facts of this photo."""
    conf, dist = r.app_confidence, r.app_distance
    t, od = d["confidence_threshold"], d["ood_threshold"]
    second = f"{r.app_second} {r.app_second_p:.0%}"
    fam = "familiar" if dist <= od * 0.6 else "borderline-familiar" if dist <= od else "unfamiliar"
    facts = f"confidence {conf:.0%} (needs {t:.0%}); distance {dist:.3f} (limit {od:.3f}, {fam}); runner-up {second}"
    sc = r.scenario
    if sc.startswith("PASS: unknown"):
        why = ("Mite damage was never trained on. " +
               ("The photo is far from every training photo, so the distance check refused it."
                if r.app_reason == "unfamiliar_image" else
                "The model was split between answers, so it fell below the confidence bar."))
    elif sc.startswith("PASS"):
        why = f"Clear, typical {r.truth} photo in a style seen in training; high confidence and close to training photos."
    elif "mite" in sc:
        why = (f"Mite damage was never trained on and the model has no 'unknown' class, so it picks the closest known "
               f"answer ({r.app_outcome}). The photo is a familiar field photo, so the distance check does not catch it: "
               "the background and leaf look like RoCoLe training photos even though the damage is new.")
    elif sc.startswith("MIXED"):
        why = ("BRACOL code-5 leaves carry more than one stress. " +
               ("It has rust annotated, so a rust answer is defensible." if "rust" in r.bracol_symptoms.split(";")[0]
                else "No rust is annotated, so a rust answer would be wrong.") +
               " One answer per photo cannot describe a leaf with several problems.")
    elif sc.startswith("SAFE FAIL") and "unfamiliar" in sc:
        why = ("The photo is farther from the training photos than the distance limit (unusual framing, lighting, "
               "background or symptom look), so the app refuses instead of guessing. Correct label was "
               f"{r.truth}; the model's top guess was {r.app_label}.")
    elif sc.startswith("SAFE FAIL"):
        why = (f"The model hesitates between {r.app_label} and {r.app_second}: the symptoms are weak, early or "
               "ambiguous, so confidence stays below the bar and the app says Not sure rather than risk a wrong answer.")
    else:
        why = (f"Confident wrong answer: it said {r.app_outcome} for a {r.truth} leaf. "
               "See visual_note for what in the photo misleads it.")
    return facts, why


def contact_sheet(rows: pd.DataFrame, path: Path, cols: int = 6, side: int = 220):
    n = len(rows)
    sheet = Image.new("RGB", (cols * side, ((n + cols - 1) // cols) * (side + 34)), "white")
    for i, r in enumerate(rows.itertuples()):
        im = Image.open(OUT / r.file).convert("RGB")
        im.thumbnail((side, side))
        x, y = (i % cols) * side, (i // cols) * (side + 34)
        sheet.paste(im, (x + (side - im.width) // 2, y))
        ImageDraw.Draw(sheet).text((x + 4, y + side + 2), f"{r.id} truth={r.truth}\napp={r.app_outcome}", fill="black")
    sheet.save(path, quality=85)


def main():
    pred = pd.read_csv(ROOT / "artifacts/phase2_eval/predictions.csv")
    decision = json.loads((ROOT / "backend/models/decision.json").read_text())
    bracol = pd.read_csv(ROOT / "data/bracol_dataset.csv").set_index("id")
    pred = pred[pred.split.isin(["test", "test_locked", "ood"])].copy()
    pred["truth"] = pred.task_label.fillna("out_of_scope")
    pred["out"] = pred.pred.where(pred.answered, "not_sure")
    pred["reason"] = np.where(pred.answered, "", np.where(pred.distance > decision["ood_threshold"],
                                                          "unfamiliar_image", "low_confidence"))
    pred["scenario"] = pred.apply(scenario, axis=1)
    pred = pred[pred.scenario.notna()].sample(frac=1, random_state=SEED)

    predictor = Predictor.from_dir(ROOT / "backend/models", 0.6, 3, ["healthy", "rust", "other"])
    OUT.mkdir(exist_ok=True)
    for f in OUT.glob("*.jpg"):
        f.unlink()

    keep, tmp = [], OUT / "_tmp.jpg"
    for sc, n in PLAN:
        pool = pred[pred.scenario == sc]
        # Spread each scenario over sources and original folders before taking repeats.
        pool = pool.assign(_k=pool.groupby(["source", "folder"]).cumcount()).sort_values("_k", kind="stable")
        got = 0
        for r in pool.itertuples():
            if got == n:
                break
            size = save_like_browser(ROOT / "data/raw" / r.path, tmp)
            res = predictor.predict(load_image(tmp.read_bytes()))
            again = r._replace(out=res["outcome"], reason=res.get("reason") or "")
            if scenario(again) != sc:  # answer changed after browser-style resizing: not a stable example
                continue
            got += 1
            pid = f"{sc.split(':')[0].replace('SAFE FAIL', 'S').replace('PASS', 'P').replace('FAIL', 'F').replace('MIXED', 'M')}"
            top = sorted(res["top3"], key=lambda g: -g["confidence"])
            keep.append({
                "scenario": sc, "id_prefix": pid, "source": r.source, "photo_style": STYLE[r.source],
                "split": r.split, "farm_id": r.farm_id if isinstance(r.farm_id, str) else "",
                "original_file": r.path, "original_folder_label": r.folder, "truth": r.truth,
                "bracol_symptoms": bracol_symptoms(r.path, bracol),
                "rust_level": r.folder.replace("rust_level_", "") if r.folder.startswith("rust_level_") else "",
                "uploaded_size": f"{size[0]}x{size[1]}", "brightness": round(r.brightness, 1),
                "sharpness": round(r.sharpness, 1),
                "app_outcome": res["outcome"], "app_reason": res.get("reason") or "",
                "app_label": res["label"], "app_confidence": res["confidence"],
                "app_second": top[1]["label"], "app_second_p": top[1]["confidence"],
                "app_distance": res["distance"], "_bytes": tmp.read_bytes(),
            })
        if got < n:
            print(f"warning: only {got}/{n} stable photos for {sc}")
    tmp.unlink(missing_ok=True)

    df = pd.DataFrame(keep)
    df["id"] = df.groupby("id_prefix").cumcount().add(1).map("{:02d}".format).radd(df.id_prefix)
    df["file"] = df.id + ".jpg"
    for i, r in df.iterrows():
        (OUT / r.file).write_bytes(r._bytes)
    df["pass_fail"] = df.scenario.str.split(":").str[0]
    facts_why = df.apply(lambda r: explain(r, decision), axis=1)
    df["measured_facts"] = [f for f, _ in facts_why]
    df["why"] = [w for _, w in facts_why]
    notes_file = OUT / "visual_notes.json"  # hand-written after looking at each photo (keyed by original_file)
    notes = json.loads(notes_file.read_text()) if notes_file.exists() else {}
    df["visual_note"] = df.original_file.map(notes).fillna("")
    for c in ("app_confidence", "app_second_p", "app_distance"):
        df[c] = df[c].round(3)
    cols = ["id", "file", "pass_fail", "scenario", "truth", "app_outcome", "app_reason", "app_label",
            "app_confidence", "app_second", "app_second_p", "app_distance", "why", "visual_note", "measured_facts",
            "source", "photo_style", "split", "farm_id", "original_folder_label", "rust_level", "bracol_symptoms",
            "brightness", "sharpness", "uploaded_size", "original_file"]
    df[cols].to_csv(OUT / "trial_cases.csv", index=False)
    contact_sheet(df[df.pass_fail == "PASS"], OUT / "_sheet_pass.jpg")
    contact_sheet(df[df.pass_fail != "PASS"], OUT / "_sheet_fail.jpg")
    print(df.groupby("scenario", sort=False).size().to_string())
    print(f"{len(df)} photos -> {OUT}")


if __name__ == "__main__":
    main()
