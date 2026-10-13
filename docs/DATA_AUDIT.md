# Data audit: coffee leaf datasets

Date: 2026-10-08. Generated tables, review sheets and per-image results are in `artifacts/audit/`
(rebuild with `ml/scripts/01_inspect_raw.py` then `ml/scripts/06_audit_data.py`).

## Method in one paragraph

Every file was decoded, checked, fingerprinted (MD5 for identical bytes, a rotation/flip-aware perceptual
hash) and embedded with DINOv2 ViT-S/14, a general-purpose image model with no coffee training. Two images
count as **the same image** if they are byte-identical, or within 4 hash bits allowing 90° rotations and
flips (close-up datasets only), or have embedding similarity ≥ 0.95 (close-ups) / ≥ 0.98 (any pair
involving whole-leaf photos). Those thresholds come from looking at sampled pairs: at 0.95–0.98,
close-up pairs are the same crop, but whole leaves on a white background are different leaves. Random pairs
inside the resulting groups were checked by eye and are the same lesion. Label reliability and expected
performance use a logistic regression on the embeddings ("linear probe"), always with groups (and RoCoLe
leaves) kept on one side of every split.

## Headline

| | |
|---|---|
| Readable files | 64,730 |
| Distinct files (byte-unique) | 8,459 |
| **Independent images in our 5-class task** | **4,665** (an image shared by two datasets counted once) |
| of which JMuBEN + JMuBEN2 | ~554 (and likely fewer, see below) out of 58,549 files |

**What to expect from training on the processed data** (linear-probe estimate; fine-tuning usually does
better *within* a dataset, but the gap between datasets is the part that carries over):

| Train on | Test on | Accuracy | Macro F1 | Classes compared |
|---|---|---|---|---|
| JMuBEN + Uganda (the original plan) | same datasets, grouped CV | 0.93 | 0.85–0.93 | all present |
| JMuBEN + Uganda | BRACOL | 0.44 | **0.29** | all 5 |
| JMuBEN + Uganda | RoCoLe | 0.72 | 0.71 | healthy, rust |
| everything except BRACOL | BRACOL | 0.42 | **0.26** | all 5 |
| everything except Uganda | Uganda | 0.50 | 0.47 | healthy, rust, phoma |
| everything except RoCoLe | RoCoLe | 0.77 | 0.78 | healthy, rust |
| BRACOL | RoCoLe | 0.77 | 0.82 | healthy, rust |
| RoCoLe | BRACOL | 0.96 | 0.94 | healthy, rust |

So: **~0.9 inside any one dataset, 0.26–0.5 macro F1 on a dataset of a different kind for the 5-class
task.** Healthy-vs-rust transfers between the two whole-leaf datasets (0.82–0.94); everything involving
JMuBEN transfers poorly. The pretrained Hugging Face baseline (trained on JMuBEN only) scored 0.21–0.26
macro F1 on Uganda/BRACOL, in line with the JMuBEN rows here.

## JMuBEN / JMuBEN2: what is really there

| Folder | Files | Byte-unique | Independent |
|---|---|---|---|
| JMuBEN Cerscospora | 7,681 | 322 | **46** |
| JMuBEN Leaf rust | 8,336 | 1,042 | **283** |
| JMuBEN Phoma | 6,571 | 691 | **128** |
| JMuBEN2 Healthy | 18,983 | 62 | **2** |
| JMuBEN2 Miner | 16,978 | 1,639 | **95** |

- 54,793 files are byte-identical copies (up to 520 copies of one file).
- The rest are augmentations: 90° rotations and flips (6,030 pairs found), crops and re-compression
  (1,256), colour shifts, and **arbitrary-angle rotations** (visible rotated corners). The last kind
  escapes both the hash and the 0.95 embedding threshold, so the true count is likely lower still.
- JMuBEN2 Healthy is essentially **two leaf crops**. A model trained on it has never seen healthy leaves.
- 36 JMuBEN "Leaf rust" images that the probe calls cercospora are one yellowed leaf with dark purple
  specks, rotated many times. It does not show orange rust pustules; it is a likely label error.
- Images are 128 px. Cercospora and Miner contain some 74–111 px images. 39% of JMuBEN is upscaled.
  Median sharpness is 11 (Uganda 57, BRACOL 182, RoCoLe 708).
- Camera EXIF: Huawei JAT-L29 and Infinix phones, Jan–Feb 2021 (Kenya, Arabica per the dataset paper).

## Uganda: which images/labels are reliable

| Folder | Files | Byte-unique | Independent | Empty files |
|---|---|---|---|---|
| Health leaves | 1,079 | 739 | 613 | 100 |
| leaf rust | 1,031 | 617 | 518 | 2 |
| phoma | 1,110 | 453 | 365 | 0 |

- 102 files in the original zip are **0 bytes**; 1,411 files are byte-identical copies.
- 18 files appear in **both** `leaf rust` and `phoma`, so their label is unknown. Drop them.
- **39 `phoma` images are JMuBEN Phoma images** (same crops, resized to 256 px). They must sit in the same
  split as their JMuBEN twins, or be removed from Uganda when it is used as an external test.
- 54% of the 256 px images are upscaled from something smaller. Field close-ups of part of a leaf,
  often blurry; real orange rust pustules are visible (unlike JMuBEN rust). No EXIF, no camera or date
  information, filenames carry only a class code. Coffee species is not stated in the download.
- Labels are **internally consistent**: the probe reaches 0.94 accuracy and flags only 20 images (1.1%).
  The flagged "phoma" images show small round orange spots with yellow halos, which look like early rust.
  Review list: `artifacts/audit/label_review.csv`, sheets `review_uganda_*.jpg`. Consistency within a
  dataset does not prove its convention matches the other datasets.

## BRACOL: mapping to our classes

| Code | Symptom columns set | Our class | In CSV | Images present | Leaves with >1 stress |
|---|---|---|---|---|---|
| 0 | none | healthy | 272 | 142 | – |
| 1 | miner | miner | 387 | 253 | 17% |
| 2 | rust | rust | 531 | 465 | 34% |
| 3 | phoma (brown leaf spot) | phoma | 348 | 346 | 2% |
| 4 | cercospora | cercospora | 147 | 136 | **41%** |
| 5 | several, none predominant | **excluded** | 62 | 59 | – |

- The CSV's own per-symptom columns confirm the code order (each code's column is 1.00).
- **Labels are "predominant stress", not exclusive.** 41% of cercospora leaves and 34% of rust leaves
  also carry another stress, mostly rust + cercospora. A single-label model will be penalised on these
  even when its answer is defensible.
- Severity is mostly level 1 (mild), so many lesions are small on a 2048 × 1024 photo.
- The published archive is truncated: 345 of 1,747 leaves are missing, and healthy is hit hardest
  (142 of 272 present).
- Whole detached leaves, upper side, on a white/grey background (99% plain edges). No EXIF.
  Arabica (Brazil) per the paper. Each file is one leaf (no duplicates found at the 0.98 threshold).

## RoCoLe: what belongs to the task

- 12 farms, 387 plants, **759 leaves**; every leaf is photographed **twice (top side H, underside E)**.
  Both photos of a leaf, and ideally the whole plant, must stay in one split.
- In scope: healthy (759 photos) and rust levels 1–4 merged into rust (568). Out of scope: red spider
  mite (166 photos, a pest).
- The two sides of a leaf get the same task label in **86.6%** of leaves. Disagreements are mostly
  rust ↔ spider mite (67 leaves) and healthy ↔ rust (24). 12 byte-identical photos are filed under both
  spider mite and a rust level. Drop the conflicting files; treat the healthy ↔ rust ones as noisy.
- Rust level 1 is subtle: healthy vs level 1 reaches 0.88 accuracy (0.96–0.98 for levels 2–4).
- The leaf covers ~26% of the photo; the rest is field background (soil, other leaves). The annotation
  polygons in `RoCoLE-csv.csv` outline the leaf and could be used to crop it.
- Samsung phones (4 models), April–May 2019. **Robusta** (Ecuador), the only robusta set we can confirm.

## All datasets compared

| | JMuBEN/2 | Uganda | BRACOL | RoCoLe |
|---|---|---|---|---|
| Typical size | 128 × 128 | 256 × 256 | 2048 × 1024 | 2048 × 1152 (720–4128) |
| Framing | close-up crop | close-up of part of a leaf | whole leaf | whole leaf on the plant |
| Background | leaf tissue | leaf + field | white/grey studio | field |
| Upscaled | 39% / 2% | 54% | no | no |
| Camera | Huawei, Infinix | unknown | unknown | Samsung ×4 |
| Coffee | Arabica | not stated | Arabica | Robusta |
| Classes | rust, phoma, cercospora / healthy, miner | healthy, rust, phoma | all 5 | healthy, rust |

**Independent images per class:**

| Class | JMuBEN/2 | Uganda | BRACOL | RoCoLe | Total |
|---|---|---|---|---|---|
| healthy | 2 | 613 | 138 | 749 | 1,502 |
| rust | 283 | 517 | 464 | 561 | 1,825 |
| miner | 95 | – | 253 | – | **348** |
| phoma | 128 | 365 | 346 | – | 839 |
| cercospora | 46 | – | 136 | – | **182** |

(RoCoLe counts photos; divide by ~2 for leaves.)

**Source shortcut risk is high.** Given only the image, the probe identifies *which dataset* it came from
with 96–100% accuracy within every class. In the original training pool (JMuBEN + Uganda), cercospora
and miner come 100% from one dataset and healthy 92% from Uganda. A model can score well there by
recognising the dataset rather than the disease, which is exactly the failure seen in the baseline.

## What this means for training

1. **Use independent images, not files.** Train on one representative per group (or weight by
   1/group size). Otherwise 19,000 copies of two healthy crops dominate.
2. **Split by group, leaf and plant.** Near-duplicate groups, RoCoLe leaves (both sides) and ideally
   RoCoLe plants must not straddle train/val/test.
3. **Mix framings in training.** Only BRACOL and RoCoLe teach whole leaves; only JMuBEN/Uganda teach
   close-ups. Phone photos are whole leaves, so the training mix must contain them.
4. **Keep one dataset out completely** as the honest external test (RoCoLe for healthy/rust, or BRACOL
   for all 5). Expect its macro F1 to land far below the in-domain number.
5. **Weak classes:** cercospora (182 independent, 46 from JMuBEN) and miner (348) will have unstable
   metrics; consider merging or reporting them separately. Miner is also a pest (leaf miner).
6. **Clean before training:** drop empty files, files with conflicting labels, BRACOL code 5 and RoCoLe
   spider mite; have an agronomist review `label_review.csv` (86 candidates) and the JMuBEN rust leaf above.
7. **Balance by source**, not just by class, so no class is learnt from a single dataset's look.

## Limits of this audit

- The probe uses frozen general-purpose features. A fine-tuned CNN will score higher in-domain; the
  cross-dataset numbers are indicative, not a guarantee in either direction.
- "Independent" relies on visual similarity. Different crops of the same leaf at very different scales,
  and arbitrary-angle rotations, can still survive as separate images (JMuBEN especially).
- Label flags are candidates for expert review, not corrections; I am not a plant pathologist.
- Coffee species and locations come from the dataset publications, not from the files (none for Uganda).
