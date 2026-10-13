# Model card: coffee leaf screener (Phase 2, `hro_v2`)

Installed in `backend/models/` on 2026-10-09 (`model_info.json` holds file checksums).

## What it does
Looks at a photo of **one coffee leaf** and answers one of:

| Answer | Meaning |
|---|---|
| `rust` | signs that look like coffee leaf rust |
| `healthy` | no signs of rust or of the other leaf diseases it knows |
| `other` | a leaf problem that is not rust (trained on phoma, cercospora, leaf miner) |
| `not_sure` | low confidence, or the photo is unlike the training photos |

It is a screening aid, not a diagnosis.

## Model
- ConvNeXt-Tiny (timm `convnext_tiny.fb_in22k_ft_in1k`), fine-tuned, 224 px input
  (resize shorter side to 256, centre crop 224, ImageNet normalisation).
- ONNX with two outputs: temperature-scaled logits (T = 0.731) and L2-normalised features.
- Seed 2 of 3, chosen on validation; best epoch 14 of 15.

## Decision rule (`decision.json`, fitted on validation only)
Answer only if **max probability ≥ 0.77** and **kNN distance to the training features ≤ 0.112**
(k = 10, bank of 2,700 training images in `ood_bank.npz`); otherwise `not_sure`.
- 0.77 = lowest threshold giving ≥ 98% accuracy on answered validation photos.
- 0.112 = 95th percentile of validation distances (5% of normal photos are flagged by design).
- Test-time augmentation was evaluated (flips, multi-scale) and not used: the plain centre crop scored best on
  validation.

## Data
Five public datasets (JMuBEN, JMuBEN2, Uganda, BRACOL, RoCoLe), deduplicated and audited
(`docs/DATA_AUDIT.md`), built by `ml/scripts/08_build_phase2_dataset.py` (`data/processed_phase2/DATASET.md`).
- Splits grouped by near-duplicate group; RoCoLe split by whole farm, with farms that share photos kept together.
- `test_locked` = RoCoLe farms C1, C2, C4, C8, never used for any choice.
- Balanced sampling: each class 1/3, each source equal within a class, near-duplicates counted once.
- Training views: close-ups whole; whole leaves as leaf box + lesion-aware zoom crops (+ full photo for RoCoLe).

## Results (backend pipeline on original photos, `ml/scripts/09_calibrate_and_evaluate.py`)
| Set | n | Accuracy | Macro F1 | Answered | Accuracy when answered |
|---|---|---|---|---|---|
| Validation | 806 | 0.964 | 0.963 | 94.3% | 0.980 |
| Test (in-domain) | 603 | 0.959 | 0.960 | 92.7% | 0.979 |
| Test, BRACOL | 199 | 0.930 | 0.913 | 95.0% | 0.952 |
| Test, JMuBEN | 116 | 0.974 | 0.975 | 89.7% | 0.990 |
| Test, JMuBEN2 | 27 | 1.000 | 1.000 | 88.9% | 1.000 |
| Test, Uganda | 261 | 0.969 | 0.964 | 92.7% | 0.992 |
| **Locked RoCoLe farms** (healthy/rust) | 405 | **0.990** | **0.990** | 98.5% | **0.995** |

- Other-disease photos answered as rust: 1.3% (test), 2.7% (validation).
- Calibration error (ECE): 0.013 validation, 0.019 locked farms.
- Seed stability (in-domain test macro F1): 0.962 ± 0.003.
- API check: 60/60 identical answers to the evaluation script; latency median 0.14 s, max 0.46 s (CPU).

**Leave-one-dataset-out** (train without one dataset, test on it; estimate for an unfamiliar photo style):
RoCoLe 0.896 · Uganda 0.586 · BRACOL 0.475 · JMuBEN 0.431 macro F1.

**Out-of-scope** (142 photos never trained on): 14% flagged `not_sure`, 78% answered `rust`.
Spider mite: 19% flagged. BRACOL mixed-stress leaves: 7% flagged, but 87% of them do have rust.

## Known limitations
1. **Unknown conditions are not reliably caught.** A problem the model never saw (e.g. mite damage) is
   usually answered `rust`. The app words a rust answer as "rust-like symptoms" and says that other problems can
   look similar.
2. **Unfamiliar photo styles are uncertain.** Leave-one-dataset-out scores of 0.43–0.90 show that accuracy can
   drop a lot on photos unlike the training data. The 0.99 on locked RoCoLe farms is "new farms, familiar photo
   style".
3. Trained on Kenya, Uganda, Brazil and Ecuador data (Arabica and Robusta); other regions and varieties untested.
4. One condition per leaf; leaves with several problems may get a misleading answer.
5. No evaluation on real user photos yet.

## Reproduce
```
ml/scripts/01_inspect_raw.py -> 06_audit_data.py -> 08_build_phase2_dataset.py
Kaggle: ml/notebooks/train_phase2.ipynb on the dataset made from data/phase2_kaggle.zip
ml/scripts/09_calibrate_and_evaluate.py (on the downloaded outputs) -> backend/models/
```
