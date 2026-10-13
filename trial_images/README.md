# Trial photos for the app

51 photos for trying the app by hand. Each one shows a known pass or fail case of the installed model
(`backend/models`, Phase 2). Built by `ml/scripts/10_build_ui_trial_set.py`.

- Every photo comes from test, locked test or out-of-scope data, so the model **never trained on any of them**.
- Photos are saved the way the browser uploads them (rotated upright, longest side ≤ 1280 px, under 2 MB), so the
  app shows exactly the answer in `trial_cases.csv`. Checked through the API: 51/51 match.
- **The mix is not representative.** Failures were picked on purpose, one group per failure type. In normal use
  the model is right on about 96% of in-domain test photos and 99% of locked field photos (`docs/MODEL_CARD.md`).

## How to use
1. Start the backend (port 8100) and the frontend (`npm run dev`, http://localhost:5180).
2. Upload a photo, e.g. `F10.jpg`.
3. Compare the app's answer with `app_outcome`, then read `why` and `visual_note` in `trial_cases.csv`.

## IDs
| IDs | Group | What it shows |
|---|---|---|
| P01–P06 | PASS | healthy recognised |
| P07–P14 | PASS | rust recognised (all sources, RoCoLe rust levels) |
| P15–P20 | PASS | phoma / miner / cercospora answered "another problem (not rust)" |
| P21–P23 | PASS | mite damage (never trained on) answered Not sure |
| F01–F03 | FAIL | other disease called rust: look-alike spots, blur, doubtful labels |
| F04–F06 | FAIL | other disease called healthy: lesion too small to see at 224 px |
| F07–F09 | FAIL | rust + miner leaf answered "other": it reports the dominant problem |
| F10–F11 | FAIL | earliest rust (level 1) missed: tiny pustules in a wide field photo |
| F12 | FAIL | healthy leaf called rust (no visible cause) |
| F13–F15 | FAIL | mite damage called rust: the model has no "unknown" answer |
| F16–F17 | FAIL | mite damage called healthy: damage too subtle |
| S01–S04 | SAFE FAIL | known condition, Not sure: split between two answers |
| S05–S07 | SAFE FAIL | known condition, Not sure: photo unlike the training photos |
| M01–M04 | MIXED | BRACOL leaves with several stresses (no single right answer) |

## Columns of `trial_cases.csv`
- `id`, `file`: the photo to upload.
- `pass_fail`, `scenario`: PASS, FAIL (wrong answer), SAFE FAIL (Not sure on a known condition) or MIXED.
- `truth`: the correct answer (`out_of_scope` = a condition the model was never trained on).
- `app_outcome`, `app_reason`: what the app shows, and why it said Not sure (`low_confidence` or `unfamiliar_image`).
- `app_label`, `app_confidence`, `app_second`, `app_second_p`: top answer and runner-up probabilities.
- `app_distance`: how far the photo is from the training photos (limit 0.112).
- `why`: an explanation built from the measured numbers.
- `visual_note`: what is in the photo that causes the failure, written after looking at each failure photo.
- Metadata: `source`, `photo_style`, `split`, `farm_id`, `original_folder_label`, `rust_level`,
  `bracol_symptoms` (BRACOL's own per-leaf annotations), `brightness`, `sharpness`, `uploaded_size`,
  `original_file`.

`_sheet_pass.jpg` and `_sheet_fail.jpg` are thumbnail overviews of the photos.
