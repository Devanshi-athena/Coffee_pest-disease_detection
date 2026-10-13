# backend/models

The served model (gitignored binaries). Required:

- `model.onnx` – classifier; outputs temperature-scaled logits and (Phase 2) L2-normalised features
- `labels.json` – class names in output order
- `preprocess.json` – resize/crop/mean/std used at training time

Optional, for the "not sure" rule:

- `decision.json` – confidence threshold, distance threshold, kNN k, test-time views (fitted on validation)
- `ood_bank.npz` – training-image features for the distance check
- `model_info.json` – provenance and checksums

Details and metrics: `docs/MODEL_CARD.md`. Restart the backend after replacing files. Without a model,
`/predict` returns 503 and `/health` reports `"model_loaded": false`.
