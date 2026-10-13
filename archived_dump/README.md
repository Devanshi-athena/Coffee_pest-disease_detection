# archived_dump

Superseded work, archived on 2026-10-08. Folders mirror their original location in the repo, so anything can
be restored by moving it back to the same relative path.

| Archived | What it was | Why archived |
|---|---|---|
| `backend/models/` (model.onnx, labels.json, preprocess.json, model_info.json) | ONNX export of the pretrained baseline [Huyt/arabica-coffee-leaf-disease-efficientnet-b0](https://huggingface.co/Huyt/arabica-coffee-leaf-disease-efficientnet-b0) | Trained on JMuBEN only; 0.21–0.30 accuracy on Uganda/BRACOL, 0.03 on RoCoLe, confidently wrong. Not safe to serve. |
| `artifacts/hf_model/` | Downloaded weights (safetensors), config and the author's scripts | Source of the model above |
| `artifacts/eval_baseline/` | External evaluation of that model (report.md, metrics, plots, examples) | Results are summarised in the conversation and docs/DATA_AUDIT.md |
| `ml/scripts/import_hf_baseline.py` | Converts the HF model to ONNX + parity check vs timm | Baseline-specific |
| `ml/scripts/05_evaluate_model.py` | Evaluates the served ONNX model on all datasets | Written for the 5-class baseline; to be adapted for Phase 1 |
| `ml/scripts/02_build_metadata.py`, `03_cluster_split.py`, `04_make_processed.py` | Original 5-class prep pipeline | Replaced by `ml/scripts/07_build_phase1_dataset.py`, which uses the audit's rotation- and embedding-aware grouping |
| `ml/notebooks/train_kaggle.ipynb` | 5-class training notebook skeleton | Phase 1 (healthy / rust / other) gets its own notebook |

Not archived because current steps still need them: `Data_dump/` (steps 6 and 8 read RoCoLe annotations from it),
`artifacts/inspect/` and `artifacts/audit/` (inputs to steps 6 and 8), `ml/src/`.

Note: the moved scripts use paths relative to the repo root, so run them from their original location if restored.

## Archived on 2026-10-12 (cleanup after Phase 2 became the baseline)

| Archived | What it was | Why archived |
|---|---|---|
| `ml/scripts/07_build_phase1_dataset.py` | Phase 1 dataset builder (healthy vs rust) | Replaced by `08_build_phase2_dataset.py` |
| `ml/src/phase1.py`, `ml/notebooks/train_phase1.ipynb`, `train_phase1.py` | Phase 1 training code and notebook (the `.py` was a stale copy) | Phase 1 was an experiment; Phase 2 is the baseline |
| `ml/src/dataset.py`, `ml/src/transforms.py` | Helpers of the original 5-class pipeline | Not imported by anything current (`phase2.py` has its own) |
| `data/processed_phase1/`, `data/phase1_kaggle.zip` | Phase 1 dataset and its Kaggle upload | Only needed to reproduce Phase 1 |

`artifacts/eval_baseline/predictions.csv` (a copy left behind by an Excel lock) was deleted: it was byte-identical
to `archived_dump/artifacts/eval_baseline/predictions.csv`. The `phase1:` block stays in `configs/config.yaml` so
the archived script still runs if restored.
