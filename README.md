# coffee-leaf-ai

Prototype that checks a photo of a coffee leaf and says whether it shows **leaf rust**, **another leaf
disease**, or looks **healthy**, with a confidence score and an honest "not sure" answer.
FastAPI backend (onnxruntime, no PyTorch) + React/Vite frontend.

**Status (Phase 2):** a trained model is installed in `backend/models/` (healthy / rust / other + "not sure").
Metrics, decision rule and limitations: `docs/MODEL_CARD.md`. The earlier pretrained baseline and the original
5-class pipeline are in `archived_dump/` (see its README for why).

```
local CPU: inspect -> audit -> build dataset  ──zip──▶  Kaggle/Colab GPU: train + ONNX export  ──▶  backend/models/  ──▶  app
```

## Layout

| Path | What |
|---|---|
| `configs/config.yaml` | classes, paths, preprocessing, and the `phase2:` dataset settings. Everything reads this. |
| `ml/scripts/01_inspect_raw.py` | per-file checks, exact and near-duplicate search (rotation/flip aware) |
| `ml/scripts/06_audit_data.py` | deep audit: independent images, label reliability, framing, shortcut risk, expected performance |
| `ml/scripts/08_build_phase2_dataset.py` | Phase 2 dataset: 3 classes, RoCoLe farm split with locked test farms, lesion-aware crops |
| `ml/notebooks/train_phase2.ipynb` + `ml/src/phase2.py` | Kaggle training: leave-one-dataset-out, 3 seeds, calibration, "not sure" rule, ONNX export |
| `ml/scripts/09_calibrate_and_evaluate.py` | runs the exported model through the backend pipeline on original photos; fixes the decision rule on validation |
| `ml/scripts/10_build_ui_trial_set.py` | builds `trial_images/`: pass and fail examples for trying the app by hand |
| `ml/src/` | shared code: config/labels, image reading + hashing, crops, splits, evaluation, ONNX export, Phase 2 training |
| `docs/DATA_AUDIT.md` | findings of the data audit and what they mean for training |
| `docs/MODEL_CARD.md` | the served (baseline) model: metrics per dataset, decision rule, limitations |
| `backend/` | FastAPI app; `models/` holds the exported model |
| `frontend/` | React + Vite + TypeScript, mobile-first |
| `trial_images/` | 51 photos + `trial_cases.csv` (pass/fail and why); photos are not in git, rebuild with step 10 |
| `archived_dump/` | superseded work: pretrained HF baseline, 5-class pipeline, Phase 1 experiment (see its README) |

## Prerequisites

- [uv](https://docs.astral.sh/uv/) (it downloads Python 3.11 for you), Node.js 20.19+ (22 recommended)
- Optional: GNU make, Docker

Separate Python 3.11 virtual envs, so the backend never pulls in the ML stack:

| env | path | requirements | used for |
|---|---|---|---|
| data prep | `ml/.venv` | `ml/requirements-prep.txt` | steps 01 and 08 |
| analysis | `ml/.venv-eval` | `ml/requirements-eval.txt` (+ CPU torch) | steps 06, 09, 10 |
| backend | `backend/.venv` | `backend/requirements.txt` | the API |

Training deps (`ml/requirements-train.txt`) are for Kaggle/Colab, where torch is preinstalled.

## 1. Setup

```bash
make setup
```
or without make (works in PowerShell and bash):
```bash
uv venv ml/.venv --python 3.11
uv pip install --python ml/.venv -r ml/requirements-prep.txt
uv venv ml/.venv-eval --python 3.11
uv pip install --python ml/.venv-eval torch torchvision --index-url https://download.pytorch.org/whl/cpu
uv pip install --python ml/.venv-eval -r ml/requirements-eval.txt
uv venv backend/.venv --python 3.11
uv pip install --python backend/.venv -r backend/requirements.txt
cd frontend && npm ci && cd ..
```
On macOS/Linux replace `Scripts/` with `bin/` in the commands below.

## 2. Data

Raw downloads (zips) live in `Data_dump/`; unpacked datasets in `data/raw/<source>/<class folder>/`:

```
data/raw/jmuben/{Cerscospora,Leaf rust,Phoma}/      # JMuBEN  (Mendeley t2r6rszp5c)
data/raw/jmuben2/{Healthy,Miner}/                   # JMuBEN2 (Mendeley tgv3zb82nd)
data/raw/uganda/{Health leaves,leaf rust,phoma}/
data/raw/bracol/{healthy,miner,rust,phoma,cercospora,unmapped_stress_code_5}/   # sorted from dataset.csv
data/raw/rocole/{healthy,rust_level_1..4,red_spider_mite}/                    # sorted from RoCoLe-classes.xlsx
```
Folder names map to classes through `label_aliases` in `config.yaml`. `Data_dump/` must stay: steps 06 and 08 read
RoCoLe's leaf annotations from its zip.

## 3. Inspect and audit

```bash
ml/.venv/Scripts/python      ml/scripts/01_inspect_raw.py           # -> artifacts/inspect/
ml/.venv-eval/Scripts/python ml/scripts/06_audit_data.py            # -> artifacts/audit/ (docs/DATA_AUDIT.md explains it)
```
(`make prep` runs these two and step 08.) Every script takes `--config` and `--help`.
Phase 1 (healthy vs rust only, step 07) was an experiment and is archived in `archived_dump/`.

## 4. Phase 2: build, train, calibrate

```bash
ml/.venv/Scripts/python      ml/scripts/08_build_phase2_dataset.py     # -> data/processed_phase2/ + data/phase2_kaggle.zip
# Kaggle: upload data/phase2_kaggle.zip as a private dataset, import ml/notebooks/train_phase2.ipynb,
# add the dataset as input, GPU T4 + internet on, Save Version -> Save & Run All. Download its outputs.
ml/.venv-eval/Scripts/python ml/scripts/09_calibrate_and_evaluate.py --model-dir <downloaded outputs>
# copy model.onnx, labels.json, preprocess.json, ood_bank.npz and artifacts/phase2_eval/decision.json to backend/models/
```
The notebook checks its own code checksums, trains leave-one-dataset-out models and 3 seeds, picks everything on
validation, and evaluates `test_locked` (RoCoLe farms never used for any choice) once. Step 09 re-runs the model
through the backend's own preprocessing on the original photos and writes the decision rule (`decision.json`).

## 5. Run the app

`backend/models/` must contain `model.onnx`, `labels.json`, `preprocess.json`, and for the "not sure" rule `decision.json` + `ood_bank.npz`. Then:

```bash
# terminal 1 (from backend/)
.venv/Scripts/python -m uvicorn app.main:app --reload --port 8100     # or: make backend
# terminal 2 (from frontend/)
npm run dev                                                           # or: make frontend
```
Open http://localhost:5180 (the frontend calls the API at http://127.0.0.1:8100; on Windows `localhost` can add ~2 s per request). To test on a phone, run `npm run dev -- --host`, set
`VITE_API_URL=http://<your-LAN-IP>:8100` in `frontend/.env.local`, and add that frontend origin to `CORS_ORIGINS`.

Without a model, the backend still starts: `/health` reports `"model_loaded": false` and `/predict` returns 503.

Docker: `docker compose up --build` (backend :8100, frontend :8180; `backend/models` is mounted read-only).

### Public demo (Render, free tier)

`render.yaml` + `deploy/render/Dockerfile`: one container builds the frontend and serves it with the API on Render's
port. The model is downloaded at build time from the GitHub release `baseline-v1` and checked against
`deploy/render/model_files.sha256`. `ORT_THREADS=1` suits the free tier's fractional CPU.
Set up once: render.com -> New -> Blueprint -> this repo -> Apply. Every push to `main` redeploys.
Free tier: sleeps after 15 min idle (about a minute to wake), a few seconds per photo.

To ship a new model: upload its files to a new release, point `MODEL_RELEASE` in the Dockerfile at it and
regenerate `model_files.sha256`.

### Public demo (Vercel, Hobby)

`vercel.json` builds the frontend with `VITE_API_URL=/api` and serves the backend as a Python function
(`api/index.py`, rewrites `/api/(.*)`). The model comes from the `baseline-v1` release via `deploy/fetch_model.py`
(checksums in `deploy/render/model_files.sha256`): bundled at build time when possible, else downloaded to `/tmp`
on cold start. Set up once: vercel.com -> Add New -> Project -> import this repo -> Deploy. Hobby is free but for
non-commercial use. Render and Vercel both redeploy on every push to `main`.

`deploy/huggingface/` does the same for a Hugging Face Docker Space (`push_space.py`); since mid-2026 Docker
Spaces need a PRO subscription.

### API

| | |
|---|---|
| `GET /health` | `{"status":"ok","model_loaded":true,"detail":null}` |
| `GET /classes` | `{"classes":[...]}` |
| `POST /predict` | multipart field `file` (JPEG/PNG/WebP, ≤ `max_upload_mb`) -> `{"label","confidence","top3":[{label,confidence}],"uncertain","reason","outcome","threshold","distance"}`; `outcome` is `healthy`/`rust`/`other`/`not_sure`, `reason` is `low_confidence`/`unfamiliar_image` |

Backend env vars: `MODEL_DIR`, `CONFIG_PATH`, `CONFIDENCE_THRESHOLD`, `MAX_UPLOAD_MB`, `CORS_ORIGINS` (comma-separated).
Frontend: `VITE_API_URL` (build time, see `frontend/.env.example`).

## Tests / CI

```bash
cd backend && .venv/Scripts/python -m pytest -q      # or: make test
cd frontend && npm run lint && npm run build
```
The tests use a fake ONNX session, so they pass without a model. `.github/workflows/ci.yml` runs the same steps.

## Scope

This covers leaf diseases from the photo of a single leaf. It does not detect pests in general.
Leaf miner is recognized only from the mines on the leaf. See the About page in the app for user-facing limitations.
