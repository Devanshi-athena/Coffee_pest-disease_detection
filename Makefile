# Needs: uv, node/npm, (docker for docker-up). On Windows run from Git Bash, or use the
# plain commands listed in README.md.
ifeq ($(OS),Windows_NT)
  BIN := Scripts
else
  BIN := bin
endif
PREP_PY := ml/.venv/$(BIN)/python
EVAL_PY := ml/.venv-eval/$(BIN)/python
API_PY := .venv/$(BIN)/python

.PHONY: setup setup-prep setup-eval setup-backend setup-frontend prep test backend frontend docker-up

setup: setup-prep setup-eval setup-backend setup-frontend

setup-prep:
	uv venv ml/.venv --python 3.11
	uv pip install --python ml/.venv -r ml/requirements-prep.txt

setup-eval:
	uv venv ml/.venv-eval --python 3.11
	uv pip install --python ml/.venv-eval torch torchvision --index-url https://download.pytorch.org/whl/cpu
	uv pip install --python ml/.venv-eval -r ml/requirements-eval.txt

setup-backend:
	uv venv backend/.venv --python 3.11
	uv pip install --python backend/.venv -r backend/requirements.txt

setup-frontend:
	cd frontend && npm ci

prep:
	$(PREP_PY) ml/scripts/01_inspect_raw.py
	$(EVAL_PY) ml/scripts/06_audit_data.py
	$(PREP_PY) ml/scripts/08_build_phase2_dataset.py

test:
	cd backend && $(API_PY) -m pytest -q

backend:
	cd backend && $(API_PY) -m uvicorn app.main:app --reload --port 8100

frontend:
	cd frontend && npm run dev

docker-up:
	docker compose up --build
