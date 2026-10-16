# Render web service (render.yaml at the repo root): one container serves the API and the built frontend.
# The model is not in git: it is downloaded from the GitHub release and checked against model_files.sha256.
# Build context: repo root.

# 1. Build the frontend. An empty VITE_API_URL makes the app call the API on its own origin.
FROM node:22-alpine AS web
WORKDIR /web
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci --no-audit --no-fund
COPY frontend/ ./
RUN printf 'VITE_API_URL=\n' > .env.production.local && npm run build

# 2. Backend + model + static site.
FROM python:3.11-slim
ARG MODEL_RELEASE=https://github.com/Devanshi-athena/Coffee_pest-disease_detection/releases/download/baseline-v1
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    CONFIG_PATH=/app/configs/config.yaml \
    MODEL_DIR=/app/models \
    STATIC_DIR=/app/static
WORKDIR /app
COPY backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY deploy/render/model_files.sha256 /app/models/
RUN cd /app/models && for f in $(sed 's/.*[ *]//' model_files.sha256); do \
      python -c "import sys, urllib.request; urllib.request.urlretrieve(sys.argv[1], sys.argv[2])" "$MODEL_RELEASE/$f" "$f"; \
    done && sha256sum -c model_files.sha256

COPY configs/ ./configs/
COPY backend/app/ ./app/
COPY --from=web /web/dist ./static
RUN useradd --create-home appuser
USER appuser

# Render sets PORT (default 10000).
EXPOSE 10000
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-10000}"]
