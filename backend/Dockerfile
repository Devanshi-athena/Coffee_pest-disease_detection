# Build from the repo root so configs/ can be copied in:
#   docker build -f backend/Dockerfile -t coffee-leaf-backend .
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    CONFIG_PATH=/app/configs/config.yaml \
    MODEL_DIR=/app/models

WORKDIR /app
COPY backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY configs/ ./configs/
COPY backend/app/ ./app/
# Model files are mounted at runtime (see docker-compose.yml) or copied here for a self-contained image.
RUN mkdir -p /app/models && useradd --create-home appuser
USER appuser

EXPOSE 8100
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8100"]
