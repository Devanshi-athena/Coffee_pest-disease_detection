# Hugging Face Space (Docker SDK): one container serves the API and the built frontend on port 7860.
# push_space.py stages this file at the Space root next to frontend/, backend/ and configs/.

# 1. Build the frontend. An empty VITE_API_URL makes the app call the API on its own origin.
FROM node:22-alpine AS web
WORKDIR /web
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci --no-audit --no-fund
COPY frontend/ ./
RUN printf 'VITE_API_URL=\n' > .env.production.local && npm run build

# 2. Backend + model + static site.
FROM python:3.11-slim
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    CONFIG_PATH=/app/configs/config.yaml \
    MODEL_DIR=/app/models \
    STATIC_DIR=/app/static
# Spaces run the container as uid 1000.
RUN useradd -m -u 1000 user
WORKDIR /app
COPY backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY configs/ ./configs/
COPY backend/app/ ./app/
COPY backend/models/ ./models/
COPY --from=web /web/dist ./static
USER user
EXPOSE 7860
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860"]
