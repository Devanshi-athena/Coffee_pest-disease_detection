"""FastAPI app: GET /health, GET /classes, POST /predict."""
from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from .config import Settings, load_settings
from .inference import MODEL_FILES, InvalidImage, ModelNotAvailable, Predictor, load_image

log = logging.getLogger("coffee-leaf-ai")
ALLOWED_TYPES = {"image/jpeg", "image/png", "image/webp"}


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or load_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.predictor, app.state.model_error = None, None
        try:
            app.state.predictor = Predictor.from_dir(settings.model_dir, settings.confidence_threshold,
                                                     settings.top_k, settings.classes, settings.threshold_override)
            log.info("Model loaded from %s", settings.model_dir)
        except ModelNotAvailable as e:
            app.state.model_error = str(e)
            log.warning("No model loaded (%s). /predict will return 503.", e)
        yield

    app = FastAPI(title="coffee-leaf-ai", version="0.1.0", lifespan=lifespan)
    app.state.settings = settings
    app.add_middleware(CORSMiddleware, allow_origins=settings.cors_origins,
                       allow_methods=["GET", "POST"], allow_headers=["*"])

    @app.get("/health")
    def health(request: Request):
        return {"status": "ok", "model_loaded": request.app.state.predictor is not None,
                "detail": request.app.state.model_error}

    @app.get("/classes")
    def classes(request: Request):
        predictor = request.app.state.predictor
        return {"classes": predictor.labels if predictor else settings.classes}

    # Sync endpoint: FastAPI runs it in a thread pool, so CPU-bound inference doesn't block the event loop.
    @app.post("/predict")
    def predict(request: Request, file: UploadFile = File(...)):
        predictor: Predictor | None = request.app.state.predictor
        if predictor is None:
            raise HTTPException(503, f"Model not loaded. Put {', '.join(MODEL_FILES)} in "
                                     f"{settings.model_dir} and restart the backend.")
        if file.content_type not in ALLOWED_TYPES:
            raise HTTPException(415, f"Unsupported file type '{file.content_type}'. Use JPEG, PNG or WebP.")
        data = file.file.read(settings.max_upload_bytes + 1)
        if len(data) > settings.max_upload_bytes:
            raise HTTPException(413, f"File too large (max {settings.max_upload_bytes // (1024 * 1024)} MB).")
        if not data:
            raise HTTPException(400, "Empty file.")
        try:
            img = load_image(data)
        except InvalidImage as e:
            raise HTTPException(400, str(e)) from e
        return predictor.predict(img)

    return app


app = create_app()
