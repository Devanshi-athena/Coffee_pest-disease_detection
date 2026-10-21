"""Vercel Python function: the FastAPI backend behind /api/* (vercel.json rewrites /api/(.*) here).

Model files: bundled into the function by the build step (backend/models/, see vercel.json) when possible,
otherwise downloaded once per cold start into /tmp from the GitHub release and checksum-verified.
The model is loaded at import time, so it works whether or not the platform sends ASGI lifespan events.
"""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))
sys.path.insert(0, str(ROOT / "deploy"))

model_dir = ROOT / "backend" / "models"
if not (model_dir / "model.onnx").is_file():
    from fetch_model import fetch

    model_dir = fetch(Path("/tmp/models"))
os.environ.setdefault("MODEL_DIR", str(model_dir))
os.environ.setdefault("CONFIG_PATH", str(ROOT / "configs" / "config.yaml"))

from app.config import load_settings  # noqa: E402
from app.inference import Predictor  # noqa: E402
from app.main import create_app  # noqa: E402

_settings = load_settings()
_api = create_app(_settings)
_api.state.predictor = Predictor.from_dir(_settings.model_dir, _settings.confidence_threshold, _settings.top_k,
                                          _settings.classes, _settings.threshold_override)
_api.state.model_error = None


async def app(scope, receive, send):
    """ASGI entry point: answer lifespan events here (the model is already loaded) and serve the API under /api."""
    if scope["type"] == "lifespan":
        while True:
            message = await receive()
            if message["type"] == "lifespan.startup":
                await send({"type": "lifespan.startup.complete"})
            elif message["type"] == "lifespan.shutdown":
                await send({"type": "lifespan.shutdown.complete"})
                return
    path = scope.get("path", "")
    if path == "/api" or path.startswith("/api/"):
        scope = {**scope, "path": path[4:] or "/"}
    await _api(scope, receive, send)
