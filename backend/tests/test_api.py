"""API tests with a fake ONNX session, so they pass before a real model exists."""
import io
from dataclasses import replace

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from app.config import load_settings
from app.inference import Decision, PreprocessSpec, Predictor, preprocess, softmax
from app.main import create_app

SPEC = PreprocessSpec(resize_shorter=256, crop_size=224, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))


class _Input:
    name = "input"


class FakeSession:
    """Mimics onnxruntime.InferenceSession: fixed logits (and optional features) for every image."""

    def __init__(self, logits, features=None):
        self.logits = np.asarray(logits, dtype=np.float32)
        self.features = None if features is None else np.asarray(features, dtype=np.float32)
        self.last_input = None

    def get_inputs(self):
        return [_Input()]

    def run(self, _outputs, feeds):
        self.last_input = feeds["input"]
        n = feeds["input"].shape[0]
        out = [np.tile(self.logits, (n, 1))]
        if self.features is not None:
            out.append(np.tile(self.features, (n, 1)))
        return out


def jpeg_bytes(size=(640, 480), fmt="JPEG") -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", size, (40, 140, 60)).save(buf, fmt)
    return buf.getvalue()


@pytest.fixture
def settings(tmp_path):
    return replace(load_settings(), model_dir=tmp_path)  # empty dir: no model


@pytest.fixture
def client(settings):
    with TestClient(create_app(settings)) as c:
        yield c


def with_model(client, logits, threshold=0.6, features=None, decision=None):
    s = client.app.state.settings
    session = FakeSession(logits, features)
    client.app.state.predictor = Predictor(session, s.classes, SPEC, decision or Decision(threshold), top_k=3)
    return session


def test_health_without_model(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["model_loaded"] is False
    assert "model.onnx" in r.json()["detail"]


def test_classes_come_from_config(client):
    assert client.get("/classes").json()["classes"] == ["healthy", "rust", "other"]


def test_predict_returns_503_without_model(client):
    r = client.post("/predict", files={"file": ("leaf.jpg", jpeg_bytes(), "image/jpeg")})
    assert r.status_code == 503
    assert "Model not loaded" in r.json()["detail"]


def test_predict_confident(client):
    session = with_model(client, [0.1, 5.0, 0.2])
    r = client.post("/predict", files={"file": ("leaf.jpg", jpeg_bytes(), "image/jpeg")})
    assert r.status_code == 200
    body = r.json()
    assert body["label"] == "rust"
    assert body["confidence"] > 0.9
    assert body["uncertain"] is False
    assert [t["label"] for t in body["top3"]][0] == "rust" and len(body["top3"]) == 3
    assert session.last_input.shape == (1, 3, 224, 224)
    assert session.last_input.dtype == np.float32


def test_predict_uncertain_when_below_threshold(client):
    with_model(client, [1.0, 1.1, 0.9])
    r = client.post("/predict", files={"file": ("leaf.png", jpeg_bytes(fmt="PNG"), "image/png")})
    assert r.status_code == 200
    assert r.json()["uncertain"] is True


def test_rejects_wrong_type(client):
    with_model(client, [1, 0, 0])
    r = client.post("/predict", files={"file": ("notes.txt", b"hello", "text/plain")})
    assert r.status_code == 415


def test_rejects_corrupt_image(client):
    with_model(client, [1, 0, 0])
    r = client.post("/predict", files={"file": ("leaf.jpg", b"not really a jpeg", "image/jpeg")})
    assert r.status_code == 400


def test_rejects_too_large(settings):
    small = replace(settings, max_upload_bytes=1000)
    with TestClient(create_app(small)) as c:
        with_model(c, [1, 0, 0])
        r = c.post("/predict", files={"file": ("leaf.jpg", jpeg_bytes((1200, 1200)), "image/jpeg")})
        assert r.status_code == 413


def test_preprocess_shape_and_normalization():
    img = Image.new("RGB", (500, 300), (124, 116, 104))  # ~ImageNet mean colour
    x = preprocess(img, SPEC)
    assert x.shape == (1, 3, 224, 224)
    assert np.abs(x).max() < 0.05


def test_softmax_sums_to_one():
    p = softmax(np.array([2.0, 1.0, 0.1, -3.0, 0.0]))
    assert np.isclose(p.sum(), 1.0) and p.argmax() == 0


def _bank():
    rng = np.random.default_rng(0)
    b = rng.normal(size=(50, 8)).astype(np.float32)
    b[:, 0] += 6  # training features cluster around +x
    return b / np.linalg.norm(b, axis=1, keepdims=True)


def test_familiar_photo_is_answered(client):
    near = np.r_[1.0, np.zeros(7)]
    dec = Decision(confidence_threshold=0.6, ood_threshold=0.2, knn_k=5, tta_views=("center", "hflip"), bank=_bank())
    session = with_model(client, [0.1, 5.0, 0.2], features=near, decision=dec)
    body = client.post("/predict", files={"file": ("leaf.jpg", jpeg_bytes(), "image/jpeg")}).json()
    assert body["outcome"] == "rust" and body["uncertain"] is False and body["reason"] is None
    assert session.last_input.shape == (2, 3, 224, 224)  # both TTA views in one batch
    assert body["distance"] < 0.2


def test_unfamiliar_photo_is_not_sure_even_if_confident(client):
    far = np.r_[-1.0, np.zeros(7)]
    dec = Decision(confidence_threshold=0.6, ood_threshold=0.2, knn_k=5, bank=_bank())
    with_model(client, [0.1, 9.0, 0.2], features=far, decision=dec)
    body = client.post("/predict", files={"file": ("leaf.jpg", jpeg_bytes(), "image/jpeg")}).json()
    assert body["confidence"] > 0.99
    assert body["outcome"] == "not_sure" and body["reason"] == "unfamiliar_image"


def test_low_confidence_reason(client):
    with_model(client, [1.0, 1.1, 0.9])
    body = client.post("/predict", files={"file": ("leaf.jpg", jpeg_bytes(), "image/jpeg")}).json()
    assert body["outcome"] == "not_sure" and body["reason"] == "low_confidence"


def test_serves_frontend_and_keeps_api_routes(settings, tmp_path):
    site = tmp_path / "site"
    site.mkdir()
    (site / "index.html").write_text("<title>Coffee Leaf AI</title>")
    with TestClient(create_app(replace(settings, static_dir=site))) as c:
        assert "Coffee Leaf AI" in c.get("/").text
        assert c.get("/health").json()["status"] == "ok"
