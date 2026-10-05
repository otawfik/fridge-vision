import io

import pytest
from PIL import Image

import app as app_module


@pytest.fixture()
def client(monkeypatch):
    monkeypatch.setattr(
        app_module,
        "analyze",
        lambda image: {
            "detections": [{"label": "apple", "confidence": 0.9}],
            "inventory": [{
                "item": "apple", "count": 1, "confidence": 0.9,
                "freshness": 0.92, "freshness_label": "fresh",
                "days_left": 19, "ready_to_eat": False, "tip": "Keep cool.",
            }],
            "recipes": [({"title": "Apple Snack", "ingredients": ["apple"],
                          "time_min": 5, "steps": ["Slice."]}, 0.8)],
            "annotated": Image.new("RGB", (64, 64), (200, 200, 200)),
        },
    )
    app_module.app.config["TESTING"] = True
    return app_module.app.test_client()


def test_index_renders(client):
    resp = client.get("/")
    assert resp.status_code == 200
    assert b"Fridge Vision" in resp.data


def test_analyze_without_file_shows_error(client):
    resp = client.post("/analyze", data={})
    assert resp.status_code == 200
    assert b"choose a photo" in resp.data


def test_analyze_with_photo_renders_results(client):
    buf = io.BytesIO()
    Image.new("RGB", (64, 64), (255, 0, 0)).save(buf, format="JPEG")
    buf.seek(0)
    resp = client.post("/analyze", data={"photo": (buf, "fridge.jpg")})
    assert resp.status_code == 200
    assert b"apple" in resp.data
    assert b"data:image/jpeg;base64" in resp.data


def test_analyze_rejects_non_image(client):
    resp = client.post("/analyze", data={"photo": (io.BytesIO(b"nope"), "x.txt")})
    assert resp.status_code == 200
    assert b"Could not read" in resp.data
