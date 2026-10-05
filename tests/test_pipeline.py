from PIL import Image

import src.pipeline as pipeline


class FakeModel:
    def score(self, crop):
        return 0.9


def test_analyze_wires_stages_together(monkeypatch):
    img = Image.new("RGB", (100, 100), (255, 255, 255))
    dets = [
        {"label": "apple", "confidence": 0.9, "bbox": (10, 10, 40, 40)},
        {"label": "banana", "confidence": 0.8, "bbox": (50, 50, 90, 90)},
    ]
    monkeypatch.setattr(pipeline, "detect_items", lambda image, conf=0.35: dets)
    monkeypatch.setattr(pipeline, "get_model", lambda: FakeModel())

    result = pipeline.analyze(img)
    assert {d["label"] for d in result["detections"]} == {"apple", "banana"}
    assert all(d["freshness"] == 0.9 for d in result["detections"])
    assert len(result["inventory"]) == 2
    assert result["inventory"][0]["item"] == "banana"  # use-soon-first
    assert len(result["recipes"]) > 0
    assert isinstance(result["annotated"], Image.Image)


def test_analyze_handles_no_detections(monkeypatch):
    img = Image.new("RGB", (100, 100), (255, 255, 255))
    monkeypatch.setattr(pipeline, "detect_items", lambda image, conf=0.35: [])
    monkeypatch.setattr(pipeline, "get_model", lambda: FakeModel())
    result = pipeline.analyze(img)
    assert result["detections"] == []
    assert result["inventory"] == []
    assert result["recipes"] == []
