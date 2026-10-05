from PIL import Image

from src import detect
from src.detect import _dedup, annotate


def test_fridge_classes_are_sane():
    assert detect.FRIDGE_CLASSES[47] == "apple"
    assert detect.FRIDGE_CLASSES[39] == "bottle"
    assert len(detect.FRIDGE_CLASSES) >= 10


def test_dedup_removes_overlapping_same_label():
    dets = [
        {"label": "apple", "confidence": 0.9, "bbox": (0, 0, 100, 100)},
        {"label": "apple", "confidence": 0.4, "bbox": (5, 5, 95, 95)},
        {"label": "banana", "confidence": 0.8, "bbox": (5, 5, 95, 95)},
    ]
    kept = _dedup(dets)
    labels = sorted(d["label"] for d in kept)
    assert labels == ["apple", "banana"]
    assert max(d["confidence"] for d in kept if d["label"] == "apple") == 0.9


def test_dedup_keeps_separate_boxes():
    dets = [
        {"label": "apple", "confidence": 0.9, "bbox": (0, 0, 10, 10)},
        {"label": "apple", "confidence": 0.8, "bbox": (50, 50, 60, 60)},
    ]
    assert len(_dedup(dets)) == 2


def test_annotate_returns_new_image_with_boxes():
    img = Image.new("RGB", (200, 200), (255, 255, 255))
    dets = [{"label": "apple", "confidence": 0.95, "bbox": (10, 10, 60, 60)}]
    out = annotate(img, dets)
    assert out.size == img.size
    assert out is not img
    # green box border should appear near the top-left corner of the box
    assert out.getpixel((11, 30)) == (46, 204, 113)


def test_annotate_empty_detections_is_identity():
    img = Image.new("RGB", (50, 50), (10, 20, 30))
    out = annotate(img, [])
    assert list(out.getdata()) == list(img.getdata())
