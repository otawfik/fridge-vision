import json

import numpy as np
import pytest
from PIL import Image

from src.features import FEATURE_DIM
from src.freshness import FreshnessModel


@pytest.fixture()
def toy_model_path(tmp_path):
    blob = {
        "scaler_mean": [0.0] * FEATURE_DIM,
        "scaler_scale": [1.0] * FEATURE_DIM,
        # weight only the top value-histogram bin: bright images score fresh
        "coef": [5.0 if i == 27 else 0.0 for i in range(FEATURE_DIM)],
        "intercept": -2.0,
        "metrics": {"test_accuracy": 0.9},
    }
    path = tmp_path / "toy.json"
    path.write_text(json.dumps(blob))
    return str(path)


def test_score_is_probability(toy_model_path):
    model = FreshnessModel(toy_model_path)
    img = Image.new("RGB", (64, 64), (200, 200, 200))
    s = model.score(img)
    assert 0.0 <= s <= 1.0


def test_label_threshold(toy_model_path):
    model = FreshnessModel(toy_model_path)
    bright = Image.new("RGB", (64, 64), (255, 255, 255))
    dark = Image.new("RGB", (64, 64), (5, 5, 5))
    assert model.label(bright) == "fresh"
    assert model.label(dark) == "spoiled"


def test_metrics_carried_through(toy_model_path):
    model = FreshnessModel(toy_model_path)
    assert model.metrics["test_accuracy"] == 0.9


def test_missing_model_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        FreshnessModel(str(tmp_path / "nope.json"))
