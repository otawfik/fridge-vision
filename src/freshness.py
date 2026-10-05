"""Freshness scoring: fresh produce vs spoiled/rotten.

The model is a logistic regression on the handcrafted color/texture features
from src/features.py, trained by scripts/train_freshness.py on real
fresh-vs-rotten produce photos. Weights ship as plain JSON so the demo has no
binary blobs and no heavyweight ML framework at inference time.

score(crop) -> P(fresh) in [0, 1]. Higher means fresher.
"""
import json
import os

import numpy as np

from .features import extract_features

DEFAULT_MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "freshness_model.json")


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


class FreshnessModel:
    def __init__(self, model_path=None):
        path = model_path or DEFAULT_MODEL_PATH
        with open(path) as f:
            blob = json.load(f)
        self.mean = np.array(blob["scaler_mean"], dtype=np.float64)
        self.scale = np.array(blob["scaler_scale"], dtype=np.float64)
        self.coef = np.array(blob["coef"], dtype=np.float64)
        self.intercept = float(blob["intercept"])
        self.metrics = blob.get("metrics", {})

    def score(self, pil_crop):
        """Probability the cropped item photo shows fresh (not spoiled) food."""
        feats = extract_features(pil_crop).astype(np.float64)
        z = (feats - self.mean) / self.scale
        logit = float(z @ self.coef + self.intercept)
        return float(_sigmoid(logit))

    def label(self, pil_crop, threshold=0.5):
        return "fresh" if self.score(pil_crop) >= threshold else "spoiled"


_model = None


def get_model():
    global _model
    if _model is None:
        _model = FreshnessModel()
    return _model
