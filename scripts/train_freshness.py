"""Train the freshness classifier and ship it as JSON.

Reads data/freshness_sample/{fresh,rotten}/*.jpg (see download_data.py),
extracts handcrafted color/texture features, and trains a logistic regression.
A HistGradientBoosting model is trained alongside for comparison. The logistic
regression ships to models/freshness_model.json (scaler stats + coefficients),
so inference needs nothing but numpy.

Usage: python scripts/train_freshness.py [--data data/freshness_sample]
"""
import argparse
import glob
import json
import os
import sys

import numpy as np
from PIL import Image
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, f1_score)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.features import extract_features  # noqa: E402


def load_samples(data_dir):
    X, y = [], []
    for label, cls in (("fresh", 1), ("rotten", 0)):
        paths = sorted(glob.glob(os.path.join(data_dir, label, "*.jpg")))
        print(f"{label}: {len(paths)} images")
        for p in paths:
            try:
                X.append(extract_features(Image.open(p)))
                y.append(cls)
            except Exception as e:  # corrupt image, skip
                print(f"  skip {p}: {e}")
    return np.array(X, dtype=np.float32), np.array(y)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/freshness_sample")
    ap.add_argument("--out", default="models/freshness_model.json")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    X, y = load_samples(args.data)
    print(f"feature dim: {X.shape[1]}, samples: {len(y)}")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=args.seed, stratify=y
    )

    scaler = StandardScaler().fit(X_train)
    Z_train, Z_test = scaler.transform(X_train), scaler.transform(X_test)

    lr = LogisticRegression(max_iter=2000, class_weight="balanced", C=1.0)
    lr.fit(Z_train, y_train)
    hgb = HistGradientBoostingClassifier(random_state=args.seed)
    hgb.fit(Z_train, y_train)

    for name, model in (("logreg", lr), ("histgb", hgb)):
        pred = model.predict(Z_test)
        print(f"\n=== {name} ===")
        print(f"accuracy={accuracy_score(y_test, pred):.3f} "
              f"f1={f1_score(y_test, pred):.3f}")
        print(classification_report(y_test, pred, target_names=["rotten", "fresh"]))
        print("confusion matrix [rotten, fresh]:")
        print(confusion_matrix(y_test, pred))

    # Ship the logistic regression: tiny, fast, JSON-serializable.
    proba = lr.predict_proba(Z_test)[:, 1]
    blob = {
        "scaler_mean": scaler.mean_.tolist(),
        "scaler_scale": scaler.scale_.tolist(),
        "coef": lr.coef_[0].tolist(),
        "intercept": float(lr.intercept_[0]),
        "metrics": {
            "test_accuracy": float(accuracy_score(y_test, lr.predict(Z_test))),
            "test_f1": float(f1_score(y_test, lr.predict(Z_test))),
            "n_test": int(len(y_test)),
            "n_train": int(len(y_train)),
            "data": "Project-AgML/fresh_rotten_fruit_classification (streamed sample)",
        },
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(blob, f)
    print(f"\nsaved {args.out}")


if __name__ == "__main__":
    main()
