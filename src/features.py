"""Handcrafted image features for the freshness classifier.

Spoilage shows up as color change (browning, dullness) and texture change
(soft spots, wrinkles), so we featurize both: HSV color histograms plus
colorfulness stats in HSV space, and a local binary pattern (LBP) texture
histogram. Everything is numpy/Pillow only, no extra dependencies.
"""
import numpy as np
from PIL import Image

FEATURE_DIM = 12 + 8 + 8 + 10 + 6  # h_hist + s_hist + v_hist + lbp + stats


def _rgb_to_hsv(arr):
    """Vectorized RGB->HSV for a float array in [0, 1]. Returns H, S, V in [0, 1]."""
    r, g, b = arr[..., 0], arr[..., 1], arr[..., 2]
    cmax = arr.max(axis=-1)
    cmin = arr.min(axis=-1)
    delta = cmax - cmin
    h = np.zeros_like(cmax)
    mask = delta > 1e-9
    r_eq = mask & (cmax == r)
    g_eq = mask & (cmax == g)
    b_eq = mask & (cmax == b)
    h[r_eq] = ((g[r_eq] - b[r_eq]) / delta[r_eq]) % 6.0
    h[g_eq] = (b[g_eq] - r[g_eq]) / delta[g_eq] + 2.0
    h[b_eq] = (r[b_eq] - g[b_eq]) / delta[b_eq] + 4.0
    h = h / 6.0
    s = np.where(cmax > 1e-9, delta / np.maximum(cmax, 1e-9), 0.0)
    return h, s, cmax


def _lbp_histogram(gray, n_bins=10):
    """Uniform-ish LBP texture histogram computed with numpy (3x3 neighborhood)."""
    g = gray.astype(np.int16)
    c = g[1:-1, 1:-1]
    code = np.zeros_like(c, dtype=np.int32)
    code |= (g[:-2, :-2] >= c).astype(np.int32) << 7
    code |= (g[:-2, 1:-1] >= c).astype(np.int32) << 6
    code |= (g[:-2, 2:] >= c).astype(np.int32) << 5
    code |= (g[1:-1, 2:] >= c).astype(np.int32) << 4
    code |= (g[2:, 2:] >= c).astype(np.int32) << 3
    code |= (g[2:, 1:-1] >= c).astype(np.int32) << 2
    code |= (g[2:, :-2] >= c).astype(np.int32) << 1
    code |= (g[1:-1, :-2] >= c).astype(np.int32)
    hist, _ = np.histogram(code, bins=n_bins, range=(0, 256))
    total = hist.sum()
    return (hist / total).astype(np.float32) if total else np.zeros(n_bins, dtype=np.float32)


def _hist(x, bins):
    hist, _ = np.histogram(x, bins=bins, range=(0.0, 1.0))
    total = hist.sum()
    return (hist / total).astype(np.float32) if total else np.zeros(bins, dtype=np.float32)


def extract_features(pil_image):
    """Extract the fixed-length freshness feature vector from a PIL image.

    Output: np.float32 array of shape (FEATURE_DIM,), roughly in [0, 1].
    """
    img = pil_image.convert("RGB").resize((128, 128))
    arr = np.asarray(img).astype(np.float32) / 255.0
    h, s, v = _rgb_to_hsv(arr)

    h_hist = _hist(h, 12)
    s_hist = _hist(s, 8)
    v_hist = _hist(v, 8)

    gray = np.asarray(img.convert("L"))
    lbp = _lbp_histogram(gray)

    # Browning: hue in the orange-brown band, decently saturated, not too bright.
    brown = ((h > 0.03) & (h < 0.11) & (s > 0.25) & (v < 0.75)).mean()
    # Dull/dark: low value or very low saturation (gray, lifeless areas).
    dull = ((v < 0.25) | (s < 0.08)).mean()
    stats = np.array([
        float(brown),
        float(dull),
        float(s.mean()),
        float(v.mean()),
        float(v.std()),
        float(s.std()),
    ], dtype=np.float32)

    return np.concatenate([h_hist, s_hist, v_hist, lbp, stats]).astype(np.float32)
