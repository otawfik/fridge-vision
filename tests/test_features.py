import numpy as np
from PIL import Image

from src.features import FEATURE_DIM, extract_features


def _solid(color, size=(64, 64)):
    return Image.new("RGB", size, color)


def test_feature_dim_is_stable():
    feats = extract_features(_solid((200, 30, 30)))
    assert feats.shape == (FEATURE_DIM,)
    assert feats.dtype == np.float32


def test_features_are_deterministic():
    img = _solid((120, 180, 60))
    a = extract_features(img)
    b = extract_features(img)
    np.testing.assert_array_equal(a, b)


def test_red_vs_green_histograms_differ():
    red = extract_features(_solid((220, 20, 20)))
    green = extract_features(_solid((20, 200, 20)))
    assert not np.allclose(red, green)


def test_brown_pixel_ratio_detects_browning():
    fresh_green = extract_features(_solid((60, 180, 60)))
    brown = extract_features(_solid((150, 100, 40)))
    # brown stat is index -6 in the stats block
    assert brown[-6] > fresh_green[-6]


def test_handles_tiny_images():
    feats = extract_features(_solid((100, 100, 100), size=(8, 8)))
    assert feats.shape == (FEATURE_DIM,)
    assert np.all(np.isfinite(feats))
