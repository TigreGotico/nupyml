"""Baseline: bilateral filtering per image."""
import numpy as np

from nupyml.image import bilateral_filter


def solve(X_train, y_train, X_test):
    return np.array([bilateral_filter(img, spatial_sigma=2.0, range_sigma=0.3, radius=3)
                     for img in X_test])
