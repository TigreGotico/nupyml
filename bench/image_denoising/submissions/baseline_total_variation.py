"""Baseline: total-variation (ROF) denoising per image."""
import numpy as np

from nupyml.image import total_variation_denoise


def solve(X_train, y_train, X_test):
    return np.array([total_variation_denoise(img, weight=0.12, n_iter=100)
                     for img in X_test])
