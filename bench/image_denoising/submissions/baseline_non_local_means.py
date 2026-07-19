"""Baseline: non-local means denoising per image."""
import numpy as np

from nupyml.image import non_local_means


def solve(X_train, y_train, X_test):
    return np.array([non_local_means(img, patch_radius=2, search_radius=5, h=0.25)
                     for img in X_test])
