"""Histogram-of-oriented-gradients features + RBF SVC.

HOG captures the edge-orientation structure of each digit, invariant to small
shifts and contrast -- then a standard classifier finishes the job.
"""
import numpy as np

from nupyml.image import histogram_of_oriented_gradients
from nupyml.pipeline import make_pipeline
from nupyml.preprocessing import StandardScaler
from nupyml.svm import SVC


def _feats(images):
    return np.array([histogram_of_oriented_gradients(
        im, orientations=8, pixels_per_cell=(4, 4), cells_per_block=(2, 2))
        for im in images])


def solve(X_train, y_train, X_test):
    Ftr, Fte = _feats(X_train), _feats(X_test)
    model = make_pipeline(StandardScaler(), SVC(C=10, gamma="scale"))
    model.fit(Ftr, y_train)
    return model.predict(Fte)
