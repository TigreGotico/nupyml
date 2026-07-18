"""HOG + LBP-histogram descriptors + random forest -- position-robust features."""
import numpy as np

from nupyml.image import histogram_of_oriented_gradients, local_binary_pattern
from nupyml.ensemble import RandomForestClassifier


def _feats(images):
    out = []
    for im in images:
        h = histogram_of_oriented_gradients(im, orientations=6,
                                            pixels_per_cell=(5, 5),
                                            cells_per_block=(2, 2))
        lbp = local_binary_pattern(im, n_points=8, radius=1)
        hist, _ = np.histogram(lbp, bins=16, range=(0, 256), density=True)
        out.append(np.concatenate([h, hist]))
    return np.array(out)


def solve(X_train, y_train, X_test):
    Ftr, Fte = _feats(X_train), _feats(X_test)
    return RandomForestClassifier(n_estimators=150, random_state=0).fit(
        Ftr, y_train).predict(Fte)
