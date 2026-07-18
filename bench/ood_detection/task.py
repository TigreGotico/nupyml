"""Out-of-distribution detection: flag samples from a DIFFERENT distribution.

The in-distribution data lies on a low-dimensional manifold (a rotated ellipsoid);
the OOD samples come from a broader, differently-shaped distribution. The
submission sees only X and returns an OOD score per point (higher = more OOD).
Scored by ROC-AUC against the hidden in/out labels -- so no threshold is needed.
"""
import numpy as np

from nupyml.metrics import roc_auc_score

KIND = "clustering"
GOAL = "Score points by how out-of-distribution they are; ROC-AUC."
METRIC = "roc_auc"
HIGHER_IS_BETTER = True
MIN_SCORE = 0.85


def load():
    rng = np.random.RandomState(0)
    d = 6
    # in-distribution: a correlated (rotated, anisotropic) Gaussian
    A = rng.randn(d, d)
    cov = A @ A.T
    inl = rng.multivariate_normal(np.zeros(d), cov, 950)
    # OOD: isotropic and wider, off the in-distribution manifold
    out = rng.randn(50, d) * 4.0 + 2.0
    X = np.vstack([inl, out])
    y = np.r_[np.zeros(950), np.ones(50)].astype(int)
    perm = rng.permutation(len(y))
    return X[perm], y[perm]


def metric(y_true, y_pred):
    return roc_auc_score(y_true, y_pred)
