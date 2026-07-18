"""Kernel density estimation -- non-parametric, no assumed number of modes."""
from nupyml.neighbors import KernelDensity


def solve(X_train, X_test):
    kde = KernelDensity(bandwidth=1.0).fit(X_train)
    return kde.score_samples(X_test)
