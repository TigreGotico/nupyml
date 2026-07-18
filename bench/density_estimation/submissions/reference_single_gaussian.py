"""A single multivariate Gaussian -- the unimodal floor (misses the structure)."""
import numpy as np


def solve(X_train, X_test):
    mu = X_train.mean(0)
    cov = np.cov(X_train, rowvar=False) + 1e-6 * np.eye(X_train.shape[1])
    d = X_train.shape[1]
    inv = np.linalg.inv(cov)
    sign, logdet = np.linalg.slogdet(cov)
    diff = X_test - mu
    maha = np.einsum("ij,jk,ik->i", diff, inv, diff)
    return -0.5 * (d * np.log(2 * np.pi) + logdet + maha)
