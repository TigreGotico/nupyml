"""Explicit feature maps that approximate kernels, so linear models scale."""
import numpy as np
import scipy.linalg
from scipy.spatial.distance import cdist

from ..base import BaseEstimator, TransformerMixin, check_is_fitted
from ..utils import check_array, check_random_state


class RBFSampler(BaseEstimator, TransformerMixin):
    """Random Fourier features for the RBF kernel (Rahimi & Recht 2007)."""

    def __init__(self, gamma=1.0, n_components=100, random_state=None):
        self.gamma = gamma
        self.n_components = n_components
        self.random_state = random_state

    def fit(self, X, y=None):
        X = check_array(X)
        rng = check_random_state(self.random_state)
        d = X.shape[1]
        # w ~ N(0, 2*gamma) is the Fourier transform of the RBF kernel
        self.random_weights_ = np.sqrt(2 * self.gamma) * rng.normal(size=(d, self.n_components))
        self.random_offset_ = rng.uniform(0, 2 * np.pi, size=self.n_components)
        self.n_features_in_ = d
        self.n_features_out_ = self.n_components
        return self

    def transform(self, X):
        check_is_fitted(self, "random_weights_")
        X = check_array(X)
        proj = X @ self.random_weights_ + self.random_offset_
        return np.sqrt(2.0 / self.n_components) * np.cos(proj)


class SkewedChi2Sampler(BaseEstimator, TransformerMixin):
    """Random features for the skewed chi-squared kernel."""

    def __init__(self, skewedness=1.0, n_components=100, random_state=None):
        self.skewedness = skewedness
        self.n_components = n_components
        self.random_state = random_state

    def fit(self, X, y=None):
        X = check_array(X)
        rng = check_random_state(self.random_state)
        d = X.shape[1]
        uniform = rng.uniform(size=(d, self.n_components))
        # inverse CDF of the sech distribution
        self.random_weights_ = np.log(np.tan(np.pi * uniform / 2.0))
        self.random_offset_ = rng.uniform(0, 2 * np.pi, size=self.n_components)
        return self

    def transform(self, X):
        check_is_fitted(self, "random_weights_")
        X = check_array(X)
        if (X < -self.skewedness).any():
            raise ValueError("X must be >= -skewedness")
        proj = np.log(X + self.skewedness) @ self.random_weights_ \
            + self.random_offset_
        return np.sqrt(2.0 / self.n_components) * np.cos(proj)


class Nystroem(BaseEstimator, TransformerMixin):
    """Low-rank kernel approximation from a random subset of training points."""

    def __init__(self, kernel="rbf", gamma=None, degree=3, coef0=1.0,
                 n_components=100, random_state=None):
        self.kernel = kernel
        self.gamma = gamma
        self.degree = degree
        self.coef0 = coef0
        self.n_components = n_components
        self.random_state = random_state

    def _kernel(self, A, B):
        gamma = self.gamma if self.gamma is not None else 1.0 / A.shape[1]
        if self.kernel == "rbf":
            return np.exp(-gamma * cdist(A, B, "sqeuclidean"))
        if self.kernel == "poly":
            return (gamma * (A @ B.T) + self.coef0) ** self.degree
        if self.kernel == "linear":
            return A @ B.T
        if self.kernel == "sigmoid":
            return np.tanh(gamma * (A @ B.T) + self.coef0)
        if callable(self.kernel):
            return self.kernel(A, B)
        raise ValueError(f"Unknown kernel: {self.kernel!r}")

    def fit(self, X, y=None):
        X = check_array(X)
        rng = check_random_state(self.random_state)
        n = len(X)
        k = min(self.n_components, n)
        idx = rng.permutation(n)[:k]
        basis = X[idx]
        K = self._kernel(basis, basis)
        # K^{-1/2} through an eigendecomposition, dropping null directions
        vals, vecs = scipy.linalg.eigh(K)
        vals = np.maximum(vals, 1e-12)
        self.normalization_ = vecs / np.sqrt(vals) @ vecs.T
        self.components_ = basis
        self.component_indices_ = idx
        self.n_features_in_ = X.shape[1]
        self.n_features_out_ = k
        return self

    def transform(self, X):
        check_is_fitted(self, "components_")
        X = check_array(X)
        return self._kernel(X, self.components_) @ self.normalization_


class AdditiveChi2Sampler(BaseEstimator, TransformerMixin):
    """Deterministic sampling of the additive chi-squared kernel map."""

    def __init__(self, sample_steps=2, sample_interval=None):
        self.sample_steps = sample_steps
        self.sample_interval = sample_interval

    def fit(self, X, y=None):
        X = check_array(X)
        if (X < 0).any():
            raise ValueError("AdditiveChi2Sampler requires non-negative X")
        if self.sample_interval is None:
            self.sample_interval_ = {1: 0.8, 2: 0.5, 3: 0.4}.get(
                self.sample_steps, 0.4)
        else:
            self.sample_interval_ = self.sample_interval
        return self

    def transform(self, X):
        check_is_fitted(self, "sample_interval_")
        X = check_array(X)
        if (X < 0).any():
            raise ValueError("AdditiveChi2Sampler requires non-negative X")
        Xc = np.clip(X, 1e-12, None)
        blocks = [np.sqrt(Xc * self.sample_interval_)]
        log_x = np.log(Xc)
        for j in range(1, self.sample_steps):
            factor = j * self.sample_interval_
            common = np.sqrt(2 * Xc * self.sample_interval_
                             / np.cosh(np.pi * factor))
            blocks.append(common * np.cos(factor * log_x))
            blocks.append(common * np.sin(factor * log_x))
        return np.hstack(blocks)


__all__ = ["RBFSampler", "Nystroem", "SkewedChi2Sampler", "AdditiveChi2Sampler"]
