"""Kernel ridge regression: ridge in a feature space you never build.

THE TWO-LINE DERIVATION
-----------------------
Ridge regression fits ``w`` minimising ``||y - Xw||^2 + alpha ||w||^2``. To make
it non-linear, map the inputs through some ``phi(x)`` into a richer space and do
ridge there. The problem: ``phi(x)`` might be huge or infinite-dimensional.

The kernel trick removes it. The ridge solution can be written so that ``phi``
appears ONLY inside inner products ``phi(x_i).phi(x_j)`` -- and a kernel computes
those directly, ``K(x_i, x_j) = phi(x_i).phi(x_j)``, without ever forming
``phi``. The prediction becomes::

    f(x) = sum_i alpha_i K(x, x_i),   alpha = (K + lambda I)^{-1} y

An RBF kernel corresponds to an INFINITE-dimensional ``phi``, and this fits it in
closed form. That is the whole magic: infinite features, one linear solve.

KRR vs SVR: THE SAME SPACE, A DIFFERENT LOSS
--------------------------------------------
Kernel ridge and support vector regression live in the same kernel space and
differ only in their loss, and the difference decides everything:

* **KRR** uses squared error. Every training point gets a non-zero ``alpha_i``,
  so prediction sums over the ENTIRE training set -- it is a DENSE model. One
  linear solve to fit; slow and memory-hungry to predict.
* **SVR** uses epsilon-insensitive loss, which ignores errors inside a tube. Most
  points land in the tube and drop out, leaving a SPARSE model -- prediction sums
  over support vectors only. Harder to fit (a QP), cheaper to predict.

So the choice is not about accuracy, which is usually close. It is dense-and-
simple against sparse-and-lean. On a small dataset KRR's closed form is the
easier win; when prediction must be fast, SVR's sparsity earns its harder fit.

THE COST
--------
``K`` is ``n x n``, and the solve is ``O(n^3)``. KRR is superb up to a few
thousand points and impossible beyond -- the kernel matrix alone is 80 GB at
100k points. That ceiling, not accuracy, is what rules it out at scale, and it is
why kernel methods lost to neural networks on large data.
"""
import numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy.spatial.distance import cdist

from ..base import BaseEstimator, RegressorMixin, check_is_fitted
from ..utils import check_X_y, check_array


def _kernel(X, Y, kernel, gamma, degree, coef0):
    if kernel == "linear":
        return X @ Y.T
    if kernel == "rbf":
        # ||x - y||^2 via cdist, then exp -- an infinite-dimensional phi,
        # evaluated without ever constructing it
        return np.exp(-gamma * cdist(X, Y, "sqeuclidean"))
    if kernel == "poly":
        return (gamma * (X @ Y.T) + coef0) ** degree
    if kernel == "laplacian":
        return np.exp(-gamma * cdist(X, Y, "cityblock"))
    if kernel == "sigmoid":
        return np.tanh(gamma * (X @ Y.T) + coef0)
    raise ValueError(f"Unknown kernel: {kernel!r}")


class KernelRidge(BaseEstimator, RegressorMixin):
    """Ridge regression in a kernel-induced feature space.

    ``alpha`` is the ridge penalty (called ``lambda`` in the derivation above);
    ``gamma`` is the RBF width. Both matter, but ``gamma`` matters more -- it sets
    how quickly similarity decays with distance, and so how wiggly the fit is.
    Too large and every point is an island (the model interpolates noise); too
    small and everything is similar to everything (the model is nearly constant).
    """

    def __init__(self, alpha=1.0, kernel="rbf", gamma=1.0, degree=3, coef0=1.0):
        self.alpha = alpha
        self.kernel = kernel
        self.gamma = gamma
        self.degree = degree
        self.coef0 = coef0

    def fit(self, X, y):
        X, y = check_X_y(X, y, y_numeric=True)
        self.X_fit_ = X
        K = _kernel(X, X, self.kernel, self.gamma, self.degree, self.coef0)

        # (K + alpha I) alpha_ = y, solved by Cholesky. K + alpha I is symmetric
        # positive definite for alpha > 0, so Cholesky is both the fastest solve
        # and the numerically safest -- and the alpha on the diagonal is exactly
        # what guarantees the matrix is invertible at all
        A = K + self.alpha * np.eye(len(X))
        self.dual_coef_ = cho_solve(cho_factor(A, lower=True), y)
        return self

    def predict(self, X):
        check_is_fitted(self, "dual_coef_")
        X = check_array(X)
        K = _kernel(X, self.X_fit_, self.kernel, self.gamma, self.degree,
                    self.coef0)
        # every training point contributes -- a DENSE model, unlike SVR
        return K @ self.dual_coef_


__all__ = ["KernelRidge"]
