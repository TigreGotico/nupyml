"""Support vector machines: SMO-trained SVC/SVR, dual coordinate LinearSVC."""
import numpy as np
from scipy.spatial.distance import cdist

from ..base import BaseEstimator, ClassifierMixin, RegressorMixin, check_is_fitted
from ..preprocessing import LabelEncoder
from ..utils import check_X_y, check_array, check_random_state


def _kernel_fn(kernel, gamma, degree, coef0):
    if kernel == "linear":
        return lambda A, B: A @ B.T
    if kernel == "rbf":
        return lambda A, B: np.exp(-gamma * cdist(A, B, "sqeuclidean"))
    if kernel == "poly":
        return lambda A, B: (gamma * (A @ B.T) + coef0) ** degree
    if kernel == "sigmoid":
        return lambda A, B: np.tanh(gamma * (A @ B.T) + coef0)
    raise ValueError(f"Unknown kernel: {kernel!r}")


def _smo(K, y, C, tol=1e-3, max_passes=10, max_iter=10000, rng=None):
    """Simplified SMO (Platt) for binary SVM. y in {-1, +1}. Returns alpha, b."""
    rng = rng or np.random
    n = len(y)
    alpha = np.zeros(n)
    b = 0.0
    passes = 0
    it = 0
    # cached decision errors
    def f(i):
        return (alpha * y) @ K[:, i] + b

    while passes < max_passes and it < max_iter:
        num_changed = 0
        for i in range(n):
            Ei = f(i) - y[i]
            if (y[i] * Ei < -tol and alpha[i] < C) or (y[i] * Ei > tol and alpha[i] > 0):
                j = rng.randint(n - 1)
                if j >= i:
                    j += 1
                Ej = f(j) - y[j]
                ai_old, aj_old = alpha[i], alpha[j]
                if y[i] != y[j]:
                    L, H = max(0, aj_old - ai_old), min(C, C + aj_old - ai_old)
                else:
                    L, H = max(0, ai_old + aj_old - C), min(C, ai_old + aj_old)
                if L == H:
                    continue
                eta = 2 * K[i, j] - K[i, i] - K[j, j]
                if eta >= 0:
                    continue
                aj = np.clip(aj_old - y[j] * (Ei - Ej) / eta, L, H)
                if abs(aj - aj_old) < 1e-5:
                    continue
                ai = ai_old + y[i] * y[j] * (aj_old - aj)
                b1 = b - Ei - y[i] * (ai - ai_old) * K[i, i] \
                    - y[j] * (aj - aj_old) * K[i, j]
                b2 = b - Ej - y[i] * (ai - ai_old) * K[i, j] \
                    - y[j] * (aj - aj_old) * K[j, j]
                alpha[i], alpha[j] = ai, aj
                if 0 < ai < C:
                    b = b1
                elif 0 < aj < C:
                    b = b2
                else:
                    b = (b1 + b2) / 2
                num_changed += 1
        passes = passes + 1 if num_changed == 0 else 0
        it += 1
    return alpha, b


class SVC(BaseEstimator, ClassifierMixin):
    """Kernel SVM classifier (one-vs-one for multiclass)."""

    def __init__(self, C=1.0, kernel="rbf", gamma="scale", degree=3, coef0=0.0,
                 tol=1e-3, max_iter=10000, random_state=None):
        self.C = C
        self.kernel = kernel
        self.gamma = gamma
        self.degree = degree
        self.coef0 = coef0
        self.tol = tol
        self.max_iter = max_iter
        self.random_state = random_state

    def _gamma_value(self, X):
        if self.gamma == "scale":
            return 1.0 / (X.shape[1] * X.var())
        if self.gamma == "auto":
            return 1.0 / X.shape[1]
        return float(self.gamma)

    def fit(self, X, y):
        X, y = check_X_y(X, y)
        rng = check_random_state(self.random_state)
        self._le = LabelEncoder().fit(y)
        self.classes_ = self._le.classes_
        y_idx = self._le.transform(y)
        self._gamma = self._gamma_value(X)
        self._kfn = _kernel_fn(self.kernel, self._gamma, self.degree, self.coef0)
        k = len(self.classes_)
        self._models = {}
        for a in range(k):
            for bcls in range(a + 1, k):
                mask = (y_idx == a) | (y_idx == bcls)
                Xa = X[mask]
                ya = np.where(y_idx[mask] == bcls, 1.0, -1.0)
                K = self._kfn(Xa, Xa)
                alpha, b = _smo(K, ya, self.C, tol=self.tol,
                                max_iter=self.max_iter, rng=rng)
                sv = alpha > 1e-8
                self._models[(a, bcls)] = (Xa[sv], ya[sv] * alpha[sv], b)
        self.support_vectors_ = np.vstack([m[0] for m in self._models.values()]) \
            if self._models else np.empty((0, X.shape[1]))
        return self

    def _binary_decision(self, model, X):
        sv, coef, b = model
        if len(sv) == 0:
            return np.full(len(X), b)
        return self._kfn(X, sv) @ coef + b

    def decision_function(self, X):
        check_is_fitted(self, "_models")
        X = check_array(X)
        k = len(self.classes_)
        if k == 2:
            return self._binary_decision(self._models[(0, 1)], X)
        votes = np.zeros((len(X), k))
        for (a, bcls), model in self._models.items():
            d = self._binary_decision(model, X)
            votes[:, bcls] += d > 0
            votes[:, a] += d <= 0
        return votes

    def predict(self, X):
        scores = self.decision_function(X)
        if scores.ndim == 1:
            return self.classes_[(scores > 0).astype(int)]
        return self.classes_[np.argmax(scores, axis=1)]


class SVR(BaseEstimator, RegressorMixin):
    """Epsilon-SVR solved as a box-constrained QP via L-BFGS-B on the dual."""

    def __init__(self, C=1.0, epsilon=0.1, kernel="rbf", gamma="scale",
                 degree=3, coef0=0.0, max_iter=500):
        self.C = C
        self.epsilon = epsilon
        self.kernel = kernel
        self.gamma = gamma
        self.degree = degree
        self.coef0 = coef0
        self.max_iter = max_iter

    def fit(self, X, y):
        import scipy.optimize
        X, y = check_X_y(X, y, y_numeric=True)
        if self.gamma == "scale":
            g = 1.0 / (X.shape[1] * X.var())
        elif self.gamma == "auto":
            g = 1.0 / X.shape[1]
        else:
            g = float(self.gamma)
        self._kfn = _kernel_fn(self.kernel, g, self.degree, self.coef0)
        K = self._kfn(X, X)
        n = len(y)

        # dual variables beta = alpha - alpha*; |beta| <= C
        # objective: 0.5 b'Kb - y'b + eps*|b|_1  (smooth approx on |b|)
        def obj(beta):
            Kb = K @ beta
            val = 0.5 * beta @ Kb - y @ beta + self.epsilon * np.abs(beta).sum()
            grad = Kb - y + self.epsilon * np.sign(beta)
            return val, grad

        res = scipy.optimize.minimize(
            obj, np.zeros(n), jac=True, method="L-BFGS-B",
            bounds=[(-self.C, self.C)] * n,
            options={"maxiter": self.max_iter})
        beta = res.x
        sv = np.abs(beta) > 1e-8
        self._sv_X = X[sv]
        self._sv_coef = beta[sv]
        # intercept from margin points
        margin = (np.abs(beta) > 1e-8) & (np.abs(beta) < self.C - 1e-8)
        if margin.any():
            pred_no_b = K[margin][:, sv] @ self._sv_coef
            self.intercept_ = float(np.mean(
                y[margin] - pred_no_b - self.epsilon * np.sign(beta[margin])))
        else:
            self.intercept_ = float(np.mean(y - K[:, sv] @ self._sv_coef))
        return self

    def predict(self, X):
        check_is_fitted(self, "_sv_X")
        X = check_array(X)
        return self._kfn(X, self._sv_X) @ self._sv_coef + self.intercept_


class LinearSVC(BaseEstimator, ClassifierMixin):
    """Linear SVM via dual coordinate descent (Hsieh et al. 2008)."""

    def __init__(self, C=1.0, max_iter=1000, tol=1e-4, random_state=None):
        self.C = C
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state

    def _fit_binary(self, X, t, rng):
        n, d = X.shape
        alpha = np.zeros(n)
        w = np.zeros(d)
        Q_diag = (X ** 2).sum(axis=1) + 1.0  # +1 for bias via augmentation
        Xb = np.hstack([X, np.ones((n, 1))])
        w = np.zeros(d + 1)
        for _ in range(self.max_iter):
            max_pg = 0.0
            for i in rng.permutation(n):
                G = t[i] * (Xb[i] @ w) - 1.0
                pg = G
                if alpha[i] == 0:
                    pg = min(G, 0.0)
                elif alpha[i] == self.C:
                    pg = max(G, 0.0)
                if abs(pg) > 1e-12:
                    old = alpha[i]
                    alpha[i] = np.clip(old - G / Q_diag[i], 0, self.C)
                    w += (alpha[i] - old) * t[i] * Xb[i]
                    max_pg = max(max_pg, abs(pg))
            if max_pg < self.tol:
                break
        return w[:-1], w[-1]

    def fit(self, X, y):
        X, y = check_X_y(X, y)
        rng = check_random_state(self.random_state)
        self._le = LabelEncoder().fit(y)
        self.classes_ = self._le.classes_
        y_idx = self._le.transform(y)
        k = len(self.classes_)
        if k == 2:
            t = np.where(y_idx == 1, 1.0, -1.0)
            w, b = self._fit_binary(X, t, rng)
            self.coef_ = w[None, :]
            self.intercept_ = np.array([b])
        else:  # one-vs-rest
            self.coef_ = np.zeros((k, X.shape[1]))
            self.intercept_ = np.zeros(k)
            for c in range(k):
                t = np.where(y_idx == c, 1.0, -1.0)
                self.coef_[c], self.intercept_[c] = self._fit_binary(X, t, rng)
        return self

    def decision_function(self, X):
        check_is_fitted(self, "coef_")
        scores = check_array(X) @ self.coef_.T + self.intercept_
        return scores.ravel() if scores.shape[1] == 1 else scores

    def predict(self, X):
        scores = self.decision_function(X)
        if scores.ndim == 1:
            return self.classes_[(scores > 0).astype(int)]
        return self.classes_[np.argmax(scores, axis=1)]


__all__ = ["SVC", "SVR", "LinearSVC"]
