"""Linear models: OLS, Ridge, Lasso, ElasticNet, Logistic, SGD, Perceptron."""
import numpy as np
import scipy.optimize
import scipy.sparse as sp

from ..base import BaseEstimator, ClassifierMixin, RegressorMixin, check_is_fitted
from ..preprocessing import LabelEncoder
from ..utils import check_X_y, check_array, check_random_state, softmax, sigmoid


def _add_intercept_stats(X, y):
    X_mean = np.asarray(X.mean(axis=0)).ravel()
    y_mean = y.mean(axis=0)
    return X_mean, y_mean


class LinearRegression(BaseEstimator, RegressorMixin):
    def __init__(self, fit_intercept=True):
        self.fit_intercept = fit_intercept

    def fit(self, X, y, sample_weight=None):
        X, y = check_X_y(X, y, y_numeric=True)
        if self.fit_intercept:
            if sample_weight is not None:
                w = np.asarray(sample_weight, dtype=np.float64)
                X_mean = np.average(X, axis=0, weights=w)
                y_mean = np.average(y, weights=w)
            else:
                X_mean, y_mean = _add_intercept_stats(X, y)
            Xc, yc = X - X_mean, y - y_mean
        else:
            Xc, yc = X, y
        if sample_weight is not None:
            sw = np.sqrt(np.asarray(sample_weight, dtype=np.float64))
            Xc, yc = Xc * sw[:, None], yc * sw
        coef, *_ = np.linalg.lstsq(Xc, yc, rcond=None)
        self.coef_ = coef
        self.intercept_ = (y_mean - X_mean @ coef) if self.fit_intercept else 0.0
        return self

    def predict(self, X):
        check_is_fitted(self, "coef_")
        return check_array(X) @ self.coef_ + self.intercept_


class Ridge(BaseEstimator, RegressorMixin):
    def __init__(self, alpha=1.0, fit_intercept=True):
        self.alpha = alpha
        self.fit_intercept = fit_intercept

    def fit(self, X, y, sample_weight=None):
        X, y = check_X_y(X, y, y_numeric=True)
        if self.fit_intercept:
            if sample_weight is not None:
                w = np.asarray(sample_weight, dtype=np.float64)
                X_mean = np.average(X, axis=0, weights=w)
                y_mean = np.average(y, weights=w)
            else:
                X_mean, y_mean = _add_intercept_stats(X, y)
            Xc, yc = X - X_mean, y - y_mean
        else:
            Xc, yc = X, y
        if sample_weight is not None:
            sw = np.sqrt(np.asarray(sample_weight, dtype=np.float64))
            Xc, yc = Xc * sw[:, None], yc * sw
        n_features = X.shape[1]
        A = Xc.T @ Xc + self.alpha * np.eye(n_features)
        self.coef_ = np.linalg.solve(A, Xc.T @ yc)
        self.intercept_ = (y_mean - X_mean @ self.coef_) if self.fit_intercept else 0.0
        return self

    def predict(self, X):
        check_is_fitted(self, "coef_")
        return check_array(X) @ self.coef_ + self.intercept_


class _CoordinateDescent(BaseEstimator, RegressorMixin):
    """Shared cyclic coordinate descent for Lasso / ElasticNet."""

    def __init__(self, alpha=1.0, l1_ratio=1.0, fit_intercept=True,
                 max_iter=1000, tol=1e-4):
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        self.fit_intercept = fit_intercept
        self.max_iter = max_iter
        self.tol = tol

    def fit(self, X, y):
        X, y = check_X_y(X, y, y_numeric=True)
        n, d = X.shape
        if self.fit_intercept:
            X_mean, y_mean = _add_intercept_stats(X, y)
            Xc, yc = X - X_mean, y - y_mean
        else:
            Xc, yc = X, y
        l1 = self.alpha * self.l1_ratio * n
        l2 = self.alpha * (1 - self.l1_ratio) * n
        w = np.zeros(d)
        col_sq = (Xc ** 2).sum(axis=0)
        resid = yc.copy()
        for it in range(self.max_iter):
            max_delta = 0.0
            for j in range(d):
                if col_sq[j] == 0.0:
                    continue
                w_j = w[j]
                rho = Xc[:, j] @ resid + col_sq[j] * w_j
                new_w = np.sign(rho) * max(abs(rho) - l1, 0.0) / (col_sq[j] + l2)
                if new_w != w_j:
                    resid += Xc[:, j] * (w_j - new_w)
                    w[j] = new_w
                    max_delta = max(max_delta, abs(new_w - w_j))
            if max_delta < self.tol:
                break
        self.coef_ = w
        self.intercept_ = (y_mean - X_mean @ w) if self.fit_intercept else 0.0
        self.n_iter_ = it + 1
        return self

    def predict(self, X):
        check_is_fitted(self, "coef_")
        return check_array(X) @ self.coef_ + self.intercept_


class Lasso(_CoordinateDescent):
    def __init__(self, alpha=1.0, fit_intercept=True, max_iter=1000, tol=1e-4):
        super().__init__(alpha=alpha, l1_ratio=1.0, fit_intercept=fit_intercept,
                         max_iter=max_iter, tol=tol)

    @classmethod
    def _get_param_names(cls):
        return ["alpha", "fit_intercept", "max_iter", "tol"]


class ElasticNet(_CoordinateDescent):
    def __init__(self, alpha=1.0, l1_ratio=0.5, fit_intercept=True,
                 max_iter=1000, tol=1e-4):
        super().__init__(alpha=alpha, l1_ratio=l1_ratio,
                         fit_intercept=fit_intercept, max_iter=max_iter, tol=tol)


class LogisticRegression(BaseEstimator, ClassifierMixin):
    """Multinomial logistic regression fit with L-BFGS and analytic gradients."""

    def __init__(self, C=1.0, fit_intercept=True, max_iter=200, tol=1e-6):
        self.C = C
        self.fit_intercept = fit_intercept
        self.max_iter = max_iter
        self.tol = tol

    def fit(self, X, y, sample_weight=None):
        X, y = check_X_y(X, y, accept_sparse=True)
        self._le = LabelEncoder().fit(y)
        self.classes_ = self._le.classes_
        y_idx = self._le.transform(y)
        n, d = X.shape
        k = len(self.classes_)
        n_out = 1 if k == 2 else k
        Y = None if k == 2 else np.eye(k)[y_idx]
        sw = np.ones(n) if sample_weight is None \
            else np.asarray(sample_weight, dtype=np.float64)

        def unpack(theta):
            W = theta[: d * n_out].reshape(d, n_out)
            b = theta[d * n_out:] if self.fit_intercept else np.zeros(n_out)
            return W, b

        def loss_grad(theta):
            W, b = unpack(theta)
            Z = X @ W + b
            if k == 2:
                z = Z.ravel()
                p = sigmoid(z)
                eps = 1e-15
                nll = -np.sum(sw * (y_idx * np.log(p + eps)
                                    + (1 - y_idx) * np.log(1 - p + eps)))
                dz = (sw * (p - y_idx))[:, None]
            else:
                P = softmax(Z, axis=1)
                nll = -np.sum(sw * np.log(P[np.arange(n), y_idx] + 1e-15))
                dz = (P - Y) * sw[:, None]
            reg = 0.5 / self.C * np.sum(W ** 2)
            gW = X.T @ dz + W / self.C
            gW = np.asarray(gW)
            grad = [gW.ravel()]
            if self.fit_intercept:
                grad.append(dz.sum(axis=0))
            return nll + reg, np.concatenate(grad)

        n_params = d * n_out + (n_out if self.fit_intercept else 0)
        res = scipy.optimize.minimize(
            loss_grad, np.zeros(n_params), jac=True, method="L-BFGS-B",
            options={"maxiter": self.max_iter, "gtol": self.tol})
        W, b = unpack(res.x)
        self.coef_ = W.T
        self.intercept_ = b
        self.n_iter_ = res.nit
        return self

    def decision_function(self, X):
        check_is_fitted(self, "coef_")
        X = check_array(X, accept_sparse=True)
        scores = X @ self.coef_.T + self.intercept_
        scores = np.asarray(scores)
        return scores.ravel() if scores.shape[1] == 1 else scores

    def predict_proba(self, X):
        scores = self.decision_function(X)
        if scores.ndim == 1:
            p = sigmoid(scores)
            return np.column_stack([1 - p, p])
        return softmax(scores, axis=1)

    def predict(self, X):
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]


class Perceptron(BaseEstimator, ClassifierMixin):
    def __init__(self, max_iter=1000, eta0=1.0, shuffle=True, random_state=None):
        self.max_iter = max_iter
        self.eta0 = eta0
        self.shuffle = shuffle
        self.random_state = random_state

    def fit(self, X, y):
        X, y = check_X_y(X, y)
        self._le = LabelEncoder().fit(y)
        self.classes_ = self._le.classes_
        if len(self.classes_) != 2:
            raise ValueError("Perceptron supports binary classification only")
        t = 2.0 * self._le.transform(y) - 1.0
        rng = check_random_state(self.random_state)
        n, d = X.shape
        w = np.zeros(d)
        b = 0.0
        for _ in range(self.max_iter):
            idx = rng.permutation(n) if self.shuffle else np.arange(n)
            errors = 0
            for i in idx:
                if t[i] * (X[i] @ w + b) <= 0:
                    w += self.eta0 * t[i] * X[i]
                    b += self.eta0 * t[i]
                    errors += 1
            if errors == 0:
                break
        self.coef_ = w
        self.intercept_ = b
        return self

    def decision_function(self, X):
        check_is_fitted(self, "coef_")
        return check_array(X) @ self.coef_ + self.intercept_

    def predict(self, X):
        return self.classes_[(self.decision_function(X) > 0).astype(int)]


class SGDClassifier(BaseEstimator, ClassifierMixin):
    """Binary/multiclass linear classifier trained with minibatch SGD.

    loss='log' gives logistic regression; loss='hinge' a linear SVM.
    """

    def __init__(self, loss="hinge", alpha=1e-4, max_iter=1000, tol=1e-3,
                 learning_rate=0.01, batch_size=32, random_state=None):
        self.loss = loss
        self.alpha = alpha
        self.max_iter = max_iter
        self.tol = tol
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.random_state = random_state

    def fit(self, X, y):
        X, y = check_X_y(X, y)
        self._le = LabelEncoder().fit(y)
        self.classes_ = self._le.classes_
        y_idx = self._le.transform(y)
        k = len(self.classes_)
        n, d = X.shape
        rng = check_random_state(self.random_state)
        n_out = 1 if k == 2 else k
        W = np.zeros((d, n_out))
        b = np.zeros(n_out)
        T = (2.0 * y_idx - 1.0)[:, None] if k == 2 else np.eye(k)[y_idx] * 2 - 1
        prev_loss = np.inf
        for epoch in range(self.max_iter):
            idx = rng.permutation(n)
            total = 0.0
            for start in range(0, n, self.batch_size):
                batch = idx[start:start + self.batch_size]
                Xb, Tb = X[batch], T[batch]
                Z = Xb @ W + b
                if self.loss == "hinge":
                    margin = 1 - Tb * Z
                    active = margin > 0
                    total += np.sum(np.maximum(margin, 0))
                    dZ = -Tb * active
                elif self.loss == "log":
                    total += np.sum(np.log1p(np.exp(-np.clip(Tb * Z, -500, 500))))
                    dZ = -Tb * sigmoid(-Tb * Z)
                else:
                    raise ValueError(f"Unknown loss: {self.loss!r}")
                gW = Xb.T @ dZ / len(batch) + self.alpha * W
                gb = dZ.mean(axis=0)
                W -= self.learning_rate * gW
                b -= self.learning_rate * gb
            epoch_loss = total / n
            if abs(prev_loss - epoch_loss) < self.tol * max(1.0, abs(prev_loss)):
                break
            prev_loss = epoch_loss
        self.coef_ = W.T
        self.intercept_ = b
        self.n_iter_ = epoch + 1
        return self

    def partial_fit(self, X, y, classes=None, sample_weight=None):
        """One SGD pass over the given chunk; keeps existing coefficients."""
        X, y = check_X_y(X, y)
        if not hasattr(self, "coef_"):
            if classes is None:
                raise ValueError("classes must be passed on the first call")
            self.classes_ = np.asarray(classes)
            self._le = LabelEncoder()
            self._le.classes_ = self.classes_
            k = len(self.classes_)
            n_out = 1 if k == 2 else k
            self.coef_ = np.zeros((n_out, X.shape[1]))
            self.intercept_ = np.zeros(n_out)
        y_idx = self._le.transform(y)
        k = len(self.classes_)
        W = self.coef_.T.copy()
        b = self.intercept_.copy()
        T = (2.0 * y_idx - 1.0)[:, None] if k == 2 else np.eye(k)[y_idx] * 2 - 1
        sw = np.ones(len(X)) if sample_weight is None \
            else np.asarray(sample_weight, dtype=np.float64)
        Z = X @ W + b
        if self.loss == "hinge":
            dZ = -T * ((1 - T * Z) > 0)
        else:
            dZ = -T * sigmoid(-T * Z)
        dZ = dZ * sw[:, None]
        W -= self.learning_rate * (X.T @ dZ / len(X) + self.alpha * W)
        b -= self.learning_rate * dZ.mean(axis=0)
        self.coef_ = W.T
        self.intercept_ = b
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


class SGDRegressor(BaseEstimator, RegressorMixin):
    def __init__(self, alpha=1e-4, max_iter=1000, tol=1e-4, learning_rate=0.01,
                 batch_size=32, random_state=None):
        self.alpha = alpha
        self.max_iter = max_iter
        self.tol = tol
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.random_state = random_state

    def fit(self, X, y):
        X, y = check_X_y(X, y, y_numeric=True)
        n, d = X.shape
        rng = check_random_state(self.random_state)
        w = np.zeros(d)
        b = 0.0
        prev_loss = np.inf
        for epoch in range(self.max_iter):
            idx = rng.permutation(n)
            total = 0.0
            for start in range(0, n, self.batch_size):
                batch = idx[start:start + self.batch_size]
                Xb, yb = X[batch], y[batch]
                err = Xb @ w + b - yb
                total += np.sum(err ** 2)
                w -= self.learning_rate * (Xb.T @ err / len(batch) + self.alpha * w)
                b -= self.learning_rate * err.mean()
            epoch_loss = total / n
            if abs(prev_loss - epoch_loss) < self.tol * max(1.0, abs(prev_loss)):
                break
            prev_loss = epoch_loss
        self.coef_ = w
        self.intercept_ = b
        self.n_iter_ = epoch + 1
        return self

    def partial_fit(self, X, y, sample_weight=None):
        X, y = check_X_y(X, y, y_numeric=True)
        if not hasattr(self, "coef_"):
            self.coef_ = np.zeros(X.shape[1])
            self.intercept_ = 0.0
        sw = np.ones(len(X)) if sample_weight is None \
            else np.asarray(sample_weight, dtype=np.float64)
        err = (X @ self.coef_ + self.intercept_ - y) * sw
        self.coef_ -= self.learning_rate * (X.T @ err / len(X)
                                            + self.alpha * self.coef_)
        self.intercept_ -= self.learning_rate * err.mean()
        return self

    def predict(self, X):
        check_is_fitted(self, "coef_")
        return check_array(X) @ self.coef_ + self.intercept_


from ._robust import (  # noqa: E402
    HuberRegressor, QuantileRegressor, TheilSenRegressor, RANSACRegressor,
)
from ._bayes_glm import (  # noqa: E402
    BayesianRidge, ARDRegression, PoissonRegressor, GammaRegressor,
    TweedieRegressor, RidgeCV, LassoCV, ElasticNetCV, LogisticRegressionCV,
    OrthogonalMatchingPursuit,
)

__all__ = [
    "LinearRegression", "Ridge", "Lasso", "ElasticNet", "LogisticRegression",
    "Perceptron", "SGDClassifier", "SGDRegressor",
    "HuberRegressor", "QuantileRegressor", "TheilSenRegressor",
    "RANSACRegressor", "BayesianRidge", "ARDRegression", "PoissonRegressor",
    "GammaRegressor", "TweedieRegressor", "RidgeCV", "LassoCV", "ElasticNetCV",
    "LogisticRegressionCV", "OrthogonalMatchingPursuit",
]
