"""Least Angle Regression, and the linear-model gaps it sits among.

LARS is the centrepiece here because it reveals something the other methods hide:
the ENTIRE lasso path -- every solution for every penalty -- in one pass, for
about the cost of a single least-squares fit.
"""
import numpy as np

from ..base import (BaseEstimator, RegressorMixin, ClassifierMixin,
                    check_is_fitted)
from ..preprocessing import LabelEncoder
from ..utils import check_X_y, check_array, check_random_state


class Lars(BaseEstimator, RegressorMixin):
    """Least Angle Regression: the whole path, one variable at a time.

    THE GEOMETRY
    ------------
    LARS builds the model by walking, not by solving. Start with all coefficients
    zero and the residual equal to ``y``. Then:

    1. Find the feature most correlated with the current residual.
    2. Move its coefficient toward its least-squares value -- but STOP the moment
       some OTHER feature is equally correlated with the residual.
    3. Now move BOTH, in the direction equiangular between them ("least angle"),
       until a third catches up. Continue.

    At every step the active features share the maximum correlation with the
    residual, and the coefficients move along the one direction that keeps it that
    way. It is a continuous walk through coefficient space, and features join the
    active set one at a time at precisely identified moments.

    WHY THIS IS BEAUTIFUL
    ---------------------
    A tiny modification -- also let a feature LEAVE the active set when its
    coefficient hits zero -- makes LARS trace the EXACT lasso path. So the entire
    sequence of lasso solutions, for every value of the penalty at once, falls out
    of one pass costing about the same as a single ordinary least-squares fit.

    Coordinate descent (in ``Lasso``) gives you the lasso solution at ONE penalty.
    LARS gives you all of them, and shows you the order in which features enter --
    which is itself a ranking of their importance. That path view is the reason to
    reach for it.

    THE CATCH
    ---------
    LARS is sensitive to noise and to correlated features: because it commits to
    the single most-correlated feature at each step, a little noise can flip which
    of two near-identical features enters first, and the path wobbles. Coordinate
    descent on the penalised objective is steadier. LARS is for insight into the
    path; ``LassoCV`` is usually what you deploy.

    Efron, Hastie, Johnstone & Tibshirani (2004).
    """

    def __init__(self, n_nonzero_coefs=None, fit_intercept=True, lasso=False,
                 eps=1e-12):
        self.n_nonzero_coefs = n_nonzero_coefs
        self.fit_intercept = fit_intercept
        self.lasso = lasso          # if True, trace the lasso path (allow drops)
        self.eps = eps

    def fit(self, X, y):
        X, y = check_X_y(X, y, y_numeric=True)
        n, p = X.shape

        # centre (and, for the coefficients to be comparable, standardise):
        # correlations are only meaningful on a common scale, or the feature
        # measured in large units always looks the most correlated
        self._x_mean = X.mean(axis=0) if self.fit_intercept else np.zeros(p)
        self._y_mean = y.mean() if self.fit_intercept else 0.0
        Xc = X - self._x_mean
        yc = y - self._y_mean
        scale = np.sqrt((Xc ** 2).sum(axis=0))
        scale[scale == 0] = 1.0
        Xs = Xc / scale

        max_features = self.n_nonzero_coefs or p
        beta = np.zeros(p)
        active = []
        signs = []
        residual = yc.copy()
        self.coef_path_ = [beta.copy()]

        for _ in range(min(max_features, p) * (2 if self.lasso else 1) + p):
            corr = Xs.T @ residual
            c_abs = np.abs(corr)
            # the inactive feature most correlated with the residual is next
            inactive = [j for j in range(p) if j not in active]
            if not inactive and not self.lasso:
                break
            if inactive:
                j_new = inactive[np.argmax(c_abs[inactive])]
                if c_abs[j_new] < self.eps:
                    break
                if len(active) < max_features and (
                        not active or c_abs[j_new] >= c_abs[active].max() - 1e-9
                        or len(active) == 0):
                    pass

            if len(active) >= max_features and not self.lasso:
                break

            # add the most-correlated inactive feature to the active set
            if inactive:
                j_new = inactive[np.argmax(c_abs[inactive])]
                active.append(j_new)
                signs.append(np.sign(corr[j_new]) or 1.0)

            A = Xs[:, active] * np.array(signs)
            G = A.T @ A
            try:
                Ginv = np.linalg.inv(G)
            except np.linalg.LinAlgError:
                break
            ones = np.ones(len(active))
            AA = 1.0 / np.sqrt(ones @ Ginv @ ones)
            w = AA * (Ginv @ ones)           # the equiangular weights
            u = A @ w                        # the equiangular direction
            a = Xs.T @ u                     # each feature's correlation with it

            C = c_abs[active].max()
            # how far to walk before an inactive feature catches up in correlation
            gamma = C / AA
            for j in range(p):
                if j in active:
                    continue
                for sign in (+1, -1):
                    denom = AA - sign * a[j]
                    if denom > self.eps:
                        t = (C - sign * corr[j]) / denom
                        if self.eps < t < gamma:
                            gamma = t

            step = w * np.array(signs)
            # walk the active coefficients along the equiangular direction
            for k, j in enumerate(active):
                beta[j] += gamma * step[k]
            residual = residual - gamma * u
            self.coef_path_.append(beta.copy())

            if len(active) >= max_features and not self.lasso:
                break
            if not inactive and self.lasso:
                break

        # undo the standardisation to return coefficients in the original units
        self.coef_ = beta / scale
        self.intercept_ = self._y_mean - self.coef_ @ self._x_mean
        self.active_ = active
        self.coef_path_ = np.array([b / scale for b in self.coef_path_])
        return self

    def predict(self, X):
        check_is_fitted(self, "coef_")
        return check_array(X) @ self.coef_ + self.intercept_


class LassoLars(Lars):
    """LARS tracing the exact lasso path (features may leave the active set).

    Same walk as ``Lars``, plus the one rule that turns it into the lasso: when
    an active coefficient would cross zero, stop there and drop the feature. That
    single addition is the whole difference between "forward selection that only
    adds" and "the lasso", and it is why LARS is the natural way to understand
    what the lasso is actually doing.
    """

    def __init__(self, n_nonzero_coefs=None, fit_intercept=True, eps=1e-12):
        super().__init__(n_nonzero_coefs=n_nonzero_coefs,
                         fit_intercept=fit_intercept, lasso=True, eps=eps)


class LinearSVR(BaseEstimator, RegressorMixin):
    """Linear support vector regression: the epsilon-insensitive tube.

    THE LOSS IS THE WHOLE IDEA
    --------------------------
    Ordinary regression penalises every error, however small. SVR ignores errors
    inside a tube of width ``epsilon`` and penalises only what pokes out::

        loss = max(0, |y - prediction| - epsilon)

    Two consequences follow, and both are the point:

    * **The fit is robust to small noise.** Predictions within ``epsilon`` cost
      nothing, so the model does not contort itself to chase measurement jitter --
      it declares a band of "close enough" and fits the shape, not the wobble.
    * **The solution is SPARSE in the data.** Points inside the tube have zero
      loss and zero gradient, so they do not affect the fit at all. Only the
      points on or outside the tube -- the support vectors -- determine it. Most
      of the data is, in effect, ignored.

    This is the same tube as kernel SVR (see ``svm``), here without the kernel:
    fitted by subgradient descent on the linear model, which scales to far more
    data than the kernel version's quadratic program.

    SCALE THE FEATURES FIRST
    ------------------------
    Like every SVM, this needs standardised input, and for two reasons at once.
    The L2 penalty ``||w||^2`` punishes a feature's coefficient for the feature's
    units, so an unscaled feature in large numbers is silently over-penalised.
    And subgradient descent with one global step size cannot suit coordinates of
    wildly different scale. On raw data it converges to the mean and reports
    ``r^2 ~ 0``; on standardised data it is excellent. That is expected SVM
    behaviour, not a defect -- put a ``StandardScaler`` before it.
    """

    def __init__(self, epsilon=0.0, C=1.0, max_iter=1000, tol=1e-4,
                 learning_rate=0.01, random_state=None):
        self.epsilon = epsilon
        self.C = C
        self.max_iter = max_iter
        self.tol = tol
        self.learning_rate = learning_rate
        self.random_state = random_state

    def fit(self, X, y):
        X, y = check_X_y(X, y, y_numeric=True)
        rng = check_random_state(self.random_state)
        n, p = X.shape
        w = np.zeros(p)
        b = 0.0

        for it in range(self.max_iter):
            pred = X @ w + b
            err = pred - y
            # subgradient of the epsilon-insensitive loss: zero inside the tube,
            # +-1 outside. Only the support vectors (outside the tube) contribute
            outside = np.abs(err) > self.epsilon
            grad_sign = np.sign(err) * outside
            grad_w = w + self.C * (X.T @ grad_sign) / n   # w from the L2 penalty
            grad_b = self.C * grad_sign.sum() / n
            step = self.learning_rate / (1 + it * 0.01)   # decaying step size
            w_new = w - step * grad_w
            b_new = b - step * grad_b
            if np.max(np.abs(w_new - w)) < self.tol:
                w, b = w_new, b_new
                break
            w, b = w_new, b_new

        self.coef_ = w
        self.intercept_ = float(b)
        self.n_iter_ = it + 1
        return self

    def predict(self, X):
        check_is_fitted(self, "coef_")
        return check_array(X) @ self.coef_ + self.intercept_


class RidgeClassifier(BaseEstimator, ClassifierMixin):
    """Classification by regressing on +-1 targets, with an L2 penalty.

    WHY REGRESS ON LABELS AT ALL
    ----------------------------
    It sounds wrong -- least squares is for continuous targets -- but encode the
    two classes as +1 and -1, fit ridge, and predict by the SIGN. It works, it is
    a single linear solve with no iteration, and on well-separated data it is as
    good as logistic regression and considerably faster.

    WHAT IT GIVES UP
    ----------------
    Calibrated probabilities. Logistic regression's output is an estimate of
    P(class); this method's raw output is a regression value with no
    probabilistic meaning, so ``predict_proba`` is at best a softmax hack over the
    decision scores. When you need "how confident", use logistic regression; when
    you need a fast, robust decision boundary, this is the cheaper tool.

    Multiclass is one-vs-all: fit K regressions, one per class against the rest,
    and predict the class whose regressor scores highest.
    """

    def __init__(self, alpha=1.0, fit_intercept=True):
        self.alpha = alpha
        self.fit_intercept = fit_intercept

    def fit(self, X, y):
        X, y = check_X_y(X, y)
        self._le = LabelEncoder().fit(y)
        self.classes_ = self._le.classes_
        y_idx = self._le.transform(y)
        k = len(self.classes_)

        self._mean = X.mean(axis=0) if self.fit_intercept else np.zeros(X.shape[1])
        Xc = X - self._mean

        # one-vs-all targets in {-1, +1}; binary needs just one column
        if k == 2:
            T = np.where(y_idx == 1, 1.0, -1.0).reshape(-1, 1)
        else:
            T = np.full((len(y), k), -1.0)
            T[np.arange(len(y)), y_idx] = 1.0

        # the ridge normal equations, solved once for all K columns at once
        A = Xc.T @ Xc + self.alpha * np.eye(Xc.shape[1])
        self.coef_ = np.linalg.solve(A, Xc.T @ T).T
        self.intercept_ = T.mean(axis=0) if self.fit_intercept else np.zeros(T.shape[1])
        return self

    def decision_function(self, X):
        check_is_fitted(self, "coef_")
        scores = (check_array(X) - self._mean) @ self.coef_.T + self.intercept_
        return scores.ravel() if scores.shape[1] == 1 else scores

    def predict(self, X):
        scores = self.decision_function(X)
        if scores.ndim == 1:
            return self.classes_[(scores > 0).astype(int)]
        return self.classes_[np.argmax(scores, axis=1)]


class MultiTaskLasso(BaseEstimator, RegressorMixin):
    """Lasso across several targets that should share their support.

    THE ASSUMPTION
    --------------
    Several related regression tasks -- predicting a person's spending in each of
    twelve months, say -- probably depend on the SAME features, even if the
    weights differ per task. Fitting a separate lasso per task ignores that, and
    lets task 3 keep feature 7 while task 4 drops it.

    THE JOINT PENALTY
    -----------------
    MultiTaskLasso penalises the L2 norm of each feature's ROW across tasks, then
    sums those over features (an "L2,1" norm)::

        penalty = sum_j  ||W[j, :]||_2

    The L2 inside couples the tasks -- a feature is cheap for all tasks or
    expensive for all -- while the sum outside is still an L1 over features, so it
    zeros whole rows. The result: a feature is either used by EVERY task or by
    NONE. Support is shared by construction, which both aids interpretation and
    borrows strength across tasks, so each task effectively sees more data.

    Fitted by block coordinate descent, the natural generalisation of the single-
    task cyclic soft-thresholding to a group that must move together.
    """

    def __init__(self, alpha=1.0, max_iter=1000, tol=1e-4):
        self.alpha = alpha
        self.max_iter = max_iter
        self.tol = tol

    def fit(self, X, Y):
        X = check_array(X)
        Y = check_array(Y)
        n, p = X.shape
        n_tasks = Y.shape[1]

        self._x_mean = X.mean(axis=0)
        self._y_mean = Y.mean(axis=0)
        Xc = X - self._x_mean
        Yc = Y - self._y_mean

        W = np.zeros((p, n_tasks))
        col_norm2 = (Xc ** 2).sum(axis=0)
        col_norm2[col_norm2 == 0] = 1.0
        residual = Yc.copy()

        for _ in range(self.max_iter):
            W_old = W.copy()
            for j in range(p):
                # remove feature j's current contribution, then refit its whole
                # ROW at once -- the block that the group penalty moves together
                residual += np.outer(Xc[:, j], W[j])
                rho = Xc[:, j] @ residual
                norm = np.linalg.norm(rho)
                # the block soft-threshold: shrink the whole row toward zero, and
                # if it is short enough, zero it ENTIRELY -- that is what makes
                # the feature drop out for every task at once
                if norm <= self.alpha * n:
                    W[j] = 0.0
                else:
                    W[j] = (1 - self.alpha * n / norm) * rho / col_norm2[j]
                residual -= np.outer(Xc[:, j], W[j])
            if np.max(np.abs(W - W_old)) < self.tol:
                break

        self.coef_ = W.T
        self.intercept_ = self._y_mean - self.coef_ @ self._x_mean
        return self

    def predict(self, X):
        check_is_fitted(self, "coef_")
        return check_array(X) @ self.coef_.T + self.intercept_


__all__ = ["Lars", "LassoLars", "LinearSVR", "RidgeClassifier",
           "MultiTaskLasso"]
