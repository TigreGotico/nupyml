"""Linear models: predictions that are weighted sums of the features.

Every model here computes ``X @ w + b`` and differs only in two choices: what
it does with that number, and what it penalises ``w`` for.

WHAT THE NUMBER MEANS
---------------------
- Regression uses it directly as the prediction.
- Classification pushes it through a squashing function to get a probability
  (``LogisticRegression``), or just reads its sign (``Perceptron``,
  ``LinearSVC``).

The boundary between classes is therefore always a flat surface -- a line in
2-D, a plane in 3-D. That is the defining limitation, and also why these models
are so well understood: the problems are usually convex, so "the" solution
exists and is unique, with no local minima to escape.

WHAT THE PENALTY DOES
---------------------
Left alone, a linear model with many features will fit noise. A penalty on
``w`` trades a little bias for a large drop in variance:

============ ================== ==============================================
penalty      model              effect
============ ================== ==============================================
none         LinearRegression   fits whatever the data says, noise included
L2 ``w^2``   Ridge              shrinks weights smoothly toward zero
L1 ``|w|``   Lasso              drives weights to EXACTLY zero: selection
both         ElasticNet         shrinks, selects, and shares among correlates
============ ================== ==============================================

Why L1 selects and L2 does not is the most useful geometric fact in this file,
and is explained in ``Lasso``.

HOW THEY ARE SOLVED
-------------------
The penalty also decides the algorithm, which is why these are separate classes
rather than one class with a flag:

- ``LinearRegression``, ``Ridge`` -- the optimum has a closed form; just solve
  a linear system.
- ``Lasso``, ``ElasticNet`` -- ``|w|`` has no derivative at zero, so no
  gradient method reaches an exact zero. Coordinate descent instead.
- ``LogisticRegression`` -- convex but no closed form; hand the gradient to a
  quasi-Newton solver.
- ``SGDClassifier``, ``SGDRegressor`` -- same objectives, approximated one
  mini-batch at a time, for data too large to hold at once.
"""
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
    """Ordinary least squares: minimise ``||y - Xw||^2``.

    THE GEOMETRY
    ------------
    The reachable predictions ``Xw`` form a subspace -- everything the columns
    of X can build. Usually ``y`` does not lie in it. The closest point that
    does is the perpendicular projection of ``y`` onto that subspace, and
    "perpendicular" means the residual is orthogonal to every column::

        X'(y - Xw) = 0    ==>    X'X w = X'y

    Those are the normal equations, and they are what makes OLS a linear-algebra
    problem rather than an optimization one.

    WHY ``lstsq`` AND NOT ``inv(X'X) @ X'y``
    ----------------------------------------
    The textbook formula is a numerical trap. Forming ``X'X`` squares the
    condition number, so correlated features -- exactly the case where the fit
    is delicate -- lose about half the available precision. Worse, if features
    are perfectly collinear, ``X'X`` is singular and the inverse does not exist,
    though a best fit still does (many, in fact).

    ``lstsq`` sidesteps both by factorising X directly and returning the
    minimum-norm solution when the problem is degenerate.

    THE INTERCEPT
    -------------
    Centring X and y, fitting without an intercept, then recovering
    ``b = mean(y) - mean(X) @ w`` gives the same answer as appending a column
    of ones, but keeps the penalty in the subclasses off the intercept -- where
    it does not belong, since shifting the units of y should not be penalised.
    """

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
    """Least squares with an L2 penalty: ``||y - Xw||^2 + alpha*||w||^2``.

    Setting the gradient to zero gives a barely-changed normal equation::

        (X'X + alpha*I) w = X'y

    A diagonal ridge added to ``X'X`` -- hence the name. That tiny change buys
    a great deal:

    *It always has a solution.* ``X'X`` may be singular; ``X'X + alpha*I`` never
    is for ``alpha > 0``, since it lifts every eigenvalue by alpha. Ridge works
    with more features than samples, where OLS is undefined.

    *It stabilises correlated features.* Given two near-identical features, OLS
    is free to put a huge positive weight on one and a huge negative weight on
    the other -- the predictions cancel, the fit looks fine, and the weights are
    nonsense driven by noise. The penalty makes that expensive, so ridge splits
    the weight between them instead.

    *It shrinks, but never to zero.* The penalty's gradient is ``2*alpha*w``,
    which vanishes as w approaches zero, so the pull weakens exactly when it
    would need to be decisive. Every feature keeps a small weight. For actual
    selection, see ``Lasso``.

    ``alpha`` chooses where on the bias-variance curve to sit: 0 is OLS,
    infinity is the null model. ``RidgeCV`` picks it by cross-validation.
    """

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
    """Cyclic coordinate descent for the L1-penalised objectives.

    THE PROBLEM WITH GRADIENTS HERE
    -------------------------------
    ``|w|`` has a corner at zero -- no derivative. Gradient descent hovers
    around the corner, and floating point ensures it lands on 1e-17 rather than
    0.0, so "sparse" solutions are not actually sparse. Yet exact zeros are the
    entire reason to use L1.

    THE FIX: OPTIMIZE ONE WEIGHT AT A TIME
    --------------------------------------
    Freeze every weight but ``w_j``. The objective in that single variable is a
    parabola plus ``alpha*|w_j|``, and that one-dimensional problem has a
    closed-form minimum -- corner included. Cycle over j until nothing moves.

    The closed form is the soft-thresholding operator::

        w_j = sign(rho) * max(|rho| - alpha, 0) / (norm + l2)

    Read it directly: ``rho`` is how much feature j correlates with what the
    other features left unexplained. If that correlation is weaker than alpha,
    ``max(..., 0)`` returns exactly 0.0 -- a real zero, produced by a max, not
    by a limit. Otherwise the weight is pulled toward zero by alpha and stops.
    That single line is what makes lasso a selector.

    Convexity is what licenses the whole approach: coordinate descent can stall
    at a non-optimal corner on a general function, but for this objective --
    smooth plus separable-convex -- coordinate-wise optimality implies global
    optimality.

    THE RESIDUAL TRICK
    ------------------
    Recomputing ``y - Xw`` after each weight would cost O(n*d) per coordinate.
    Instead the residual is updated in place by the one column that changed::

        resid += X[:, j] * (w_j_old - w_j_new)

    O(n) per coordinate, and the reason this scales.
    """

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
    """L1-penalised least squares: ``||y - Xw||^2 + alpha*||w||_1``.

    WHY L1 SELECTS AND L2 DOES NOT
    ------------------------------
    Both penalties can be read as "minimise the error subject to a budget on
    w". The shape of the budget region decides everything.

    L2's region is a ball. L1's is a diamond -- ``|w1| + |w2| <= t`` -- with
    corners ON the axes, and a corner is exactly a point where some coordinate
    is zero.

    The solution is where the growing error contours first touch the budget
    region. A smooth ball is usually touched on a smooth part, at some generic
    point with all coordinates non-zero. A pointy diamond is most easily
    touched at a corner: corners stick out. So L1 lands on the axes, and lands
    there for a whole range of alpha, not by coincidence.

    That geometric picture is exactly what soft-thresholding does numerically:
    it flattens a whole interval of ``rho`` onto zero rather than passing
    through it.

    WHAT TO WATCH OUT FOR
    ---------------------
    Given a group of correlated features, lasso tends to keep one arbitrarily
    and zero the rest -- convenient for parsimony, misleading if you read the
    survivor as "the important one". ``ElasticNet`` exists for that case.
    """

    def __init__(self, alpha=1.0, fit_intercept=True, max_iter=1000, tol=1e-4):
        super().__init__(alpha=alpha, l1_ratio=1.0, fit_intercept=fit_intercept,
                         max_iter=max_iter, tol=tol)

    @classmethod
    def _get_param_names(cls):
        return ["alpha", "fit_intercept", "max_iter", "tol"]


class ElasticNet(_CoordinateDescent):
    """Both penalties at once: ``alpha * (l1_ratio*||w||_1 + (1-l1_ratio)/2*||w||^2)``.

    Lasso's weakness is correlated features: it picks one and discards its
    twins, and which one it picks can flip with a small change in the data.
    Ridge's weakness is that it never selects.

    Mixing them gives the "grouping effect": the L1 part still zeroes whole
    irrelevant features, while the L2 part encourages correlated survivors to
    share weight rather than fight over it. The result is stabler than lasso
    and sparser than ridge.

    ``l1_ratio`` slides between them -- 1.0 is lasso, 0.0 is ridge.
    """

    def __init__(self, alpha=1.0, l1_ratio=0.5, fit_intercept=True,
                 max_iter=1000, tol=1e-4):
        super().__init__(alpha=alpha, l1_ratio=l1_ratio,
                         fit_intercept=fit_intercept, max_iter=max_iter, tol=tol)


class LogisticRegression(BaseEstimator, ClassifierMixin):
    """Linear classification by maximum likelihood. A classifier, despite the name.

    THE MODEL
    ---------
    A linear score ``z = Xw + b`` is unbounded, and probabilities are not. The
    sigmoid maps one onto the other::

        p = 1 / (1 + exp(-z))

    This is not an arbitrary squashing choice. Rearranged, it says the model is
    linear in the LOG-ODDS::

        log(p / (1 - p)) = z

    which is what "linear model" means here, and why coefficients are read as
    "a unit of this feature multiplies the odds by ``exp(w_j)``". For more than
    two classes, softmax generalises it.

    THE LOSS
    --------
    Fit by maximising the likelihood of the observed labels, equivalently
    minimising the negative log-likelihood::

        -sum_i [ y_i*log(p_i) + (1-y_i)*log(1-p_i) ]

    Squared error is a poor fit here: it is non-convex under the sigmoid, and
    it barely punishes confident mistakes. Log-loss is convex, and its penalty
    for a confidently wrong answer grows without bound -- which is what you
    want a classifier to fear.

    THE GRADIENT
    ------------
    Despite the sigmoid and the log, the gradient collapses to::

        dL/dw = X' (p - y)

    "error times input", the same form as least squares. The sigmoid's
    derivative cancels exactly against the log-loss's -- not a coincidence, but
    a property of matching an exponential-family likelihood to its canonical
    link.

    NO CLOSED FORM
    --------------
    ``p`` depends on ``w`` non-linearly, so unlike least squares there is
    nothing to solve directly. The problem is convex with no local minima, so
    the analytic gradient is handed to L-BFGS, which builds a curvature
    estimate from successive gradients and converges in far fewer passes than
    plain gradient descent -- without ever forming the Hessian.

    ``C`` is INVERSE regularisation strength (small C, strong penalty), which
    is the SVM convention rather than the ``alpha`` used by Ridge.
    """

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
    _estimator_tags = {"binary_only": True}

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
    """Linear classification by mini-batch stochastic gradient descent.

    The models here are not new: ``loss="log"`` is logistic regression and
    ``loss="hinge"`` is a linear SVM. What changes is how they are fit.

    WHY APPROXIMATE THE GRADIENT ON PURPOSE
    ---------------------------------------
    The true gradient sums over every sample, so one step costs a full pass.
    But that sum is an average, and an average can be estimated from a sample:
    a mini-batch gives a noisy gradient for a fraction of the work. Many rough
    steps beat one exact step -- the noise mostly cancels over a sequence of
    updates, and the parameters are moving anyway.

    This is what makes the method scale to data that does not fit in memory,
    and it is what ``partial_fit`` exposes: stream a chunk, take a step, drop
    the chunk.

    THE TWO LOSSES
    --------------
    * ``hinge``: ``max(0, 1 - y*z)``. Zero once a point is correct AND at least
      a margin away, so correct-and-confident points contribute no gradient at
      all -- the boundary is set by the points near it. Not differentiable at
      the kink; the sub-gradient is used.
    * ``log``: ``log(1 + exp(-y*z))``. Never exactly zero, so every point keeps
      nudging forever, but smooth and probabilistic.

    The price of SGD is that ``learning_rate`` now matters: too large diverges,
    too small crawls. The closed-form and quasi-Newton solvers in this module
    have no such knob.
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

from ._lars import (Lars, LassoLars, LinearSVR, RidgeClassifier,
                    MultiTaskLasso)

__all__ = [
    "LinearRegression", "Ridge", "Lasso", "ElasticNet", "LogisticRegression",
    "Perceptron", "SGDClassifier", "SGDRegressor",
    "HuberRegressor", "QuantileRegressor", "TheilSenRegressor",
    "RANSACRegressor", "BayesianRidge", "ARDRegression", "PoissonRegressor",
    "GammaRegressor", "TweedieRegressor", "RidgeCV", "LassoCV", "ElasticNetCV",
    "LogisticRegressionCV", "OrthogonalMatchingPursuit",
    "Lars", "LassoLars", "LinearSVR", "RidgeClassifier", "MultiTaskLasso",
]
