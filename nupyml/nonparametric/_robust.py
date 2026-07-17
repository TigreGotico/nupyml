"""Regression when the usual assumptions break: errors in x, and outliers.

Ordinary least squares assumes the predictors are exact and the errors are
well-behaved. Both fail often enough to need answers.
"""
import numpy as np
from scipy.optimize import least_squares

from ..base import BaseEstimator, RegressorMixin, check_is_fitted
from ..utils import check_X_y, check_array, check_random_state


class TotalLeastSquares(BaseEstimator, RegressorMixin):
    """Errors in the predictors too, not only the target.

    THE ASSUMPTION OLS HIDES
    ------------------------
    Ordinary least squares minimises VERTICAL distances -- errors in ``y`` only.
    That silently assumes ``x`` is measured exactly. Often it is not: both
    variables are noisy measurements, and there is no reason ``x`` deserves to be
    treated as ground truth while ``y`` carries all the error.

    When ``x`` is noisy, OLS is not merely suboptimal, it is BIASED -- it
    systematically underestimates the slope, pulling it toward zero. This is
    "regression attenuation", and it is why a real relationship can look weaker
    than it is purely from measurement noise in the predictor.

    THE FIX
    -------
    Total least squares (a.k.a. Deming or orthogonal regression) minimises
    PERPENDICULAR distance to the line, treating both variables symmetrically. No
    variable is privileged, and the attenuation bias is gone.

    THE SOLUTION IS AN EIGENVECTOR
    ------------------------------
    Perpendicular distance to a line is minimised by the direction of GREATEST
    variance -- the first principal component of the stacked ``[X, y]`` data. So
    TLS is PCA in disguise: the fit direction is the top singular vector, and the
    normal to the fitted hyperplane is the SMALLEST one. It is a genuinely
    different geometry from OLS, not a tweak of it.
    """

    def fit(self, X, y):
        X, y = check_X_y(X, y, y_numeric=True)
        self.n_features_in_ = X.shape[1]

        # centre, so the line passes through the mean and only its DIRECTION is
        # left to find -- which is what the SVD gives
        self._x_mean = X.mean(axis=0)
        self._y_mean = y.mean()
        Z = np.column_stack([X - self._x_mean, y - self._y_mean])

        # smallest singular vector = normal to the best-fit hyperplane, since the
        # direction of least variance is the one distances are measured along
        _, _, Vt = np.linalg.svd(Z, full_matrices=False)
        normal = Vt[-1]

        # the hyperplane is normal.[x, y] = 0; solve for y to get slope + intercept
        n_xy = normal[:-1]
        n_y = normal[-1]
        if abs(n_y) < 1e-12:
            raise ValueError("target is orthogonal to the fitted plane; "
                             "TLS is ill-posed for this data")
        self.coef_ = -n_xy / n_y
        self.intercept_ = self._y_mean - self.coef_ @ self._x_mean
        return self

    def predict(self, X):
        check_is_fitted(self, "coef_")
        return check_array(X) @ self.coef_ + self.intercept_


class LTSRegressor(BaseEstimator, RegressorMixin):
    """Least Trimmed Squares: fit the majority, ignore the worst residuals.

    THE BREAKDOWN POINT
    -------------------
    OLS has a breakdown point of ZERO: a SINGLE outlier, placed far enough away,
    can drag the fitted line arbitrarily. One bad data point is enough to make the
    whole model meaningless, because a squared residual grows without limit and
    the fit will do anything to reduce the largest one.

    LTS minimises the sum of the smallest ``h`` squared residuals and simply
    ignores the rest::

        minimise  sum of the h smallest (y_i - x_i.beta)^2

    With ``h ~ n/2`` its breakdown point approaches 50% -- up to half the data can
    be arbitrarily corrupted and the fit still tracks the clean majority. That is
    the highest breakdown any regression estimator can achieve; past 50% the
    "outliers" are the majority and there is no principled way to tell which half
    is real.

    THE COMBINATORIAL CATCH
    -----------------------
    Which ``h`` points are the clean ones is unknown, and trying every subset is
    combinatorial. FAST-LTS uses random starts refined by "C-steps": fit on a
    subset, keep the ``h`` best-fitting points, refit, repeat. Each C-step
    provably cannot increase the trimmed loss, so it converges -- to a LOCAL
    optimum, which is why many random starts are run and the best kept. It is the
    same restart-and-refine shape as k-means, and it comes with the same caveat:
    no guarantee of the global optimum.

    Rousseeuw (1984); Rousseeuw & Van Driessen (2006).
    """

    def __init__(self, h=None, n_starts=50, max_c_steps=20, random_state=None):
        self.h = h
        self.n_starts = n_starts
        self.max_c_steps = max_c_steps
        self.random_state = random_state

    def _fit_subset(self, X, y, idx):
        A = np.column_stack([np.ones(len(idx)), X[idx]])
        beta, *_ = np.linalg.lstsq(A, y[idx], rcond=None)
        return beta

    def _trimmed_loss(self, X, y, beta, h):
        resid2 = (y - (X @ beta[1:] + beta[0])) ** 2
        return np.sort(resid2)[:h].sum(), np.argsort(resid2)[:h]

    def fit(self, X, y):
        X, y = check_X_y(X, y, y_numeric=True)
        rng = check_random_state(self.random_state)
        n, p = X.shape
        # the high-breakdown default: fit slightly more than half the data
        h = self.h if self.h is not None else (n + p + 1) // 2
        h = int(np.clip(h, p + 1, n))

        best_loss, best_beta = np.inf, None
        for _ in range(self.n_starts):
            # a random minimal subset to start; p+1 points define a fit
            idx = rng.choice(n, size=min(p + 1, n), replace=False)
            beta = self._fit_subset(X, y, idx)
            for _ in range(self.max_c_steps):
                # the C-step: keep the h best-fitting points, refit on them. This
                # cannot increase the trimmed loss, which is why it converges
                loss, keep = self._trimmed_loss(X, y, beta, h)
                new_beta = self._fit_subset(X, y, keep)
                if np.allclose(new_beta, beta):
                    break
                beta = new_beta
            loss, keep = self._trimmed_loss(X, y, beta, h)
            if loss < best_loss:
                best_loss, best_beta, best_keep = loss, beta, keep

        self.intercept_ = float(best_beta[0])
        self.coef_ = best_beta[1:]
        self.inlier_mask_ = np.zeros(n, dtype=bool)
        self.inlier_mask_[best_keep] = True     # which points the fit trusted
        self.h_ = h
        return self

    def predict(self, X):
        check_is_fitted(self, "coef_")
        return check_array(X) @ self.coef_ + self.intercept_


class NonlinearLeastSquares(BaseEstimator, RegressorMixin):
    """Fit an arbitrary parametric model ``f(x, params)`` by Levenberg-Marquardt.

    WHEN A LINE IS THE WRONG SHAPE ENTIRELY
    ---------------------------------------
    Everything else here fits a flexible curve of no particular form. Sometimes
    you KNOW the form -- an exponential decay, a Michaelis-Menten saturation, a
    logistic growth -- and want its PARAMETERS, because the parameters mean
    something (a rate constant, a half-saturation point). That is a fundamentally
    different goal from prediction, and it is what this is for.

    LEVENBERG-MARQUARDT
    -------------------
    The optimizer interpolates between two methods depending on how well it is
    doing:

    * far from the optimum -- behave like GRADIENT DESCENT: small, safe, reliable
      steps that cannot overshoot;
    * near the optimum -- behave like GAUSS-NEWTON: use the curvature to leap
      almost straight to the solution.

    A damping parameter slides between them, raised after a step that made things
    worse (be cautious) and lowered after a step that helped (be bold). It is the
    trust-region idea in miniature, and it is why LM is the default nonlinear
    least-squares solver everywhere: robust when far, fast when close.

    ``model`` is ``f(X, params) -> predictions``; ``p0`` is the starting guess,
    which matters -- LM finds a LOCAL optimum, and a nonlinear objective can have
    several.
    """

    def __init__(self, model, p0, loss="linear", max_nfev=1000):
        self.model = model
        self.p0 = np.asarray(p0, dtype=np.float64)
        self.loss = loss
        self.max_nfev = max_nfev

    def fit(self, X, y):
        X, y = check_X_y(X, y, y_numeric=True)

        # LM minimises a vector of residuals, not a scalar -- it exploits the
        # least-squares structure, which is exactly why it beats a generic
        # minimiser on this problem shape
        def residuals(params):
            return self.model(X, params) - y

        result = least_squares(residuals, self.p0, loss=self.loss,
                               max_nfev=self.max_nfev)
        self.params_ = result.x
        self.success_ = result.success
        self.cost_ = float(result.cost)
        self.n_iter_ = result.nfev
        return self

    def predict(self, X):
        check_is_fitted(self, "params_")
        return self.model(check_array(X), self.params_)


__all__ = ["TotalLeastSquares", "LTSRegressor", "NonlinearLeastSquares"]
