"""Support vector machines: the widest possible margin.

THE IDEA
--------
Many boundaries separate two classes; the perceptron takes whichever it trips
over first. An SVM asks which is BEST, and answers: the one furthest from both
classes. Maximising that margin is a bet about generalisation -- a boundary
squeezed against the training points is one noisy sample away from being wrong.

Only the points ON the margin matter. Everything else could be deleted without
moving the boundary an inch. Those few are the SUPPORT VECTORS, and they are why
the model is compact regardless of how much data it was fitted on.

Real data is rarely separable, so "soft margin" allows violations at a price
``C``: large C means few violations and a narrow margin (risking overfit), small
C means a wide margin that tolerates errors. That is the bias-variance dial.

THE KERNEL TRICK
----------------
The SVM optimization touches the data ONLY through inner products
``x_i . x_j`` -- never through individual coordinates. So replace every inner
product with ``k(x_i, x_j)``, and you are implicitly working in whatever space
that kernel corresponds to. An RBF kernel corresponds to an infinite-dimensional
space, in which almost any dataset is linearly separable.

Nothing is ever mapped there. That is the trick: infinite dimensions, finite
work, because the algorithm only asked for inner products in the first place.

WHY THE OPTIMIZER IS THE INTERESTING PART
-----------------------------------------
Unlike most models here, an SVM is a genuinely CONSTRAINED optimization -- box
constraints on every variable plus one equality -- so gradient descent does not
apply. ``_smo`` implements Sequential Minimal Optimization properly, and is
worth reading as a worked example of solving such a problem: KKT conditions,
working-set selection, and an incrementally maintained gradient.

THE COST
--------
The kernel matrix is n-by-n, so kernel SVMs are for thousands of samples, not
millions. Beyond that either use ``LinearSVC`` (which never forms a kernel and
scales linearly), or approximate the kernel explicitly with
``nupyml.kernel_approximation`` and feed a linear model.

Note also that the margin is a distance, not a probability -- see
``probability=True`` and ``nupyml.calibration``.
"""
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


def _smo(K, y, C, tol=1e-3, max_iter=-1):
    """Sequential Minimal Optimization with second-order working-set selection.

    THE PROBLEM
    -----------
    Training an SVM means solving this quadratic program over the dual
    variables ``alpha`` (one per training sample)::

        minimize    0.5 * alpha' Q alpha  -  sum(alpha)
        subject to  0 <= alpha_i <= C_i          (the "box" constraint)
                    sum_i alpha_i * y_i = 0      (the "linear" constraint)

    where ``Q_ij = y_i * y_j * K(x_i, x_j)``. Q is dense and n-by-n, so for
    100k samples it would need 80 GB. Any practical solver therefore has to
    avoid touching all of Q at once.

    THE IDEA
    --------
    SMO optimizes exactly TWO alphas at a time and holds the rest fixed. Two is
    the smallest number that can move at all: the linear constraint pins the
    weighted sum, so changing one alpha alone would break it. With only two
    free variables the subproblem has a closed-form solution -- no inner
    optimizer is needed.

    Each step therefore asks two questions:

    1. *Which pair should we move?*  (working-set selection, below)
    2. *How far?*  (the closed-form update, below)

    CHOOSING THE PAIR
    -----------------
    The optimum is characterised by the KKT conditions, which here reduce to a
    simple statement: define ``I_up`` as the samples whose alpha may still
    increase in the useful direction and ``I_low`` as those that may decrease.
    At the optimum::

        max_{t in I_up} (-y_t * G_t)  <=  min_{t in I_low} (-y_t * G_t)

    where ``G`` is the gradient of the objective. Any pair that violates this
    is a "violating pair", and moving it is guaranteed to improve the
    objective. The gap between those two quantities is the natural stopping
    criterion.

    Picking ``i`` as the maximal violator is first-order information: it says
    which direction is steepest, but not how much progress a step will make. A
    steep direction with a tiny feasible step is worse than a shallow one with
    a long step. So ``j`` is chosen by second-order information: for each
    candidate we can predict the exact objective decrease in closed form and
    take the best. This costs one extra O(n) scan and buys far fewer
    iterations.

    KEEPING THE GRADIENT CHEAP
    --------------------------
    Recomputing G from scratch would be O(n^2) per step and dominate
    everything. But only two alphas change per step, so::

        G += y * (K[i] * delta_i * y_i  +  K[j] * delta_j * y_j)

    updates it exactly in O(n). This is what makes SMO practical.

    Selection is fully determined by the gradient, so the solver is
    deterministic -- it takes no random seed.

    Parameters
    ----------
    K : (n, n) array
        Precomputed kernel matrix.
    y : (n,) array
        Targets in {-1, +1}.
    C : float or (n,) array
        Upper bound on each alpha. Per-sample values implement sample weights.
    tol : float
        Stop once the KKT violation falls below this.
    max_iter : int
        Cap on pair updates; <= 0 means run to convergence.

    Returns
    -------
    (alpha, b, n_iter)
        The decision value for a new point is ``sum_i alpha_i y_i K(x, x_i) + b``.

    References
    ----------
    Platt (1998), "Sequential Minimal Optimization".
    Fan, Chen & Lin (2005), "Working Set Selection Using Second Order
    Information for Training SVM", JMLR 6:1889-1918.
    """
    n = len(y)
    C = np.broadcast_to(np.asarray(C, dtype=np.float64), (n,)).astype(np.float64)
    alpha = np.zeros(n)
    # gradient of the dual objective; at alpha = 0 it is -e
    G = -np.ones(n)
    Kdiag = np.diag(K).copy()
    tau = 1e-12
    eps = 1e-12
    cap = max_iter if max_iter and max_iter > 0 else max(10_000_000, 100 * n)

    it = 0
    for it in range(1, cap + 1):
        neg_yG = -y * G
        free_up = alpha < C - eps
        free_low = alpha > eps
        # I_up and I_low: the directions each alpha is still free to move in
        up = (free_up & (y > 0)) | (free_low & (y < 0))
        low = (free_low & (y > 0)) | (free_up & (y < 0))
        if not up.any() or not low.any():
            break
        i = int(np.where(up, neg_yG, -np.inf).argmax())
        m_up = neg_yG[i]
        M_low = np.where(low, neg_yG, np.inf).min()
        if m_up - M_low < tol:            # KKT satisfied within tolerance
            break

        # second-order pick of j: the candidate promising the biggest drop in
        # the objective, not merely the biggest violation
        b_t = m_up - neg_yG
        # Q_ii + Q_tt - 2 Q_it collapses to ||phi_i - phi_t||^2, always >= 0
        a_t = np.maximum(Kdiag[i] + Kdiag - 2.0 * K[i], tau)
        cand = low & (b_t > 0)
        if not cand.any():
            break
        obj = np.where(cand, -(b_t ** 2) / a_t, np.inf)
        j = int(obj.argmin())

        quad = max(Kdiag[i] + Kdiag[j] - 2.0 * K[i, j], tau)
        ai_old, aj_old = alpha[i], alpha[j]
        if y[i] != y[j]:
            # the pair moves along a_i - a_j = const
            delta = (-G[i] - G[j]) / quad
            diff = ai_old - aj_old
            ai, aj = ai_old + delta, aj_old + delta
            if diff > 0:
                if aj < 0:
                    aj, ai = 0.0, diff
            else:
                if ai < 0:
                    ai, aj = 0.0, -diff
            if diff > C[i] - C[j]:
                if ai > C[i]:
                    ai, aj = C[i], C[i] - diff
            else:
                if aj > C[j]:
                    aj, ai = C[j], C[j] + diff
        else:
            # the pair moves along a_i + a_j = const
            delta = (G[i] - G[j]) / quad
            total = ai_old + aj_old
            ai, aj = ai_old - delta, aj_old + delta
            if total > C[i]:
                if ai > C[i]:
                    ai, aj = C[i], total - C[i]
            else:
                if aj < 0:
                    aj, ai = 0.0, total
            if total > C[j]:
                if aj > C[j]:
                    aj, ai = C[j], total - C[j]
            else:
                if ai < 0:
                    ai, aj = 0.0, total
        alpha[i], alpha[j] = ai, aj
        # only coordinates i and j changed, so the gradient updates in O(n)
        G += y * (K[i] * ((ai - ai_old) * y[i]) + K[j] * ((aj - aj_old) * y[j]))

    # rho from the free support vectors, falling back to the KKT bracket
    yG = y * G
    free = (alpha > eps) & (alpha < C - eps)
    if free.any():
        rho = float(yG[free].mean())
    else:
        at_upper, at_lower = alpha >= C - eps, alpha <= eps
        ub_mask = (at_upper & (y < 0)) | (at_lower & (y > 0))
        lb_mask = (at_upper & (y > 0)) | (at_lower & (y < 0))
        ub = yG[ub_mask].min() if ub_mask.any() else np.inf
        lb = yG[lb_mask].max() if lb_mask.any() else -np.inf
        rho = float((ub + lb) / 2) if np.isfinite(ub + lb) else 0.0
    return alpha, -rho, it


class SVC(BaseEstimator, ClassifierMixin):
    """Kernel SVM classifier (one-vs-one for multiclass)."""

    def __init__(self, C=1.0, kernel="rbf", gamma="scale", degree=3, coef0=0.0,
                 tol=1e-3, probability=False, max_iter=-1,
                 random_state=None):
        self.C = C
        self.kernel = kernel
        self.gamma = gamma
        self.degree = degree
        self.coef0 = coef0
        self.tol = tol
        self.probability = probability
        self.max_iter = max_iter
        self.random_state = random_state

    def _gamma_value(self, X):
        if self.gamma == "scale":
            return 1.0 / (X.shape[1] * X.var())
        if self.gamma == "auto":
            return 1.0 / X.shape[1]
        return float(self.gamma)

    def fit(self, X, y, sample_weight=None):
        X, y = check_X_y(X, y)
        self._le = LabelEncoder().fit(y)
        self.classes_ = self._le.classes_
        y_idx = self._le.transform(y)
        w = np.ones(len(X)) if sample_weight is None \
            else np.asarray(sample_weight, dtype=np.float64)
        self._gamma = self._gamma_value(X)
        self._kfn = _kernel_fn(self.kernel, self._gamma, self.degree, self.coef0)
        k = len(self.classes_)
        self._models = {}
        self.n_iter_ = 0
        for a in range(k):
            for bcls in range(a + 1, k):
                mask = (y_idx == a) | (y_idx == bcls)
                Xa = X[mask]
                ya = np.where(y_idx[mask] == bcls, 1.0, -1.0)
                K = self._kfn(Xa, Xa)
                alpha, b, n_it = _smo(K, ya, self.C * w[mask], tol=self.tol,
                                      max_iter=self.max_iter)
                sv = alpha > 1e-8
                self.n_iter_ = max(getattr(self, "n_iter_", 0), n_it)
                self._models[(a, bcls)] = (Xa[sv], ya[sv] * alpha[sv], b)
        self.support_vectors_ = np.vstack([m[0] for m in self._models.values()]) \
            if self._models else np.empty((0, X.shape[1]))
        if self.probability:
            self._fit_platt(X, y)
        return self

    def _fit_platt(self, X, y):
        """SVM margins are not probabilities; fit a sigmoid to cross-validated
        decision values (Platt 1999) so predict_proba is calibrated."""
        from ..calibration import CalibratedClassifierCV
        from ..base import clone
        base = clone(self).set_params(probability=False)
        self._calibrator = CalibratedClassifierCV(
            base, method="sigmoid", cv=3).fit(X, y)

    def predict_proba(self, X):
        if not self.probability:
            raise AttributeError(
                "predict_proba is only available when probability=True; a "
                "decision_function value is a margin, not a probability")
        check_is_fitted(self, "_calibrator")
        return self._calibrator.predict_proba(X)

    def predict_log_proba(self, X):
        return np.log(self.predict_proba(X))

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
    """Epsilon-SVR.

    Solved in the 2n-variable dual (alpha, alpha*) rather than collapsed to
    beta = alpha - alpha*: the collapsed form needs |beta|, which is
    non-smooth at zero and which a quasi-Newton solver cannot drive to exact
    zeros, so no sample is ever dropped from the support set. The bias is
    folded into the kernel (K + 1), which removes the equality constraint and
    leaves a smooth box-constrained QP.
    """

    def __init__(self, C=1.0, epsilon=0.1, kernel="rbf", gamma="scale",
                 degree=3, coef0=0.0, max_iter=2000):
        self.C = C
        self.epsilon = epsilon
        self.kernel = kernel
        self.gamma = gamma
        self.degree = degree
        self.coef0 = coef0
        self.max_iter = max_iter

    def _gamma_value(self, X):
        if self.gamma == "scale":
            return 1.0 / (X.shape[1] * X.var())
        if self.gamma == "auto":
            return 1.0 / X.shape[1]
        return float(self.gamma)

    def fit(self, X, y, sample_weight=None):
        import scipy.optimize
        X, y = check_X_y(X, y, y_numeric=True)
        n = len(y)
        w = np.ones(n) if sample_weight is None \
            else np.asarray(sample_weight, dtype=np.float64)
        g = self._gamma_value(X)
        self._kfn = _kernel_fn(self.kernel, g, self.degree, self.coef0)
        K = self._kfn(X, X)
        Kb = K + 1.0
        eps = self.epsilon
        upper = self.C * w

        def obj(z):
            a, astar = z[:n], z[n:]
            beta = a - astar
            Kbb = Kb @ beta
            val = 0.5 * beta @ Kbb + eps * z.sum() - y @ beta
            gb = Kbb - y
            return val, np.concatenate([gb + eps, -gb + eps])

        res = scipy.optimize.minimize(
            obj, np.zeros(2 * n), jac=True, method="L-BFGS-B",
            bounds=[(0.0, u) for u in np.r_[upper, upper]],
            options={"maxiter": self.max_iter})
        beta = res.x[:n] - res.x[n:]
        sv = np.abs(beta) > 1e-8
        self._sv_X = X[sv]
        self._sv_coef = beta[sv]
        self.support_ = np.where(sv)[0]
        self.support_vectors_ = X[sv]
        self.dual_coef_ = beta[sv][None, :]
        # the folded bias is just the sum of the dual coefficients
        self.intercept_ = float(beta.sum())
        return self

    def predict(self, X):
        check_is_fitted(self, "_sv_X")
        X = check_array(X)
        if len(self._sv_X) == 0:
            return np.full(len(X), self.intercept_)
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


def _nu_svc_dual(K, t, nu, max_iter=500):
    """Scholkopf's nu-SVC dual:

        min 0.5 a'Qa   s.t. 0 <= a_i <= 1/n,  a'y = 0,  sum(a) >= nu

    nu is then both an upper bound on the fraction of margin errors and a
    lower bound on the fraction of support vectors.
    """
    import scipy.optimize
    n = len(t)
    Q = np.outer(t, t) * K

    def obj(a):
        Qa = Q @ a
        return 0.5 * a @ Qa, Qa

    # a feasible start: split nu evenly across each class
    a0 = np.zeros(n)
    for sign in (1.0, -1.0):
        idx = np.where(t == sign)[0]
        a0[idx] = min(nu / 2 / max(len(idx), 1), 1.0 / n)
    res = scipy.optimize.minimize(
        obj, a0, jac=True, method="SLSQP", bounds=[(0.0, 1.0 / n)] * n,
        constraints=[
            {"type": "eq", "fun": lambda a: a @ t, "jac": lambda a: t},
            {"type": "ineq", "fun": lambda a: a.sum() - nu,
             "jac": lambda a: np.ones_like(a)},
        ],
        options={"maxiter": max_iter, "ftol": 1e-9})
    return np.clip(res.x, 0.0, 1.0 / n)


class NuSVC(SVC):
    """SVC parameterized by nu: an upper bound on the training-error fraction
    and a lower bound on the support-vector fraction."""

    def __init__(self, nu=0.5, kernel="rbf", gamma="scale", degree=3,
                 coef0=0.0, tol=1e-3, probability=False, max_iter=500,
                 random_state=None):
        super().__init__(C=1.0, kernel=kernel, gamma=gamma, degree=degree,
                         coef0=coef0, tol=tol, probability=probability,
                         max_iter=max_iter, random_state=random_state)
        self.nu = nu

    @classmethod
    def _get_param_names(cls):
        return ["coef0", "degree", "gamma", "kernel", "max_iter", "nu",
                "probability", "random_state", "tol"]

    def fit(self, X, y, sample_weight=None):
        X, y = check_X_y(X, y)
        self._le = LabelEncoder().fit(y)
        self.classes_ = self._le.classes_
        y_idx = self._le.transform(y)
        self._gamma = self._gamma_value(X)
        self._kfn = _kernel_fn(self.kernel, self._gamma, self.degree, self.coef0)
        k = len(self.classes_)
        self._models = {}
        for a_cls in range(k):
            for b_cls in range(a_cls + 1, k):
                mask = (y_idx == a_cls) | (y_idx == b_cls)
                Xa = X[mask]
                t = np.where(y_idx[mask] == b_cls, 1.0, -1.0)
                K = self._kfn(Xa, Xa)
                alpha = _nu_svc_dual(K, t, self.nu, self.max_iter)
                sv = alpha > 1e-8
                coef = (alpha * t)[sv]
                f0 = K[:, sv] @ coef
                # rho/b come from the margin SVs of each class
                upper = 1.0 / len(t)
                margin = sv & (alpha < upper - 1e-9)
                pos = margin & (t > 0)
                neg = margin & (t < 0)
                if pos.any() and neg.any():
                    b = -(f0[pos].mean() + f0[neg].mean()) / 2
                else:
                    b = -f0[sv].mean() if sv.any() else 0.0
                self._models[(a_cls, b_cls)] = (Xa[sv], coef, float(b))
        self.support_vectors_ = np.vstack([m[0] for m in self._models.values()]) \
            if self._models else np.empty((0, X.shape[1]))
        if self.probability:
            self._fit_platt(X, y)
        return self


class NuSVR(SVR):
    """SVR where nu bounds the fraction of support vectors.

    nu-SVR and epsilon-SVR have the same solution set (Chang & Lin 2002): for
    each nu there is an epsilon giving the identical fit. Rather than solve the
    harder nu dual directly, bisect epsilon until the support-vector fraction
    matches nu, which reuses the well-conditioned epsilon-SVR solver.
    """

    def __init__(self, nu=0.5, C=1.0, kernel="rbf", gamma="scale", degree=3,
                 coef0=0.0, max_iter=500, tol=1e-3):
        super().__init__(C=C, epsilon=0.1, kernel=kernel, gamma=gamma,
                         degree=degree, coef0=coef0, max_iter=max_iter)
        self.nu = nu
        self.tol = tol

    @classmethod
    def _get_param_names(cls):
        return ["C", "coef0", "degree", "gamma", "kernel", "max_iter", "nu",
                "tol"]

    def _sv_fraction(self, X, y, eps):
        self.epsilon = eps
        SVR.fit(self, X, y)
        return len(self._sv_coef) / len(y)

    def fit(self, X, y):
        X, y = check_X_y(X, y, y_numeric=True)
        lo, hi = 0.0, float(np.abs(y - np.median(y)).max())
        # sv fraction falls monotonically as the tube widens
        if self._sv_fraction(X, y, hi) > self.nu:
            self.epsilon_ = self.epsilon
            self.sv_fraction_ = len(self._sv_coef) / len(y)
            return self          # even the widest tube keeps too many SVs
        best_eps = lo
        for _ in range(40):
            mid = (lo + hi) / 2
            frac = self._sv_fraction(X, y, mid)
            if frac >= self.nu:
                # nu lower-bounds the support fraction, so only tubes that
                # keep enough SVs are admissible; the fraction is a step
                # function of epsilon and bisection can otherwise settle on
                # the wrong side of a jump
                best_eps = mid
                lo = mid
                if frac - self.nu < self.tol:
                    break
            else:
                hi = mid
        self._sv_fraction(X, y, best_eps)
        self.epsilon_ = best_eps
        self.sv_fraction_ = len(self._sv_coef) / len(y)
        return self


__all__ = ["SVC", "SVR", "LinearSVC", "NuSVC", "NuSVR"]
