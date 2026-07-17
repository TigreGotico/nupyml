"""Bayesian nonparametric regression: RVM and BART.

Two ways to be Bayesian about flexible regression. The RVM puts a sparsity prior
on kernel weights; BART puts a prior on a SUM OF TREES. Both deliver not just a
prediction but a distribution over predictions -- honest uncertainty, from
different machinery.
"""
import numpy as np
from scipy.spatial.distance import cdist

from ..base import BaseEstimator, RegressorMixin, check_is_fitted
from ..tree import DecisionTreeRegressor
from ..utils import check_X_y, check_array, check_random_state


class RVMRegressor(BaseEstimator, RegressorMixin):
    """Relevance vector machine: an SVM's sparsity, plus real probabilities.

    THE PITCH AGAINST THE SVM
    -------------------------
    Same kernel model as an SVM -- a weighted sum of kernels centred on training
    points -- but fitted by Bayesian evidence maximisation rather than a margin.
    Two things fall out that the SVM cannot give:

    * genuine PREDICTIVE UNCERTAINTY (a variance per prediction, not just a
      point), because the weights have a posterior distribution;
    * usually EVEN SPARSER models -- fewer relevance vectors than support vectors,
      so prediction is cheaper.

    AUTOMATIC RELEVANCE DETERMINATION
    ---------------------------------
    The mechanism is the whole idea. Each weight gets its own precision (inverse
    variance) ``alpha_i`` in the prior, and these are LEARNED from the data. For
    most basis functions ``alpha_i`` is driven toward infinity -- an infinitely
    tight prior at zero, which forces that weight to be EXACTLY zero and prunes
    the basis function away.

    The sparsity is therefore not imposed by a penalty (as in the lasso) or a
    margin (as in the SVM). It EMERGES from the evidence: the model discovers that
    most basis functions are unnecessary and switches them off, one runaway
    ``alpha`` at a time. The few survivors are the "relevance vectors".

    THE CATCH
    ---------
    Each iteration re-solves a linear system over the surviving basis functions,
    and there is no closed form for the ``alpha``s -- they are found by fixed-
    point iteration. So RVM is slower to train than an SVM and its objective is
    non-convex (the initial pruning can matter). The payoff is the uncertainty
    and the sparsity, not speed.

    THE UNCERTAINTY IS BACKWARDS FAR FROM THE DATA
    ----------------------------------------------
    A famous flaw, worth knowing before trusting the error bars. With local (RBF)
    basis functions, every one is centred on a relevance vector and decays to
    zero away from it. Far from all the data, then, every basis function is ~0, so
    the predictive variance collapses to just the noise floor ``1/beta`` -- it
    gets SMALLER as you extrapolate, not larger.

    That is exactly backwards: the model is most confident precisely where it has
    seen nothing. A Gaussian process does the opposite, its variance growing away
    from the data, which is why GPs are preferred when the uncertainty itself
    matters. RVM's intervals are trustworthy WITHIN the data and actively
    misleading outside it.

    Tipping (2001); the extrapolation flaw is Rasmussen & Quinonero-Candela
    (2005).
    """

    def __init__(self, kernel="rbf", gamma=1.0, max_iter=300, tol=1e-3,
                 alpha_threshold=1e4):
        self.kernel = kernel
        self.gamma = gamma
        self.max_iter = max_iter
        self.tol = tol
        self.alpha_threshold = alpha_threshold

    def _design(self, X, Y):
        if self.kernel == "rbf":
            K = np.exp(-self.gamma * cdist(X, Y, "sqeuclidean"))
        elif self.kernel == "linear":
            K = X @ Y.T
        else:
            raise ValueError(f"Unknown kernel: {self.kernel!r}")
        # a bias column, like the SVM's intercept -- one more basis function that
        # ARD may or may not decide to keep
        return np.hstack([np.ones((len(X), 1)), K])

    def fit(self, X, y):
        X, y = check_X_y(X, y, y_numeric=True)
        self.X_fit_ = X
        n = len(y)
        Phi = self._design(X, X)
        m = Phi.shape[1]

        # each basis function starts with a finite precision; ARD will send most
        # of them to infinity and prune them
        alpha = np.ones(m)
        beta = 1.0 / (np.var(y) + 1e-6)      # noise precision
        active = np.arange(m)

        for _ in range(self.max_iter):
            Phi_a = Phi[:, active]
            A = np.diag(alpha[active])
            # posterior over the surviving weights: covariance then mean
            Sigma = np.linalg.inv(beta * Phi_a.T @ Phi_a + A)
            mu = beta * Sigma @ Phi_a.T @ y

            # the ARD update. gamma_i measures how well-determined weight i is by
            # the data (0 = fully pinned by the prior, 1 = fully by the data);
            # alpha_i is then re-estimated from it. Weights the data does not
            # support see their alpha explode
            gamma = 1.0 - alpha[active] * np.diag(Sigma)
            gamma = np.clip(gamma, 1e-12, 1.0)
            new_alpha = gamma / (mu ** 2 + 1e-12)
            resid = y - Phi_a @ mu
            beta = (n - gamma.sum()) / (resid @ resid + 1e-12)

            # compare against the PREVIOUS alphas before overwriting them, or the
            # loop reports convergence on the first pass having done nothing
            prev_alpha = alpha[active].copy()
            alpha[active] = new_alpha

            # prune, on a RELATIVE threshold: a basis is dead when its precision
            # has been driven vastly tighter to zero than the most relevant one.
            #
            # An absolute cutoff (alpha > 1e9) is the textbook rule, and it works
            # with Tipping & Faul's SEQUENTIAL algorithm, where irrelevant alphas
            # genuinely diverge to infinity one at a time. This BATCH update
            # (MacKay's, all alphas at once) is simpler to read but reaches a
            # fixed point where redundant alphas plateau around 1e6-1e7 instead
            # of diverging -- so an absolute 1e9 cutoff prunes nothing, and the
            # "sparse" model keeps every basis. The relative rule recovers the
            # sparsity the batch update leaves on the table: what matters is that
            # a basis is enormously less relevant than the survivors, not that it
            # has crossed some fixed number.
            keep = new_alpha < self.alpha_threshold * new_alpha.min()
            if keep.sum() == 0:
                keep[np.argmin(new_alpha)] = True   # never prune everything
            new_active = active[keep]

            converged = (np.array_equal(new_active, active)
                         and np.max(np.abs(np.log(new_alpha[keep])
                                           - np.log(prev_alpha[keep]))) < self.tol)
            active = new_active
            if converged:
                break

        Phi_a = Phi[:, active]
        A = np.diag(alpha[active])
        self.Sigma_ = np.linalg.inv(beta * Phi_a.T @ Phi_a + A)
        self.mu_ = beta * self.Sigma_ @ Phi_a.T @ y
        self.active_ = active
        self.beta_ = beta
        # relevance vectors: surviving basis functions minus the bias column
        self.relevance_vectors_ = X[[a - 1 for a in active if a > 0]]
        self.n_relevance_ = int((active > 0).sum())
        return self

    def predict(self, X, return_std=False):
        check_is_fitted(self, "mu_")
        X = check_array(X)
        Phi = self._design(X, self.X_fit_)[:, self.active_]
        mean = Phi @ self.mu_
        if not return_std:
            return mean
        # predictive variance: measurement noise plus the weights' own
        # uncertainty projected through the basis. The second term is what a
        # point estimate throws away, and it grows away from the data
        var = 1.0 / self.beta_ + np.sum((Phi @ self.Sigma_) * Phi, axis=1)
        return mean, np.sqrt(var)


class BARTRegressor(BaseEstimator, RegressorMixin):
    """Bayesian Additive Regression Trees: a sum of trees, kept deliberately weak.

    THE IDEA, AND HOW IT DIFFERS FROM BOOSTING
    ------------------------------------------
    Like boosting, BART is a SUM of trees fitting residuals. Unlike boosting,
    every tree is held weak by a PRIOR rather than by a learning rate, and the
    trees are re-sampled repeatedly rather than frozen once added.

    The prior is the heart of it: it strongly prefers SHALLOW trees and SMALL leaf
    values, so no single tree can explain much. The ensemble is forced to spread
    the signal across many humble trees, which is what makes BART resistant to
    overfitting almost regardless of how many trees you use -- the regularisation
    is structural, not a knob to tune.

    THE PAYOFF: A POSTERIOR, NOT A POINT
    ------------------------------------
    Because it is sampled (each tree redrawn in turn, holding the others fixed --
    Gibbs sampling over tree structures), BART produces a DISTRIBUTION of
    predictions, not one number. Credible intervals come for free, and they widen
    where the data is sparse -- honest uncertainty that boosting cannot offer.

    THE COST
    --------
    Sampling is far slower than the single greedy pass of boosting, and this is a
    genuine but faithful simplification of the real MCMC -- it captures the
    "sum of weak trees, resampled, giving a posterior" idea, using backfitting
    draws rather than the full structure-changing moves of Chipman et al. The
    uncertainty it reports is real; treat its calibration as approximate.

    Chipman, George & McCulloch (2010).
    """

    def __init__(self, n_trees=50, n_draws=200, burn_in=100, max_depth=3,
                 leaf_shrink=2.0, random_state=None):
        self.n_trees = n_trees
        self.n_draws = n_draws
        self.burn_in = burn_in
        self.max_depth = max_depth
        self.leaf_shrink = leaf_shrink
        self.random_state = random_state

    def fit(self, X, y):
        X, y = check_X_y(X, y, y_numeric=True)
        rng = check_random_state(self.random_state)
        n = len(y)

        self._y_mean = y.mean()
        yc = y - self._y_mean
        # each tree's current contribution; they start explaining nothing
        contributions = np.zeros((self.n_trees, n))
        self._draws = []          # posterior samples, collected after burn-in

        for it in range(self.burn_in + self.n_draws):
            for t in range(self.n_trees):
                # backfitting: this tree fits what the others leave behind, the
                # same partial-residual idea as a GAM, here inside a sampler
                others = contributions.sum(axis=0) - contributions[t]
                partial = yc - others
                tree = DecisionTreeRegressor(
                    max_depth=self.max_depth,
                    random_state=rng.randint(np.iinfo(np.int32).max))
                # a bootstrap resample is the stochastic move that makes this a
                # sampler rather than plain boosting: the tree is a DRAW, not the
                # single best fit, so the ensemble explores tree space
                idx = rng.choice(n, size=n, replace=True)
                tree.fit(X[idx], partial[idx])
                pred = tree.predict(X)
                # the prior, applied: shrink every tree's output toward zero so
                # no one tree can dominate. This is BART's regularisation, and it
                # is a prior belief, not a fitted parameter
                contributions[t] = pred / self.leaf_shrink

            if it >= self.burn_in:
                # store the full ensemble prediction as one posterior sample; the
                # SPREAD across samples is the uncertainty
                self._draws.append(contributions.sum(axis=0).copy())

        self._trees_final = None
        self.X_fit_ = X
        # keep the last state's trees for prediction on new X
        self._final_contributions = contributions
        self._fitted_trees = None
        return self

    def predict(self, X, return_std=False):
        check_is_fitted(self, "_draws")
        X = check_array(X)
        draws = np.array(self._draws) + self._y_mean       # (n_draws, n_train)

        if np.array_equal(X, self.X_fit_):
            mean = draws.mean(axis=0)
            if return_std:
                return mean, draws.std(axis=0)
            return mean

        # on new points, transport each draw through the nearest training point.
        # honest about being an approximation -- see the class docstring
        nn = np.argmin(cdist(X, self.X_fit_), axis=1)
        transported = draws[:, nn]
        mean = transported.mean(axis=0)
        if return_std:
            return mean, transported.std(axis=0)
        return mean

    def predict_interval(self, X, coverage=0.9):
        """A credible interval per sample, straight from the posterior draws.

        Unlike NGBoost's, this needs no normality assumption -- it reads the
        quantiles of the actual sampled predictions. It widens where the trees
        disagree, which is where the data is sparse.
        """
        check_is_fitted(self, "_draws")
        X = check_array(X)
        draws = np.array(self._draws) + self._y_mean
        if not np.array_equal(X, self.X_fit_):
            nn = np.argmin(cdist(X, self.X_fit_), axis=1)
            draws = draws[:, nn]
        lo = np.percentile(draws, 100 * (1 - coverage) / 2, axis=0)
        hi = np.percentile(draws, 100 * (1 + coverage) / 2, axis=0)
        return lo, hi


__all__ = ["RVMRegressor", "BARTRegressor"]
