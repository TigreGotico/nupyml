"""Matrix factorisation with a POSTERIOR on the factors (Salakhutdinov, 2008)."""
import numpy as np
from ..base import BaseEstimator
from ..utils import check_array, check_random_state


class BayesianPMF(BaseEstimator):
    """Matrix factorisation with a POSTERIOR on the factors (Salakhutdinov, 2008).

    Ordinary matrix factorisation gives point estimates of the user/item factors and
    so cannot say how uncertain a prediction is -- risky for a cold-start user with
    one rating. Bayesian PMF places Gaussian priors on the factors and samples their
    POSTERIOR by Gibbs sampling (each user/item factor is Gaussian given the others),
    averaging predictions over the samples. The result is a predictive DISTRIBUTION:
    the mean is the rating, and its spread widens for users/items with little data.
    Works on a dense rating matrix with NaN for missing entries.
    """

    def __init__(self, n_factors=5, n_samples=50, burn_in=20, alpha=2.0,
                 random_state=None):
        self.n_factors = n_factors
        self.n_samples = n_samples
        self.burn_in = burn_in
        self.alpha = alpha                                 # observation precision
        self.random_state = random_state

    def fit(self, R):
        R = np.asarray(R, float)
        rng = check_random_state(self.random_state)
        n, m = R.shape
        k = self.n_factors
        mask = ~np.isnan(R)
        U = rng.randn(n, k) * 0.1
        V = rng.randn(m, k) * 0.1
        preds = []
        for it in range(self.n_samples + self.burn_in):
            U = self._sample_factors(R, mask, V, rng)      # p(U | V, R)
            V = self._sample_factors(R.T, mask.T, U, rng)  # p(V | U, R)
            if it >= self.burn_in:
                preds.append(U @ V.T)
        self.samples_ = np.array(preds)
        self.mean_ = self.samples_.mean(axis=0)
        self.std_ = self.samples_.std(axis=0)
        return self

    def _sample_factors(self, R, mask, W, rng):
        n, k = R.shape[0], W.shape[1]
        out = np.zeros((n, k))
        prior = np.eye(k)
        for i in range(n):
            obs = mask[i]
            if not obs.any():
                out[i] = rng.multivariate_normal(np.zeros(k), prior)
                continue
            Wi = W[obs]
            prec = prior + self.alpha * Wi.T @ Wi          # posterior precision
            cov = np.linalg.inv(prec)
            mean = cov @ (self.alpha * Wi.T @ R[i, obs])
            out[i] = rng.multivariate_normal(mean, cov)
        return out

    def predict(self, i, j):
        return self.mean_[i, j], self.std_[i, j]


__all__ = ["BayesianPMF"]
