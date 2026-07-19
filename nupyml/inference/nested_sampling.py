"""Nested sampling: compute the Bayesian evidence by shrinking the prior."""
import numpy as np

from ..base import BaseEstimator
from ..utils import check_random_state


class NestedSampling(BaseEstimator):
    """Compute the Bayesian EVIDENCE by shrinking the prior (Skilling, 2004).

    The evidence ``Z = ∫ L(θ) π(θ) dθ`` -- the normalising constant that model
    comparison needs -- is a nightmare integral in high dimensions. Nested sampling
    reparametrises it as a ONE-dimensional integral over the prior mass enclosed above
    each likelihood level. It keeps ``n_live`` points sampled from the prior, repeatedly
    removes the LOWEST-likelihood one (banking its shrinking slice of prior mass as
    evidence) and replaces it with a fresh point of higher likelihood. The accumulated
    slices give ``Z``, and the discarded points, weighted, give posterior samples.
    ``log_likelihood(theta)`` and a box prior.
    """

    def __init__(self, log_likelihood, bounds, n_live=100, max_iter=2000,
                 random_state=None):
        self.log_likelihood = log_likelihood
        self.bounds = bounds
        self.n_live = n_live
        self.max_iter = max_iter
        self.random_state = random_state

    def run(self):
        rng = check_random_state(self.random_state)
        lo, hi = np.asarray(self.bounds[0], float), np.asarray(self.bounds[1], float)
        d = len(np.atleast_1d(lo))
        live = rng.uniform(lo, hi, (self.n_live, d))
        live_ll = np.array([self.log_likelihood(x) for x in live])
        logZ = -np.inf
        log_width = np.log(1.0 - np.exp(-1.0 / self.n_live))
        samples, weights = [], []
        for i in range(self.max_iter):
            worst = live_ll.argmin()
            Lworst = live_ll[worst]
            logwt = log_width - i / self.n_live + Lworst   # weight of this slice
            logZ = np.logaddexp(logZ, logwt)
            samples.append(live[worst].copy()); weights.append(logwt)
            # replace the worst with a higher-likelihood draw from the prior
            for _ in range(200):
                cand = rng.uniform(lo, hi, d)
                if self.log_likelihood(cand) > Lworst:
                    live[worst] = cand
                    live_ll[worst] = self.log_likelihood(cand)
                    break
            if logwt < logZ - 8:                           # remaining mass negligible
                break
        self.log_evidence_ = logZ
        self.samples_ = np.array(samples)
        w = np.exp(np.array(weights) - logZ)
        self.posterior_weights_ = w / w.sum()
        self.posterior_mean_ = (self.posterior_weights_[:, None]
                                * self.samples_).sum(axis=0)
        return self


__all__ = ["NestedSampling"]
