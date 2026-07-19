"""A Bayes net of linear-Gaussian variables -- inference in closed form."""
import numpy as np
from ..base import BaseEstimator


class GaussianBayesianNetwork(BaseEstimator):
    """A Bayes net of linear-Gaussian variables -- inference in closed form.

    Each variable is ``x_i = b_i + sum_j w_ij x_j + noise`` over its parents ``j``
    (a linear regression with Gaussian noise). Because linear combinations of
    Gaussians are Gaussian, the WHOLE network collapses to a single multivariate
    Gaussian, whose mean and covariance this computes from the local coefficients.
    Then any conditional query -- "given these variables, what is the distribution
    of those?" -- is the standard Gaussian conditioning formula, exact and cheap,
    with no sampling. Fit learns each node's coefficients by regression on its
    parents.
    """

    def __init__(self, structure):
        # structure: dict node -> list of parent nodes (a DAG)
        self.structure = structure

    def fit(self, data):
        # data: dict node -> 1-D array (or 2-D array with .columns via names list)
        self.nodes_ = list(self.structure.keys())
        self.coef_, self.bias_, self.noise_var_ = {}, {}, {}
        for node in self.nodes_:
            parents = self.structure[node]
            y = np.asarray(data[node], dtype=float)
            if parents:
                Xp = np.column_stack([data[p] for p in parents])
                Xd = np.column_stack([np.ones(len(y)), Xp])
                coef, *_ = np.linalg.lstsq(Xd, y, rcond=None)
                self.bias_[node] = coef[0]
                self.coef_[node] = dict(zip(parents, coef[1:]))
                resid = y - Xd @ coef
            else:
                self.bias_[node] = y.mean()
                self.coef_[node] = {}
                resid = y - y.mean()
            self.noise_var_[node] = max(resid.var(), 1e-9)
        self._build_joint()
        return self

    def _build_joint(self):
        # solve x = b + W x + eps  ->  x = (I-W)^{-1}(b + eps)
        n = len(self.nodes_)
        idx = {node: i for i, node in enumerate(self.nodes_)}
        W = np.zeros((n, n))
        b = np.zeros(n)
        for node in self.nodes_:
            b[idx[node]] = self.bias_[node]
            for p, w in self.coef_[node].items():
                W[idx[node], idx[p]] = w
        M = np.linalg.inv(np.eye(n) - W)
        self.mean_ = M @ b
        noise = np.diag([self.noise_var_[node] for node in self.nodes_])
        self.cov_ = M @ noise @ M.T
        self._idx = idx

    def marginal(self, node):
        i = self._idx[node]
        return self.mean_[i], self.cov_[i, i]

    def condition(self, evidence):
        """Return (mean, cov) over the unobserved nodes given observed values."""
        obs = list(evidence)
        hidden = [n for n in self.nodes_ if n not in evidence]
        oi = [self._idx[n] for n in obs]
        hi = [self._idx[n] for n in hidden]
        mu_h, mu_o = self.mean_[hi], self.mean_[oi]
        Coo = self.cov_[np.ix_(oi, oi)]
        Cho = self.cov_[np.ix_(hi, oi)]
        Chh = self.cov_[np.ix_(hi, hi)]
        xo = np.array([evidence[n] for n in obs]) - mu_o
        gain = Cho @ np.linalg.inv(Coo)
        mean = mu_h + gain @ xo
        cov = Chh - gain @ Cho.T
        return dict(zip(hidden, mean)), cov


__all__ = ["GaussianBayesianNetwork"]
