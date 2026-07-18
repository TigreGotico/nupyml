"""PGM v2: approximate inference on loopy graphs, and continuous Bayes nets.

Exact inference (``variable_elimination``, tree ``belief_propagation``) is only
tractable when the graph is a tree or has small treewidth. ``loopy_belief_propagation``
runs the same message-passing on a graph WITH cycles -- no longer exact, but
often an excellent approximation, and the workhorse behind error-correcting codes
and computer vision. ``GaussianBayesianNetwork`` leaves the discrete world
entirely: each variable is a linear-Gaussian function of its parents, so the whole
network is one big multivariate Gaussian you can condition and query in closed form.
"""
import numpy as np

from ..base import BaseEstimator
from ._factor import Factor


def loopy_belief_propagation(factors, query, evidence=None, max_iter=50,
                             tol=1e-4, damping=0.0):
    """Approximate marginals by passing messages until they stop changing.

    Build a factor graph (variables on one side, factors on the other) and
    iterate: each variable tells each neighbouring factor the product of what its
    OTHER factors said; each factor tells each neighbouring variable the product
    of its potential with what its OTHER variables said, summed over everything
    else. On a tree this converges in one sweep to the exact answer; on a LOOPY
    graph it usually still converges, to a good approximation -- the basis of
    turbo/LDPC decoding. ``damping`` blends each new message with the old to help
    convergence oscillating cases.
    """
    evidence = evidence or {}
    factors = [f.reduce(evidence) if any(v in evidence for v in f.variables)
               else f.copy() for f in factors]
    factors = [f for f in factors if f.variables]        # drop fully-observed
    variables = sorted({v for f in factors for v in f.variables})
    card = {}
    for f in factors:
        card.update(dict(zip(f.variables, f.cardinalities)))

    # messages: var->factor and factor->var, initialised uniform
    m_vf = {(v, fi): np.ones(card[v]) for fi, f in enumerate(factors)
            for v in f.variables}
    m_fv = {(fi, v): np.ones(card[v]) for fi, f in enumerate(factors)
            for v in f.variables}

    def normalize(x):
        s = x.sum()
        return x / s if s > 0 else x

    for _ in range(max_iter):
        max_delta = 0.0
        # variable -> factor: product of incoming factor messages except target
        for (v, fi), _old in list(m_vf.items()):
            msg = np.ones(card[v])
            for fj, f in enumerate(factors):
                if v in f.variables and fj != fi:
                    msg = msg * m_fv[(fj, v)]
            m_vf[(v, fi)] = normalize(msg)
        # factor -> variable: multiply potential by other vars' messages, sum out
        for (fi, v), old in list(m_fv.items()):
            f = factors[fi]
            belief = f.copy()
            for u in f.variables:
                if u != v:
                    fac = Factor((u,), (card[u],), m_vf[(u, fi)])
                    belief = belief.multiply(fac)
            marg = belief.marginalize([u for u in f.variables if u != v])
            new = normalize(marg._align((v,)).ravel())
            if damping:
                new = normalize((1 - damping) * new + damping * old)
            max_delta = max(max_delta, np.abs(new - old).max())
            m_fv[(fi, v)] = new
        if max_delta < tol:
            break

    # belief at the query variable = product of all its incoming factor messages
    b = np.ones(card[query])
    for fi, f in enumerate(factors):
        if query in f.variables:
            b = b * m_fv[(fi, query)]
    return normalize(b)


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


__all__ = ["loopy_belief_propagation", "GaussianBayesianNetwork"]
