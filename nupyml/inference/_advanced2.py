"""Probabilistic v3: sparse Bayesian kernel machines and dropout-based Bayesian
neural networks.

Two routes to calibrated predictions with uncertainty. The Relevance Vector
Machine is the sparse-Bayesian cousin of the SVM: an evidence-maximising prior
prunes almost every basis function, leaving a few "relevance vectors" and a full
predictive distribution. MC-dropout reinterprets a dropout net as an approximate
Bayesian model, so its predictions come with an honest error bar -- for free.
"""
import numpy as np
from scipy.spatial.distance import cdist

from ..base import BaseEstimator, RegressorMixin
from ..utils import check_array, check_random_state


class RelevanceVectorMachine(BaseEstimator, RegressorMixin):
    """A sparse Bayesian kernel regressor with a predictive distribution
    (Tipping, 2001).

    The SVM gives a sparse predictor but no probabilities, and its sparsity is a
    side effect of the hinge loss. The RVM starts from a Bayesian model with an
    INDIVIDUAL precision (``alpha_i``) on every basis weight and maximises the
    evidence for those precisions. Most ``alpha_i`` diverge to infinity, forcing
    their weight to exactly zero -- so the model keeps only a handful of RELEVANCE
    VECTORS, usually far fewer than the SVM's support vectors, and yields a full
    predictive mean AND variance. Kernel (RBF) basis; ``predict(..., return_std=True)``
    gives the error bars.
    """

    def __init__(self, gamma=1.0, max_iter=500, tol=1e-5, threshold=1e5):
        self.gamma = gamma
        self.max_iter = max_iter
        self.tol = tol
        self.threshold = threshold

    def _design(self, X, Xr):
        K = np.exp(-self.gamma * cdist(X, Xr, "sqeuclidean"))
        return np.column_stack([np.ones(len(X)), K])       # + bias column

    def fit(self, X, y):
        X = check_array(X); y = np.asarray(y, float).ravel()
        self.X_train_ = X
        Phi = self._design(X, X)
        n, m = Phi.shape
        alpha = np.ones(m)
        beta = 1.0 / (np.var(y) + 1e-6)
        active = np.ones(m, bool)
        for _ in range(self.max_iter):
            Pa = Phi[:, active]
            A = np.diag(alpha[active])
            Sigma = np.linalg.inv(A + beta * Pa.T @ Pa)
            mu = beta * Sigma @ Pa.T @ y
            gamma_i = 1.0 - alpha[active] * np.diag(Sigma)
            old = alpha[active].copy()
            new_alpha = gamma_i / (mu ** 2 + 1e-12)
            alpha[active] = new_alpha
            rss = np.sum((y - Pa @ mu) ** 2)
            beta = (n - gamma_i.sum()) / (rss + 1e-12)
            active = alpha < self.threshold                # prune diverged weights
            if not active.any():
                active[0] = True
            if np.max(np.abs(np.log(new_alpha + 1e-12)
                             - np.log(old + 1e-12))) < self.tol:
                break
        self.active_ = active
        self.alpha_, self.beta_ = alpha, beta
        Pa = Phi[:, active]
        self.Sigma_ = np.linalg.inv(np.diag(alpha[active]) + beta * Pa.T @ Pa)
        self.mu_ = beta * self.Sigma_ @ Pa.T @ y
        # relevance vectors = retained training points (drop the bias column)
        rv = np.where(active[1:])[0]
        self.relevance_vectors_ = X[rv]
        self.n_relevance_ = len(rv)
        return self

    def predict(self, X, return_std=False):
        Phi = self._design(check_array(X), self.X_train_)[:, self.active_]
        mean = Phi @ self.mu_
        if not return_std:
            return mean
        var = 1.0 / self.beta_ + np.einsum("ij,jk,ik->i", Phi, self.Sigma_, Phi)
        return mean, np.sqrt(var)


class BayesianNeuralNetwork(BaseEstimator, RegressorMixin):
    """A dropout net read as an approximate BAYESIAN model (Gal & Ghahramani, 2016).

    Dropout is normally a training-time regulariser switched off at test time.
    Gal & Ghahramani showed that KEEPING dropout on at test time and averaging many
    stochastic forward passes is a valid approximation to Bayesian inference over
    the weights -- each pass is a sample from an approximate posterior. So a plain
    dropout MLP becomes a model that reports predictive UNCERTAINTY: the spread of
    its Monte-Carlo predictions widens where data is scarce (extrapolation) and
    narrows where it is dense. No extra parameters, just don't turn dropout off.
    """

    def __init__(self, hidden=(32, 32), dropout=0.1, epochs=400, lr=0.01,
                 n_samples=50, weight_decay=1e-4, random_state=None):
        self.hidden = hidden
        self.dropout = dropout
        self.epochs = epochs
        self.lr = lr
        self.n_samples = n_samples
        self.weight_decay = weight_decay
        self.random_state = random_state

    def fit(self, X, y):
        from ..nn import Linear
        from ..autograd import Tensor
        X = check_array(X); y = np.asarray(y, float).reshape(-1, 1)
        rng = check_random_state(self.random_state)
        dims = [X.shape[1], *self.hidden, 1]
        self.layers_ = [Linear(dims[i], dims[i + 1], rng=rng)
                        for i in range(len(dims) - 1)]
        self._rng = rng
        params = [p for l in self.layers_ for p in l.parameters()]
        xt = Tensor(X); yt = Tensor(y)
        for _ in range(self.epochs):
            out = self._forward(xt, train=True)
            mse = ((out - yt) ** 2).mean()
            reg = sum((p * p).sum() for p in params) * self.weight_decay
            loss = mse + reg
            for p in params:
                p.zero_grad()
            loss.backward()
            for p in params:
                p.data -= self.lr * p.grad
        return self

    def _forward(self, x, train):
        from ..autograd import Tensor
        h = x
        for i, layer in enumerate(self.layers_):
            h = layer(h)
            if i < len(self.layers_) - 1:
                h = h.relu()
                if train or self.dropout > 0:            # dropout stays ON for MC
                    mask = (self._rng.rand(*h.shape) > self.dropout) / \
                        (1 - self.dropout)
                    h = h * Tensor(mask)
        return h

    def predict(self, X, return_std=False):
        from ..autograd import Tensor
        X = check_array(X)
        xt = Tensor(X)
        # Monte-Carlo: many stochastic passes with dropout left on
        preds = np.stack([self._forward(xt, train=True).data.ravel()
                          for _ in range(self.n_samples)])
        mean = preds.mean(axis=0)
        if not return_std:
            return mean
        return mean, preds.std(axis=0)


__all__ = ["RelevanceVectorMachine", "BayesianNeuralNetwork"]
