"""Gradient-based meta-learning: learn an INITIALISATION that adapts fast.

Both algorithms answer the same question -- "what starting weights let a few
gradient steps solve any task in this family?" -- and both are first-order, so
they need no second derivatives. Reptile just moves the initialisation toward each
task's fine-tuned weights. MAML (first order) moves it along the gradient of the
QUERY loss measured AFTER adapting, which ties the initialisation more directly to
post-adaptation performance. The demo model is a tanh MLP regressor; a task is a
function to fit from a few points.
"""
import numpy as np

from ..utils import check_random_state


class _MLP:
    """A tiny tanh MLP with explicit forward/backward, so params are plain arrays."""

    def __init__(self, in_dim, hidden, rng):
        self.p = {
            "W1": rng.randn(in_dim, hidden) * 0.1, "b1": np.zeros(hidden),
            "W2": rng.randn(hidden, 1) * 0.1, "b2": np.zeros(1),
        }

    def forward(self, X):
        z1 = X @ self.p["W1"] + self.p["b1"]
        h = np.tanh(z1)
        out = h @ self.p["W2"] + self.p["b2"]
        return out, (X, z1, h)

    def grads(self, X, y):
        out, (X, z1, h) = self.forward(X)
        y = y.reshape(-1, 1)
        d_out = 2.0 * (out - y) / len(X)                 # d MSE / d out
        g = {}
        g["W2"] = h.T @ d_out; g["b2"] = d_out.sum(axis=0)
        dh = d_out @ self.p["W2"].T * (1 - h ** 2)
        g["W1"] = X.T @ dh; g["b1"] = dh.sum(axis=0)
        return g

    def sgd(self, X, y, lr, steps):
        for _ in range(steps):
            g = self.grads(X, y)
            for k in self.p:
                self.p[k] -= lr * g[k]

    def loss(self, X, y):
        out, _ = self.forward(X)
        return float(np.mean((out.ravel() - y.ravel()) ** 2))

    def copy_params(self):
        return {k: v.copy() for k, v in self.p.items()}

    def set_params(self, p):
        self.p = {k: v.copy() for k, v in p.items()}


class _MetaBase:
    def __init__(self, hidden=40, in_dim=1, inner_lr=0.02, inner_steps=5,
                 meta_lr=0.1, random_state=None):
        self.hidden = hidden; self.in_dim = in_dim
        self.inner_lr = inner_lr; self.inner_steps = inner_steps
        self.meta_lr = meta_lr; self.random_state = random_state

    def _init_model(self):
        rng = check_random_state(self.random_state)
        self.model_ = _MLP(self.in_dim, self.hidden, rng)
        return self.model_

    def adapt(self, X, y, steps=None):
        """Fine-tune a COPY of the meta-initialisation on a new task's few points."""
        X = np.atleast_2d(np.asarray(X, float))
        if X.shape[0] == 1 and X.shape[1] != self.in_dim:
            X = X.T
        m = _MLP(self.in_dim, self.hidden, np.random.RandomState(0))
        m.set_params(self.meta_params_)
        m.sgd(X, np.asarray(y, float), self.inner_lr,
              steps or self.inner_steps)
        self.adapted_ = m
        return self

    def predict(self, X):
        X = np.atleast_2d(np.asarray(X, float))
        if X.shape[0] == 1 and X.shape[1] != self.in_dim:
            X = X.T
        return self.adapted_.forward(X)[0].ravel()


class Reptile(_MetaBase):
    """Move the initialisation toward each task's fine-tuned weights (Nichol, 2018).

    The whole algorithm: sample a task, SGD on it for a few steps to get adapted
    weights, then nudge the meta-initialisation a fraction of the way toward those
    weights. Repeated over many tasks, the initialisation drifts to a point that is
    a short hop from the solution of EVERY task -- no meta-gradient, no second
    derivatives, just "average of where fine-tuning wants to go". Startlingly, this
    matches MAML on many benchmarks.
    """

    def meta_train(self, task_sampler, n_iterations=1000):
        self._init_model()
        self.meta_params_ = self.model_.copy_params()
        for _ in range(n_iterations):
            X, y = task_sampler()
            X = np.atleast_2d(np.asarray(X, float))
            if X.shape[0] == 1 and X.shape[1] != self.in_dim:
                X = X.T
            m = _MLP(self.in_dim, self.hidden, np.random.RandomState(0))
            m.set_params(self.meta_params_)
            m.sgd(X, np.asarray(y, float), self.inner_lr, self.inner_steps)
            for k in self.meta_params_:                  # step toward adapted params
                self.meta_params_[k] += self.meta_lr * (m.p[k] - self.meta_params_[k])
        return self


class MAML(_MetaBase):
    """First-order MAML: optimise the init for post-adaptation loss (Finn, 2017).

    MAML also adapts on a support set, but its meta-update is the gradient of the
    loss on a separate QUERY set, measured AFTER adaptation -- "make the init such
    that one adaptation step generalises". The true version differentiates through
    the adaptation (a second derivative); the first-order variant reuses the
    adapted-point gradient, which is far cheaper and almost as good. That query
    gradient is what distinguishes it from Reptile's parameter averaging.
    """

    def meta_train(self, task_sampler, n_iterations=1000):
        self._init_model()
        self.meta_params_ = self.model_.copy_params()
        for _ in range(n_iterations):
            X, y = task_sampler()
            X = np.atleast_2d(np.asarray(X, float))
            if X.shape[0] == 1 and X.shape[1] != self.in_dim:
                X = X.T
            y = np.asarray(y, float)
            n = len(X)
            half = max(1, n // 2)
            xs, ys, xq, yq = X[:half], y[:half], X[half:], y[half:]
            if len(xq) == 0:
                xq, yq = xs, ys
            m = _MLP(self.in_dim, self.hidden, np.random.RandomState(0))
            m.set_params(self.meta_params_)
            m.sgd(xs, ys, self.inner_lr, self.inner_steps)   # inner adaptation
            gq = m.grads(xq, yq)                              # query-loss gradient
            for k in self.meta_params_:                       # first-order meta step
                self.meta_params_[k] -= self.meta_lr * gq[k]
        return self


__all__ = ["Reptile", "MAML"]
