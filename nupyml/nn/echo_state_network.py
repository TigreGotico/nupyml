"""A random recurrent RESERVOIR with only the readout trained (Jaeger, 2001)."""
import numpy as np
from ..base import BaseEstimator
from ..utils import check_random_state


class EchoStateNetwork(BaseEstimator):
    """A random recurrent RESERVOIR with only the readout trained (Jaeger, 2001).

    Training recurrent nets is hard (backprop through time, vanishing gradients).
    Reservoir computing sidesteps it entirely: fix a large RANDOM recurrent network
    -- the "reservoir" -- whose rich nonlinear dynamics echo the input history, and
    train ONLY a linear readout from its state by ridge regression. No
    backpropagation, closed-form fit, yet it models complex temporal systems. The
    one thing that must be tuned is the reservoir's SPECTRAL RADIUS (< 1 for the
    "echo state property": past inputs must eventually fade). Fits ``X: (T, in)``
    to ``y: (T, out)`` and forecasts.
    """

    def __init__(self, n_reservoir=200, spectral_radius=0.95, sparsity=0.1,
                 leak=1.0, ridge=1e-6, washout=10, random_state=None):
        self.n_reservoir = n_reservoir
        self.spectral_radius = spectral_radius
        self.sparsity = sparsity
        self.leak = leak
        self.ridge = ridge
        self.washout = washout
        self.random_state = random_state

    def _init_reservoir(self, n_in, rng):
        self.W_in_ = rng.uniform(-1, 1, (self.n_reservoir, n_in))
        W = rng.uniform(-1, 1, (self.n_reservoir, self.n_reservoir))
        mask = rng.rand(*W.shape) > self.sparsity
        W[mask] = 0.0
        radius = np.max(np.abs(np.linalg.eigvals(W)))
        self.W_res_ = W * (self.spectral_radius / (radius + 1e-12))

    def _run(self, X):
        T = len(X)
        state = np.zeros(self.n_reservoir)
        states = np.empty((T, self.n_reservoir))
        for t in range(T):
            pre = self.W_in_ @ X[t] + self.W_res_ @ state
            state = (1 - self.leak) * state + self.leak * np.tanh(pre)
            states[t] = state
        return states

    def fit(self, X, y):
        X = np.atleast_2d(np.asarray(X, float))
        if X.shape[0] == 1:
            X = X.T
        y = np.asarray(y, float)
        Y = y.reshape(len(y), -1)
        rng = check_random_state(self.random_state)
        self._init_reservoir(X.shape[1], rng)
        states = self._run(X)
        S = np.hstack([states, np.ones((len(states), 1))])[self.washout:]
        Yt = Y[self.washout:]
        A = S.T @ S + self.ridge * np.eye(S.shape[1])   # ridge-regressed readout
        self.W_out_ = np.linalg.solve(A, S.T @ Yt)
        self._last_state = states[-1]
        return self

    def predict(self, X):
        X = np.atleast_2d(np.asarray(X, float))
        if X.shape[0] == 1:
            X = X.T
        states = self._run(X)
        S = np.hstack([states, np.ones((len(states), 1))])
        out = S @ self.W_out_
        return out.ravel() if out.shape[1] == 1 else out


__all__ = ["EchoStateNetwork"]
