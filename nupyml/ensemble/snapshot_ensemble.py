"""Many models for the price of ONE training run (Huang et al., 2017)."""
import numpy as np
from ..base import (BaseEstimator, ClassifierMixin, RegressorMixin, clone)
from ..utils import check_array, check_random_state


def _mlp_params(dims, rng):
    from ..nn import Linear
    return [Linear(dims[i], dims[i + 1], rng=rng) for i in range(len(dims) - 1)]


def _mlp_forward(layers, x):
    h = x
    for i, l in enumerate(layers):
        h = l(h)
        if i < len(layers) - 1:
            h = h.relu()
    return h


class SnapshotEnsemble(BaseEstimator, RegressorMixin):
    """Many models for the price of ONE training run (Huang et al., 2017).

    Ensembling usually means training N models from scratch. A snapshot ensemble
    trains ONE model but with a CYCLIC learning rate: each cycle anneals the rate to
    near zero (converging to a local minimum), a SNAPSHOT of the weights is saved,
    then the rate jumps back up to escape that minimum toward another. Because SGD
    with restarts visits several diverse minima, averaging those snapshots gives an
    ensemble almost for free -- the diversity comes from the optimisation path, not
    from separate runs. Snapshots of a small MLP regressor.
    """

    def __init__(self, n_snapshots=5, hidden=32, cycle_epochs=80, lr_max=0.05,
                 random_state=None):
        self.n_snapshots = n_snapshots
        self.hidden = hidden
        self.cycle_epochs = cycle_epochs
        self.lr_max = lr_max
        self.random_state = random_state

    def fit(self, X, y):
        from ..autograd import Tensor
        X = check_array(X); y = np.asarray(y, float).reshape(-1, 1)
        rng = check_random_state(self.random_state)
        dims = [X.shape[1], self.hidden, 1]
        net = _mlp_params(dims, rng)
        params = [p for l in net for p in l.parameters()]
        xt, yt = Tensor(X), Tensor(y)
        self.snapshots_ = []
        for _ in range(self.n_snapshots):
            for e in range(self.cycle_epochs):
                # cosine cyclic learning rate: high -> ~0 within each cycle
                lr = 0.5 * self.lr_max * (1 + np.cos(np.pi * e / self.cycle_epochs))
                out = _mlp_forward(net, xt)
                loss = ((out - yt) ** 2).mean()
                for p in params:
                    p.zero_grad()
                loss.backward()
                for p in params:
                    p.data -= lr * p.grad
            self.snapshots_.append([l.weight.data.copy() for l in net] +
                                   [l.bias.data.copy() for l in net])
            self._net = net
        return self

    def _predict_with(self, snapshot, X):
        from ..autograd import Tensor
        net = self._net
        n = len(net)
        for i, l in enumerate(net):
            l.weight.data[...] = snapshot[i]
            l.bias.data[...] = snapshot[n + i]
        return _mlp_forward(net, Tensor(X)).data.ravel()

    def predict(self, X):
        X = check_array(X)
        return np.mean([self._predict_with(s, X) for s in self.snapshots_], axis=0)


__all__ = ["SnapshotEnsemble"]
