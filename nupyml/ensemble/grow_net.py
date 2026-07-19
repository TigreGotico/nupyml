"""Gradient boosting of shallow NEURAL nets (Badirli et al., 2020)."""
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


class GrowNet(BaseEstimator, RegressorMixin):
    """Gradient boosting of shallow NEURAL nets (Badirli et al., 2020).

    Gradient boosting normally adds trees; GrowNet adds small NEURAL networks. Each
    weak net is trained on the current gradient (the residual for squared loss) and
    added with a learning rate, so the ensemble descends the loss in function space
    just like tree boosting -- but the weak learners are differentiable and can share
    features with the running prediction. It closes much of the gap to deep networks
    on tabular data while staying an additive, stage-wise model. Squared-loss
    regression here.
    """

    def __init__(self, n_stages=20, hidden=16, boost_lr=0.3, epochs=60, lr=0.02,
                 random_state=None):
        self.n_stages = n_stages
        self.hidden = hidden
        self.boost_lr = boost_lr
        self.epochs = epochs
        self.lr = lr
        self.random_state = random_state

    def fit(self, X, y):
        from ..autograd import Tensor
        X = check_array(X); y = np.asarray(y, float)
        rng = check_random_state(self.random_state)
        self.init_ = y.mean()
        pred = np.full(len(y), self.init_)
        self.stages_ = []
        dims = [X.shape[1], self.hidden, 1]
        xt = Tensor(X)
        for _ in range(self.n_stages):
            residual = y - pred                          # negative gradient (sq loss)
            net = _mlp_params(dims, rng)
            params = [p for l in net for p in l.parameters()]
            rt = Tensor(residual.reshape(-1, 1))
            for _ in range(self.epochs):
                out = _mlp_forward(net, xt)
                loss = ((out - rt) ** 2).mean()
                for p in params:
                    p.zero_grad()
                loss.backward()
                for p in params:
                    p.data -= self.lr * p.grad
            step = _mlp_forward(net, xt).data.ravel()
            pred = pred + self.boost_lr * step
            self.stages_.append(net)
        return self

    def predict(self, X):
        from ..autograd import Tensor
        X = check_array(X)
        xt = Tensor(X)
        pred = np.full(len(X), self.init_)
        for net in self.stages_:
            pred = pred + self.boost_lr * _mlp_forward(net, xt).data.ravel()
        return pred


__all__ = ["GrowNet"]
