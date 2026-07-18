"""Ensembles v4: feature-subspace bagging, diversity-penalised nets, boosted nets,
and cyclic-snapshot ensembles.

Four ways to build diverse ensembles. The random subspace method decorrelates
members by giving each a different FEATURE subset. Negative correlation learning
trains neural members JOINTLY with a penalty that pushes them apart. GrowNet boosts
shallow NEURAL nets instead of trees. Snapshot ensembles get many models for the
price of one training run by saving cyclic-learning-rate minima.
"""
import numpy as np

from ..base import (BaseEstimator, ClassifierMixin, RegressorMixin, clone)
from ..utils import check_array, check_random_state
from ..tree import DecisionTreeClassifier


class RandomSubspaceClassifier(BaseEstimator, ClassifierMixin):
    """Decorrelate members by giving each a FEATURE subset (Ho, 1998).

    Bagging resamples rows; the random subspace method resamples COLUMNS. Each member
    is trained on a random subset of the features, so different members key on
    different signals and their errors decorrelate -- exactly the diversity that
    makes averaging work. It shines when there are many features, some redundant
    (text, genomics, images), and is the "feature-bagging" half of what a random
    forest does. Any base estimator; predictions are averaged over members, each
    seeing only its own feature subset.
    """

    def __init__(self, estimator=None, n_estimators=25, max_features=0.5,
                 random_state=None):
        self.estimator = estimator
        self.n_estimators = n_estimators
        self.max_features = max_features
        self.random_state = random_state

    def fit(self, X, y):
        X = check_array(X); y = np.asarray(y)
        self.classes_ = np.unique(y)
        rng = check_random_state(self.random_state)
        d = X.shape[1]
        k = max(1, int(self.max_features * d) if isinstance(self.max_features, float)
                else self.max_features)
        base = self.estimator if self.estimator is not None else DecisionTreeClassifier()
        self.subsets_, self.models_ = [], []
        for _ in range(self.n_estimators):
            cols = rng.choice(d, k, replace=False)       # a random feature subspace
            m = clone(base)
            if "random_state" in m.get_params():
                m.set_params(random_state=rng.randint(1 << 30))
            m.fit(X[:, cols], y)
            self.subsets_.append(cols); self.models_.append(m)
        return self

    def predict_proba(self, X):
        X = check_array(X)
        votes = np.zeros((len(X), len(self.classes_)))
        for cols, m in zip(self.subsets_, self.models_):
            pred = m.predict(X[:, cols])
            for i, c in enumerate(self.classes_):
                votes[:, i] += (pred == c)
        return votes / votes.sum(axis=1, keepdims=True)

    def predict(self, X):
        return self.classes_[self.predict_proba(X).argmax(axis=1)]


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


class NegativeCorrelationLearning(BaseEstimator, RegressorMixin):
    """Train neural members to DISAGREE, on purpose (Liu & Yao, 1999).

    An ensemble only helps if its members make DIFFERENT errors, but training them
    independently gives no reason for that. Negative correlation learning trains all
    members TOGETHER with an extra penalty on each member's correlation with the
    ensemble's error -- so every net is pushed to specialise where the others are
    weak. The single knob ``lambda`` interpolates from independent training (0) to
    maximally diverse. It directly optimises the bias-variance-COVARIANCE
    decomposition that governs ensemble error. Small MLP members.
    """

    def __init__(self, n_estimators=5, hidden=16, lam=0.5, epochs=300, lr=0.02,
                 random_state=None):
        self.n_estimators = n_estimators
        self.hidden = hidden
        self.lam = lam
        self.epochs = epochs
        self.lr = lr
        self.random_state = random_state

    def fit(self, X, y):
        from ..autograd import Tensor
        X = check_array(X); y = np.asarray(y, float).reshape(-1, 1)
        rng = check_random_state(self.random_state)
        dims = [X.shape[1], self.hidden, 1]
        self.members_ = [_mlp_params(dims, rng) for _ in range(self.n_estimators)]
        params = [p for m in self.members_ for l in m for p in l.parameters()]
        xt, yt = Tensor(X), Tensor(y)
        for _ in range(self.epochs):
            outs = [_mlp_forward(m, xt) for m in self.members_]
            ens = outs[0]
            for o in outs[1:]:
                ens = ens + o
            ens = ens * (1.0 / self.n_estimators)
            loss = Tensor(np.zeros(1))
            for o in outs:
                mse = ((o - yt) ** 2).mean()
                penalty = ((o - ens) * (ens - yt)).mean()   # negative-correlation term
                loss = loss + mse + self.lam * penalty
            for p in params:
                p.zero_grad()
            loss.backward()
            for p in params:
                p.data -= self.lr * p.grad
        return self

    def predict(self, X):
        from ..autograd import Tensor
        X = check_array(X)
        outs = [_mlp_forward(m, Tensor(X)).data.ravel() for m in self.members_]
        return np.mean(outs, axis=0)


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


__all__ = ["RandomSubspaceClassifier", "NegativeCorrelationLearning", "GrowNet",
           "SnapshotEnsemble"]
