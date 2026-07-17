"""Mixture of experts: divide the problem, and learn the division.

THE IDEA
--------
An ensemble usually asks every member about every sample and then averages. A
mixture of experts does not. A GATING network decides which expert to trust for
THIS input, and the output is a gated combination::

    y = sum_i  gate_i(x) * expert_i(x)      with  sum_i gate_i(x) = 1

Nothing tells the experts what to specialise in. They are trained jointly with
the gate, and specialisation EMERGES: an expert that happens to be slightly
better on some region gets a larger gate there, which means it sees more of that
region's gradient, which makes it better there still. The gate and the experts
bootstrap a division of labour that nobody designed.

THE CONTRAST WITH BAGGING AND BOOSTING
--------------------------------------
* **Bagging** -- every member sees everything, and they vote. Members are
  interchangeable, and the point is variance reduction.
* **Boosting** -- members are sequential, each fixing the last one's errors.
* **Mixture of experts** -- members are SPECIALISTS, and a learned gate routes
  each input to the right one. Nobody is interchangeable, and averaging them
  would be actively wrong: an expert's output is meaningless outside its region.

That last point is why this is not just another averaging scheme. If the data
really is a union of different regimes -- different physics in different regions
-- then a single averaged model fits none of them, and a mixture fits each.

WHY IT MATTERS BEYOND THIS FILE
-------------------------------
This is the idea behind sparse MoE layers in large language models. There, the
gate picks the top ``k`` experts out of many and runs only those, so the model
has enormous CAPACITY (all the experts' parameters) at a small COST per token
(only k of them run). Capacity and compute stop being the same quantity, which
is the entire reason the architecture is interesting at scale.

The load-balancing problem below is the same one those models face, and it is
the hard part in both.
"""
import numpy as np

from ..autograd import Tensor
from ..autograd.functional import softmax
from ..base import BaseEstimator, ClassifierMixin, RegressorMixin, check_is_fitted
from ..nn import Adam, Linear, Module, ModuleList, Sequential, ReLU
from ..preprocessing import LabelEncoder
from ..utils import check_X_y, check_array, check_random_state


class _Gate(Module):
    """Softmax over experts, per sample. A router, not a predictor."""

    def __init__(self, n_features, n_experts, rng=None):
        super().__init__()
        self.net = Linear(n_features, n_experts, rng=rng)

    def forward(self, x):
        return softmax(self.net(Tensor._wrap(x)), axis=1)


class _Expert(Module):
    def __init__(self, n_features, n_out, hidden, rng=None):
        super().__init__()
        layers, prev = [], n_features
        for h in hidden:
            layers += [Linear(prev, h, rng=rng), ReLU()]
            prev = h
        layers.append(Linear(prev, n_out, rng=rng))
        self.net = Sequential(*layers)

    def forward(self, x):
        return self.net(Tensor._wrap(x))


class _BaseMoE(BaseEstimator):
    """Experts and a gate, trained together end to end.

    THE FAILURE MODE: EXPERT COLLAPSE
    ---------------------------------
    The bootstrap that creates specialisation can run the other way. If one
    expert starts slightly ahead, the gate sends it more traffic, so it trains
    more, so it gets further ahead, so the gate sends it everything. The other
    experts receive almost no gradient and stay at their initialisation forever.
    You have paid for N experts and are running one.

    Nothing in the objective prevents this -- a single good expert is a perfectly
    fine way to minimise the loss, and the optimizer will happily take it.

    LOAD BALANCING
    --------------
    So an auxiliary loss penalises unequal gate usage. ``load_balance_coef``
    weights it, and the mechanism is worth stating plainly: it is a penalty on the
    variance of each expert's total gate mass across the batch, which pushes the
    gate toward using everyone.

    This is a genuine trade-off, not a free fix. Too much and the gate is forced
    to spread work evenly REGARDLESS of who is good at what -- destroying the
    specialisation that is the entire point. Too little and one expert eats the
    problem. ``expert_usage_`` reports where a fitted model landed, because you
    cannot tell from the loss curve.
    """

    def __init__(self, n_experts=4, expert_hidden=(32,), n_epochs=200,
                 learning_rate=1e-2, batch_size=32, load_balance_coef=0.1,
                 random_state=None):
        self.n_experts = n_experts
        self.expert_hidden = expert_hidden
        self.n_epochs = n_epochs
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.load_balance_coef = load_balance_coef
        self.random_state = random_state

    def _build(self, n_features, n_out, rng):
        self.gate_ = _Gate(n_features, self.n_experts, rng=rng)
        self.experts_ = ModuleList([
            _Expert(n_features, n_out, self.expert_hidden, rng=rng)
            for _ in range(self.n_experts)])

    def _parameters(self):
        return list(self.gate_.parameters()) + list(self.experts_.parameters())

    def _expert_output(self, expert, X):
        """What the gate is allowed to mix. Regression mixes the raw output;
        classification must map to probabilities first -- see the classifier."""
        return expert(X)

    def _mixture(self, X):
        """Gate-weighted combination of every expert's output."""
        gates = self.gate_(X)                       # (n, n_experts)
        total = None
        for i, expert in enumerate(self.experts_):
            weighted = self._expert_output(expert, X) * gates[:, i].reshape(-1, 1)
            total = weighted if total is None else total + weighted
        return total, gates

    def _load_balance_loss(self, gates):
        """Penalise unequal expert usage -- see the class docstring.

        The importance of an expert is its total gate mass over the batch. Its
        VARIANCE across experts is zero exactly when every expert is used
        equally, so penalising it pushes toward balance without dictating which
        expert gets which sample.
        """
        importance = gates.sum(axis=0)
        return importance.var() / (importance.mean() ** 2 + 1e-8)

    def _fit_network(self, X, target, loss_fn, rng):
        opt = Adam(self._parameters(), lr=self.learning_rate)
        n = len(X)
        self.history_ = []
        for _ in range(self.n_epochs):
            perm = rng.permutation(n)
            for start in range(0, n - self.batch_size + 1, self.batch_size):
                b = perm[start:start + self.batch_size]
                opt.zero_grad()
                out, gates = self._mixture(X[b])
                loss = loss_fn(out, target[b])
                if self.load_balance_coef > 0:
                    loss = loss + self._load_balance_loss(gates) * self.load_balance_coef
                loss.backward()
                opt.step()
            with_all, gates = self._mixture(X)
            self.history_.append(float(loss_fn(with_all, target).item()))

        # where the specialisation actually landed; unreadable from the loss
        _, gates = self._mixture(X)
        self.expert_usage_ = gates.data.mean(axis=0)
        return self

    def gate_weights(self, X):
        """Each expert's share of each sample. This is the interpretable part --
        it shows the division of labour the model discovered."""
        check_is_fitted(self, "gate_")
        return self.gate_(check_array(X)).data


class MixtureOfExpertsRegressor(_BaseMoE, RegressorMixin):
    """Regression by gated specialists.

    Shines when the data is genuinely piecewise -- different regimes in different
    regions. On a smooth homogeneous problem a single network of the same total
    size will match it, because there is nothing to specialise in.
    """

    def fit(self, X, y):
        X, y = check_X_y(X, y, y_numeric=True)
        rng = check_random_state(self.random_state)
        self._build(X.shape[1], 1, rng)
        target = y.reshape(-1, 1)
        return self._fit_network(
            X, target, lambda out, t: ((out - Tensor(t)) ** 2).mean(), rng)

    def predict(self, X):
        check_is_fitted(self, "experts_")
        return self._mixture(check_array(X))[0].data.ravel()


class MixtureOfExpertsClassifier(_BaseMoE, ClassifierMixin):
    """Classification by gated specialists.

    The mixture is over PROBABILITIES, not logits. A convex combination is only
    meaningful for quantities that live on a simplex: mixing probabilities gives
    a distribution, whereas mixing logits gives a number whose softmax is not the
    mixture of anything. The gate says "70% expert A", and that should mean 70% of
    A's BELIEF, which is only true on this side of the softmax.
    """

    def _expert_output(self, expert, X):
        return softmax(expert(X), axis=1)

    def fit(self, X, y):
        X, y = check_X_y(X, y)
        rng = check_random_state(self.random_state)
        self._le = LabelEncoder().fit(y)
        self.classes_ = self._le.classes_
        y_idx = self._le.transform(y)
        k = len(self.classes_)
        self._build(X.shape[1], k, rng)

        onehot = np.eye(k)[y_idx]

        def loss_fn(out, t):
            # `out` is already the gate-weighted mixture of the experts'
            # probabilities, so it is a distribution and needs no further
            # softmax -- only the log of the mixture
            return -(Tensor(t) * out.clip(1e-9, 1.0).log()).sum(axis=1).mean()

        return self._fit_network(X, onehot, loss_fn, rng)

    def predict_proba(self, X):
        check_is_fitted(self, "experts_")
        return self._mixture(check_array(X))[0].data

    def predict(self, X):
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]


__all__ = ["MixtureOfExpertsRegressor", "MixtureOfExpertsClassifier"]
