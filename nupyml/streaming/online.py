"""Online learners: update per example, bounded memory, regret guarantees."""
import numpy as np

from ..base import BaseEstimator, ClassifierMixin, check_is_fitted
from ..utils import check_array, check_random_state


class FTRLProximal(BaseEstimator, ClassifierMixin):
    """Follow-The-Regularized-Leader: sparse online logistic regression.

    THE PROBLEM IT SOLVES
    ---------------------
    Plain online gradient descent with an L1 penalty does NOT produce sparse
    weights: each step nudges a coefficient a little, and the noisy stream of
    gradients keeps knocking near-zero weights back off zero. You want the
    sparsity of the lasso, in a streaming setting, and naive SGD will not give it.

    THE FIX
    -------
    FTRL accumulates the SUM of all gradients seen (that is the "follow the
    leader" part -- move toward what all the data so far prefers) and applies the
    L1 penalty to that accumulated quantity in closed form. A weight is set to
    EXACTLY zero whenever its accumulated gradient stays within the L1 threshold,
    and it flips on the moment the evidence crosses it. So sparsity is genuine and
    stable, not an artifact that the next gradient undoes.

    THE PER-COORDINATE LEARNING RATE
    --------------------------------
    Each weight gets its own step size that shrinks as it accumulates gradient --
    a rare feature seen a handful of times keeps a large, exploratory step, while
    a common one settles down. Exactly right for the sparse, wildly-imbalanced
    feature counts of click prediction, which is the application FTRL was built for
    and dominates.

    McMahan et al. (2013).
    """

    def __init__(self, alpha=0.1, beta=1.0, l1=1.0, l2=1.0):
        self.alpha = alpha              # learning-rate scale
        self.beta = beta                # learning-rate smoothing
        self.l1 = l1
        self.l2 = l2

    def _init(self, n_features):
        self.n_features_ = n_features
        self.z_ = np.zeros(n_features)   # accumulated gradient (the "leader")
        self.n_ = np.zeros(n_features)   # accumulated squared gradient (per-coord LR)
        self.classes_ = np.array([0, 1])

    def _weights(self, active):
        """Reconstruct the weights from (z, n) via the L1 closed form.

        A weight is exactly zero while |z| stays under the L1 threshold, and jumps
        to a shrunken value once it crosses -- soft-thresholding, applied to the
        ACCUMULATED gradient rather than to a running weight. That is what makes
        the zeros stick.
        """
        w = np.zeros(len(active))
        for i, j in enumerate(active):
            z = self.z_[j]
            if abs(z) <= self.l1:
                w[i] = 0.0               # the coefficient stays off
            else:
                lr = (self.beta + np.sqrt(self.n_[j])) / self.alpha
                w[i] = -(z - np.sign(z) * self.l1) / (lr + self.l2)
        return w

    def partial_fit(self, x, y):
        """One example: predict, then update (z, n) from the gradient."""
        x = np.asarray(x, float).ravel()
        if not hasattr(self, "z_"):
            self._init(len(x))
        active = np.where(x != 0)[0]     # only touch the features that fired
        w = self._weights(active)
        p = 1.0 / (1.0 + np.exp(-np.clip(x[active] @ w, -30, 30)))
        g = (p - y) * x[active]          # logistic gradient on the active coords
        # per-coordinate learning-rate bookkeeping
        sigma = (np.sqrt(self.n_[active] + g ** 2) - np.sqrt(self.n_[active])) \
            / self.alpha
        self.z_[active] += g - sigma * w
        self.n_[active] += g ** 2
        return self

    def predict_proba_one(self, x):
        x = np.asarray(x, float).ravel()
        active = np.where(x != 0)[0]
        w = self._weights(active)
        p = 1.0 / (1.0 + np.exp(-np.clip(x[active] @ w, -30, 30)))
        return np.array([1 - p, p])

    def fit(self, X, y):
        """Stream a whole dataset through, one example at a time."""
        X, y = check_array(X), np.asarray(y)
        for xi, yi in zip(X, y):
            self.partial_fit(xi, yi)
        return self

    def predict_proba(self, X):
        check_is_fitted(self, "z_")
        return np.array([self.predict_proba_one(xi) for xi in check_array(X)])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] > 0.5).astype(int)

    @property
    def sparsity_(self):
        """Fraction of weights that are exactly zero -- the point of FTRL."""
        w = np.zeros(self.n_features_)
        nz = np.abs(self.z_) > self.l1
        return 1.0 - nz.mean()


class Hedge(BaseEstimator):
    """Combine expert predictions online, with a regret guarantee.

    THE SETTING
    -----------
    ``n_experts`` advisors each make a prediction every round; you must commit to
    a weighting BEFORE seeing which were right, then suffer each expert's loss in
    proportion to your weight on it. You want to do nearly as well as the best
    single expert -- but you do not know which that is until the end, and the
    losses may be chosen adversarially.

    THE ALGORITHM
    -------------
    Keep a weight per expert; predict with their weighted vote; then multiply each
    expert's weight by ``exp(-learning_rate * its loss)``. Experts that err are
    down-weighted exponentially, so the mass concentrates on the good ones fast.

    THE GUARANTEE
    -------------
    Total loss exceeds the BEST expert's by only ``O(sqrt(T log n))`` -- vanishing
    per round, and with NO assumption on how the losses arise. This "multiplicative
    weights" update is one of the most reused ideas in all of algorithms (it
    reappears in boosting, in game theory, in optimization); Hedge is its cleanest
    statement.

    Freund & Schapire (1997).
    """

    def __init__(self, n_experts, learning_rate=0.5):
        self.n_experts = n_experts
        self.learning_rate = learning_rate

    def reset(self):
        self.weights_ = np.ones(self.n_experts) / self.n_experts
        self.cumulative_loss_ = np.zeros(self.n_experts)
        return self

    def predict(self, expert_predictions):
        """The weighted combination of the experts' predictions this round."""
        if not hasattr(self, "weights_"):
            self.reset()
        return float(self.weights_ @ np.asarray(expert_predictions))

    def update(self, expert_losses):
        """Down-weight each expert by the exponential of its loss."""
        losses = np.asarray(expert_losses, float)
        self.cumulative_loss_ += losses
        # multiplicative weights: the good experts keep their mass, the bad ones
        # lose it exponentially
        self.weights_ *= np.exp(-self.learning_rate * losses)
        self.weights_ /= self.weights_.sum()
        return self

    def regret(self):
        """How much worse than the single best expert in hindsight -- the quantity
        the guarantee bounds."""
        my_loss = self.cumulative_loss_ @ self.weights_
        return float(my_loss - self.cumulative_loss_.min())


class OnlineGradientDescent(BaseEstimator, ClassifierMixin):
    """Plain SGD as a streaming baseline, with a decaying step size.

    The simplest online learner: one gradient step per example. Kept as the
    contrast to FTRL -- same logistic loss, but with an L2 penalty it does NOT
    yield sparse weights, which is exactly the gap FTRL was designed to close.
    The ``1/sqrt(t)`` step decay is what gives it the standard online-convex
    ``O(sqrt(T))`` regret.
    """

    def __init__(self, learning_rate=0.1, l2=0.0):
        self.learning_rate = learning_rate
        self.l2 = l2

    def partial_fit(self, x, y):
        x = np.asarray(x, float).ravel()
        if not hasattr(self, "w_"):
            self.w_ = np.zeros(len(x))
            self.b_ = 0.0
            self._t = 0
        self._t += 1
        lr = self.learning_rate / np.sqrt(self._t)   # decaying step
        p = 1.0 / (1.0 + np.exp(-np.clip(x @ self.w_ + self.b_, -30, 30)))
        g = p - y
        self.w_ -= lr * (g * x + self.l2 * self.w_)
        self.b_ -= lr * g
        self.classes_ = np.array([0, 1])
        return self

    def fit(self, X, y):
        X, y = check_array(X), np.asarray(y)
        for xi, yi in zip(X, y):
            self.partial_fit(xi, yi)
        return self

    def predict_proba(self, X):
        check_is_fitted(self, "w_")
        z = check_array(X) @ self.w_ + self.b_
        p = 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))
        return np.column_stack([1 - p, p])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] > 0.5).astype(int)


__all__ = ["FTRLProximal", "Hedge", "OnlineGradientDescent"]
