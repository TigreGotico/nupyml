"""Restricted Boltzmann machines: learning a distribution you cannot normalise.

AN ENERGY MODEL, NOT A FUNCTION
-------------------------------
Everything else in this library maps input to output. An RBM does not. It assigns
an ENERGY to each joint configuration of visible units ``v`` and hidden units
``h``, and defines a probability from it::

    E(v, h) = -v.b_v - h.b_h - v.W.h
    p(v, h) = exp(-E(v, h)) / Z

Low energy means likely. Training means pushing energy down on the data and up
everywhere else -- sculpting a landscape rather than fitting a curve.

"Restricted" means no visible-visible or hidden-hidden connections: the graph is
bipartite. That restriction is the entire reason the model is tractable, because
it makes the conditionals factorise::

    p(h_j = 1 | v) = sigmoid(b_h_j + v.W_.j)      all h independent given v
    p(v_i = 1 | h) = sigmoid(b_v_i + W_i..h)      all v independent given h

So a whole layer samples in one vectorized shot. Lift the restriction and you
have a Boltzmann machine, which is beautiful and useless.

THE INTRACTABLE PART
--------------------
``Z`` sums over EVERY configuration -- 2^n_visible * 2^n_hidden of them. It cannot
be computed, so the likelihood cannot be computed. But the GRADIENT has a shape
that saves us::

    d log p(v) / d W = <v h>_data - <v h>_model
                       ^^^^^^^^^^   ^^^^^^^^^^^
                       positive     negative phase
                       (easy)       (needs samples from the model)

The positive phase is one conditional pass. The negative phase needs samples from
the model itself, which means running a Gibbs chain to equilibrium -- unboundedly
long.

CONTRASTIVE DIVERGENCE: THE APPROXIMATION THAT WORKS
----------------------------------------------------
Hinton's CD-k: don't run the chain to equilibrium. Start it AT THE DATA and run k
steps -- often k=1::

    v_0 (data) -> h_0 -> v_1 -> h_1 -> ... -> v_k

then use ``<v_k h_k>`` in place of the model expectation. It is not the true
gradient and it is not the gradient of any function, so nothing guarantees it
descends anything. It works anyway, and it is what made deep belief networks
briefly the way to train deep networks.

The intuition: the true gradient asks "where does the model put mass that the
data does not?" -- a global question. CD asks the local version: "starting from
the data, which way does the model immediately drift?" Push back against that
drift. If the model does not move away from the data, it already fits it.

WHY LEARN THIS
--------------
RBMs are no longer state of the art. They are here because the pattern -- an
unnormalisable density, a positive and negative phase, and sampling to fill in
what cannot be computed -- recurs throughout energy-based models, and this is
where it is clearest.
"""
import numpy as np

from ..base import BaseEstimator, TransformerMixin, check_is_fitted
from ..utils import check_array, check_random_state


def _sigmoid(x):
    """Logistic function, written to not overflow on large negative input.

    ``exp(800)`` overflows to inf, so the naive ``1/(1+exp(-x))`` warns and
    returns garbage on saturated units. Evaluating each half of the domain with
    the form that keeps the exponent negative is exact everywhere.
    """
    out = np.empty_like(x, dtype=np.float64)
    pos = x >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
    e = np.exp(x[~pos])
    out[~pos] = e / (1.0 + e)
    return out


class BernoulliRBM(BaseEstimator, TransformerMixin):
    """An RBM over binary units, trained by contrastive divergence.

    ``transform`` returns ``p(h|v)`` -- the hidden probabilities, not samples --
    because as a feature extractor you want the deterministic representation. The
    sampling is for training, where the noise is doing real work.

    Parameters
    ----------
    n_components : number of hidden units.
    learning_rate : CD is a noisy gradient; this wants to be small.
    n_iter : passes over the data.
    cd_k : Gibbs steps per update. k=1 is the classic and usually enough.
    """

    def __init__(self, n_components=64, learning_rate=0.05, batch_size=32,
                 n_iter=20, cd_k=1, random_state=None):
        self.n_components = n_components
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.n_iter = n_iter
        self.cd_k = cd_k
        self.random_state = random_state

    def _sample_hidden(self, v, rng):
        p = _sigmoid(v @ self.components_.T + self.intercept_hidden_)
        return p, (rng.uniform(size=p.shape) < p).astype(np.float64)

    def _sample_visible(self, h, rng):
        p = _sigmoid(h @ self.components_ + self.intercept_visible_)
        return p, (rng.uniform(size=p.shape) < p).astype(np.float64)

    def fit(self, X, y=None):
        X = check_array(X)
        rng = check_random_state(self.random_state)
        n_features = X.shape[1]

        # small random weights: symmetric or large init leaves every hidden unit
        # computing the same thing, and nothing in the update breaks the tie
        self.components_ = rng.normal(0, 0.01, size=(self.n_components, n_features))
        self.intercept_hidden_ = np.zeros(self.n_components)
        self.intercept_visible_ = np.zeros(n_features)

        for _ in range(self.n_iter):
            for batch in np.array_split(rng.permutation(len(X)),
                                        max(1, len(X) // self.batch_size)):
                self._cd_update(X[batch], rng)
        return self

    def partial_fit(self, X, y=None):
        X = check_array(X)
        rng = check_random_state(self.random_state)
        if not hasattr(self, "components_"):
            self.components_ = rng.normal(0, 0.01,
                                          size=(self.n_components, X.shape[1]))
            self.intercept_hidden_ = np.zeros(self.n_components)
            self.intercept_visible_ = np.zeros(X.shape[1])
        self._cd_update(X, rng)
        return self

    def _cd_update(self, v_data, rng):
        """One CD-k step: positive phase, k Gibbs steps, negative phase."""
        n = len(v_data)

        # positive phase: correlations under the data
        p_h_data, h_sample = self._sample_hidden(v_data, rng)
        pos = p_h_data.T @ v_data

        # the Gibbs chain, started at the data rather than at random -- the whole
        # of what makes CD cheap
        h = h_sample
        for _ in range(self.cd_k):
            p_v_model, v_model = self._sample_visible(h, rng)
            p_h_model, h = self._sample_hidden(v_model, rng)

        # the last hidden step uses probabilities, not samples: the sample adds
        # variance to the gradient and buys nothing, since this expectation is
        # the only place the chain's noise would otherwise enter
        neg = p_h_model.T @ p_v_model

        lr = self.learning_rate / n
        self.components_ += lr * (pos - neg)
        self.intercept_hidden_ += lr * (p_h_data.sum(0) - p_h_model.sum(0))
        self.intercept_visible_ += lr * (v_data.sum(0) - p_v_model.sum(0))

    def transform(self, X):
        """p(h|v): the learned features."""
        check_is_fitted(self, "components_")
        X = check_array(X)
        return _sigmoid(X @ self.components_.T + self.intercept_hidden_)

    def inverse_transform(self, H):
        return _sigmoid(np.asarray(H) @ self.components_ + self.intercept_visible_)

    def gibbs(self, v, n_steps=1, random_state=None):
        """Run the chain from ``v``. Long chains wander toward the model's own
        distribution, which is how you see what an RBM has actually learned."""
        check_is_fitted(self, "components_")
        rng = check_random_state(random_state
                                 if random_state is not None else self.random_state)
        v = check_array(v).copy()
        for _ in range(n_steps):
            _, h = self._sample_hidden(v, rng)
            _, v = self._sample_visible(h, rng)
        return v

    def free_energy(self, X):
        """F(v) = -log sum_h exp(-E(v, h)), which IS tractable.

        The intractable ``Z`` is the sum over v as well; summing out only h is a
        closed form, because the hidden units are independent given v. So relative
        likelihoods of two visible vectors are computable even though absolute
        ones are not -- ``F`` is ``-log p(v)`` up to the constant ``log Z``, which
        cancels in any comparison.

        Useful for monitoring: free energy on held-out data drifting away from
        free energy on training data is this model's overfitting signal.
        """
        check_is_fitted(self, "components_")
        X = check_array(X)
        wx = X @ self.components_.T + self.intercept_hidden_
        # logaddexp(0, wx) is log(1 + exp(wx)) computed without overflowing
        return -X @ self.intercept_visible_ - np.logaddexp(0, wx).sum(axis=1)

    def score_samples(self, X):
        """Negative free energy: higher means the model likes the sample more.

        NOT a log-likelihood -- the missing ``log Z`` makes the absolute value
        meaningless. Only differences between samples mean anything.
        """
        return -self.free_energy(X)


__all__ = ["BernoulliRBM"]
