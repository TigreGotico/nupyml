"""Linear-chain CRF and the structured perceptron: labelling a whole sequence.

THE PROBLEM
-----------
Tag every word in a sentence with its part of speech; every pixel-row with
whether it is text; every time step with the activity. The catch is that the tags
DEPEND ON EACH OTHER: a determiner is usually followed by a noun, a "B-name" tag
by an "I-name" tag. Predict each label independently and you get locally
plausible, globally incoherent nonsense -- a sequence of tags no valid sentence
would ever carry.

The fix is to score and predict the WHOLE label sequence jointly, so the
transitions between labels are part of what is optimised.

WHY NOT JUST AN HMM
-------------------
An HMM also models label transitions, so why the CRF? Because an HMM is
GENERATIVE -- it models how the observations are produced, ``P(words, tags)``, and
to stay tractable it must assume each word depends only on its own tag. That
forbids the overlapping, arbitrary features that actually help: "is this word
capitalised", "does it end in -ing", "is the previous word 'the'".

A CRF is DISCRIMINATIVE. It models ``P(tags | words)`` directly, so it never has
to explain how the words arose and is free to use any features of the entire
input at any position. Same chain structure, but it spends its modelling budget
on the question you actually care about. This is the same generative-vs-
discriminative split as naive Bayes versus logistic regression -- and the CRF is
precisely the sequence-shaped logistic regression.

THE ENGINE
----------
Two dynamic programs, both shared with the HMM:

* **Viterbi** finds the single best label sequence (for prediction).
* **Forward-backward** sums over ALL label sequences to get the normaliser and
  the marginals (for the gradient during training).

That the same two algorithms power the HMM, the CRF, and -- in ``nn/ctc.py`` --
connectionist temporal classification is the deep point worth carrying away.

Lafferty, McCallum & Pereira (2001).
"""
import numpy as np
from scipy.special import logsumexp

from ..base import BaseEstimator, check_is_fitted
from ..utils import check_random_state


class LinearChainCRF(BaseEstimator):
    """A conditional random field over a chain, trained by gradient ascent.

    Features are provided as arrays: ``X`` is a list of sequences, each of shape
    ``(length, n_features)``. The model learns emission weights (feature -> label)
    and transition weights (label -> label). Prediction is Viterbi; training
    maximises the conditional log-likelihood, whose gradient is the classic
    "observed minus expected feature counts".
    """

    def __init__(self, n_states, learning_rate=0.1, n_epochs=50, l2=0.01,
                 random_state=None):
        self.n_states = n_states
        self.learning_rate = learning_rate
        self.n_epochs = n_epochs
        self.l2 = l2
        self.random_state = random_state

    def _scores(self, X_seq):
        """Emission scores: for each position, a score per label.

        ``(length, n_states)``. Transition scores are position-independent and
        held separately in ``self.transitions_``.
        """
        return X_seq @ self.emissions_.T

    def _forward_backward(self, emit):
        """Log-space forward and backward passes -> log Z and marginals.

        Log space throughout, via ``logsumexp``: the raw probabilities underflow
        to zero within a dozen steps, and this is the exact same numerical care
        the HMM needs. ``alpha`` accumulates from the left, ``beta`` from the
        right; their combination gives every position's label posterior.
        """
        T = len(emit)
        trans = self.transitions_

        alpha = np.zeros((T, self.n_states))
        alpha[0] = emit[0]
        for t in range(1, T):
            # sum over previous labels, in log space
            alpha[t] = emit[t] + logsumexp(alpha[t - 1][:, None] + trans, axis=0)

        beta = np.zeros((T, self.n_states))
        for t in range(T - 2, -1, -1):
            beta[t] = logsumexp(trans + emit[t + 1] + beta[t + 1], axis=1)

        log_Z = logsumexp(alpha[-1])
        # node marginals: P(label at t) given the whole sequence
        node_marg = np.exp(alpha + beta - log_Z)
        return log_Z, node_marg, alpha, beta

    def _edge_marginals(self, emit, alpha, beta, log_Z):
        """P(label t = i, label t+1 = j) for the transition gradient."""
        T = len(emit)
        trans = self.transitions_
        edge = np.zeros((self.n_states, self.n_states))
        for t in range(T - 1):
            m = (alpha[t][:, None] + trans + emit[t + 1] + beta[t + 1]) - log_Z
            edge += np.exp(m)
        return edge

    def fit(self, X, y):
        rng = check_random_state(self.random_state)
        n_features = X[0].shape[1]
        # small random init; symmetric weights would give every label the same
        # score and the gradient nothing to break
        self.emissions_ = rng.normal(0, 0.01, size=(self.n_states, n_features))
        self.transitions_ = rng.normal(0, 0.01, size=(self.n_states, self.n_states))
        self.log_likelihood_ = []

        for _ in range(self.n_epochs):
            total_ll = 0.0
            grad_emit = np.zeros_like(self.emissions_)
            grad_trans = np.zeros_like(self.transitions_)

            for X_seq, y_seq in zip(X, y):
                emit = self._scores(X_seq)
                log_Z, node_marg, alpha, beta = self._forward_backward(emit)

                # score of the TRUE path minus the normaliser = log P(y | x)
                true_score = sum(emit[t, y_seq[t]] for t in range(len(y_seq)))
                true_score += sum(self.transitions_[y_seq[t], y_seq[t + 1]]
                                  for t in range(len(y_seq) - 1))
                total_ll += true_score - log_Z

                # the gradient is OBSERVED minus EXPECTED feature counts -- the
                # same shape as logistic regression's, lifted to sequences
                for t in range(len(y_seq)):
                    grad_emit[y_seq[t]] += X_seq[t]          # observed
                    grad_emit -= node_marg[t][:, None] * X_seq[t]   # expected
                edge = self._edge_marginals(emit, alpha, beta, log_Z)
                for t in range(len(y_seq) - 1):
                    grad_trans[y_seq[t], y_seq[t + 1]] += 1.0    # observed
                grad_trans -= edge                                # expected

            # L2 shrinks the weights; ascent, so subtract the penalty gradient
            grad_emit -= self.l2 * self.emissions_
            grad_trans -= self.l2 * self.transitions_
            self.emissions_ += self.learning_rate * grad_emit / len(X)
            self.transitions_ += self.learning_rate * grad_trans / len(X)
            self.log_likelihood_.append(total_ll / len(X))

        return self

    def _viterbi(self, emit):
        """The single best label sequence -- max, not sum, over paths."""
        T = len(emit)
        trans = self.transitions_
        score = np.zeros((T, self.n_states))
        back = np.zeros((T, self.n_states), dtype=int)
        score[0] = emit[0]
        for t in range(1, T):
            # for each label, the best previous label to have come from
            m = score[t - 1][:, None] + trans
            back[t] = np.argmax(m, axis=0)
            score[t] = emit[t] + np.max(m, axis=0)

        # backtrack from the best final label
        path = [int(np.argmax(score[-1]))]
        for t in range(T - 1, 0, -1):
            path.append(int(back[t, path[-1]]))
        return path[::-1]

    def predict(self, X):
        check_is_fitted(self, "emissions_")
        return [self._viterbi(self._scores(seq)) for seq in X]

    def score(self, X, y):
        """Token-level accuracy over all sequences."""
        preds = self.predict(X)
        correct = sum(p == t for pred, true in zip(preds, y)
                      for p, t in zip(pred, true))
        total = sum(len(t) for t in y)
        return correct / total


class StructuredPerceptron(BaseEstimator):
    """The CRF's cheap cousin: perceptron updates over whole sequences.

    THE SHORTCUT
    ------------
    The CRF's gradient needs forward-backward to compute EXPECTED feature counts
    -- a sum over all label sequences. The structured perceptron replaces that
    expectation with the single BEST WRONG sequence, found by Viterbi::

        predict the best sequence; if it is wrong, add the true sequence's
        features and subtract the predicted sequence's.

    So training needs only Viterbi (a max), never forward-backward (a sum). It is
    dramatically simpler and often nearly as accurate -- the same relationship the
    plain perceptron has to logistic regression, lifted to structured output.

    AVERAGING IS NOT OPTIONAL
    -------------------------
    Like the plain perceptron, the last weight vector is jumpy -- it lurches on
    whichever sequence it saw last. AVERAGING the weights over all updates is what
    makes it competitive; the averaged vector is far more stable, and Collins
    showed it comes with a real generalisation bound. Skipping it is the usual
    reason a structured perceptron "does not work".

    Collins (2002).
    """

    def __init__(self, n_states, n_epochs=20, average=True, random_state=None):
        self.n_states = n_states
        self.n_epochs = n_epochs
        self.average = average
        self.random_state = random_state

    def _viterbi(self, emit, trans):
        T = len(emit)
        score = np.zeros((T, self.n_states))
        back = np.zeros((T, self.n_states), dtype=int)
        score[0] = emit[0]
        for t in range(1, T):
            m = score[t - 1][:, None] + trans
            back[t] = np.argmax(m, axis=0)
            score[t] = emit[t] + np.max(m, axis=0)
        path = [int(np.argmax(score[-1]))]
        for t in range(T - 1, 0, -1):
            path.append(int(back[t, path[-1]]))
        return path[::-1]

    def fit(self, X, y):
        rng = check_random_state(self.random_state)
        n_features = X[0].shape[1]
        emit_w = np.zeros((self.n_states, n_features))
        trans_w = np.zeros((self.n_states, self.n_states))
        # running sums for the averaged weights
        emit_sum = np.zeros_like(emit_w)
        trans_sum = np.zeros_like(trans_w)
        n_updates = 0

        for _ in range(self.n_epochs):
            order = rng.permutation(len(X))
            for i in order:
                X_seq, y_seq = X[i], y[i]
                emit = X_seq @ emit_w.T
                pred = self._viterbi(emit, trans_w)

                if pred != list(y_seq):
                    # push weights toward the truth, away from the prediction --
                    # only where they disagree, which is the perceptron rule
                    for t in range(len(y_seq)):
                        emit_w[y_seq[t]] += X_seq[t]
                        emit_w[pred[t]] -= X_seq[t]
                    for t in range(len(y_seq) - 1):
                        trans_w[y_seq[t], y_seq[t + 1]] += 1
                        trans_w[pred[t], pred[t + 1]] -= 1
                n_updates += 1
                emit_sum += emit_w
                trans_sum += trans_w

        if self.average:
            # the averaged weights -- the version that actually generalises
            self.emissions_ = emit_sum / n_updates
            self.transitions_ = trans_sum / n_updates
        else:
            self.emissions_ = emit_w
            self.transitions_ = trans_w
        return self

    def predict(self, X):
        check_is_fitted(self, "emissions_")
        return [self._viterbi(seq @ self.emissions_.T, self.transitions_)
                for seq in X]

    def score(self, X, y):
        preds = self.predict(X)
        correct = sum(p == t for pred, true in zip(preds, y)
                      for p, t in zip(pred, true))
        return correct / sum(len(t) for t in y)


__all__ = ["LinearChainCRF", "StructuredPerceptron"]
