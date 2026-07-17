"""Connectionist Temporal Classification: training without an alignment.

THE PROBLEM
-----------
Speech and handwriting give you an input of length T (audio frames) and a target
of length U (characters), with T >> U and NO correspondence between them. Nobody
labelled which frame produced which letter. Cross-entropy needs a target per
frame, and there isn't one.

The old answer was to align first -- with an existing recogniser, or by hand.
CTC's answer: do not align. Sum over EVERY alignment.

THE MODEL
---------
Let the network emit a label per frame, plus one extra symbol: BLANK, meaning
"nothing here". Then define a collapsing rule from a frame-level path to a label
sequence:

1. merge runs of the same label
2. delete the blanks

So for the target "cat", the paths ``c-c-a-a-t``, ``-c-a-t-``, ``cc-aa--t`` all
collapse to it, and CTC maximises the TOTAL probability of every path that
collapses correctly::

    P(target | x) = sum over all valid paths

Blank does two jobs. It absorbs the frames where nothing is being said, and it
separates repeats: "hello" needs a blank between the two l's, because without
one the merge rule would collapse them into a single l.

WHY THIS IS TRACTABLE
---------------------
The number of paths is exponential in T. But paths share prefixes, so the sum
factorises exactly the way the HMM forward algorithm's does -- and for the same
reason. Everything that reaches a given state at time t shares an identical
future, so accumulate that work once::

    alpha[t, s] = probability of all paths reaching state s at time t

Dynamic programming turns an exponential sum into O(T * U). This is the
forward-backward algorithm; see ``nupyml.hmm``, which does the same thing with
observed emissions.

THE EXTENDED SEQUENCE
---------------------
The target is padded with blanks around and between every label::

    "cat"  ->  - c - a - t -

Length ``2U + 1``. This is a bookkeeping trick with real content: it makes "may I
be at a blank here?" a state rather than a special case, so the recursion has
only three transitions to consider (stay, advance one, advance two), and the
skip-by-two is legal exactly when it would not merge two identical labels.

THE GRADIENT
------------
Hand-derived rather than taped. The tape would have to record T*U tiny
operations, and the closed form is both simpler and exact::

    dL/dlogit[t, k] = softmax(logit)[t, k] - posterior[t, k]

which is the familiar "predicted minus target" of any softmax cross-entropy --
except the target is not a label but the POSTERIOR probability that label k is
responsible for frame t, computed by combining alpha and beta. CTC turns out to
be cross-entropy against a soft target that it infers for itself.

Graves et al. (2006).
"""
import numpy as np
from scipy.special import logsumexp

from ..autograd import Tensor
from .module import Module

NEG_INF = -np.inf


def _extend_target(labels, blank):
    """``[c, a, t]`` -> ``[-, c, -, a, -, t, -]``: blanks around every label."""
    extended = np.full(2 * len(labels) + 1, blank, dtype=np.int64)
    extended[1::2] = labels
    return extended


def _forward_log(log_probs, ext, blank):
    """alpha[t, s]: log-probability of every path reaching state s by time t.

    Three ways to arrive at state s at time t:

    * stay at s (the previous frame was already s)
    * advance from s-1
    * skip from s-2 -- allowed ONLY when s is a real label, s-2 is a different
      label, and s-1 is therefore a blank that can be skipped. If s and s-2 were
      the SAME label, skipping the blank between them would let the merge rule
      collapse them into one, which would spell the wrong word.
    """
    T = len(log_probs)
    S = len(ext)
    alpha = np.full((T, S), NEG_INF)
    # a path may start at the leading blank or at the first real label
    alpha[0, 0] = log_probs[0, ext[0]]
    if S > 1:
        alpha[0, 1] = log_probs[0, ext[1]]
    for t in range(1, T):
        for s in range(S):
            paths = [alpha[t - 1, s]]
            if s > 0:
                paths.append(alpha[t - 1, s - 1])
            if s > 1 and ext[s] != blank and ext[s] != ext[s - 2]:
                paths.append(alpha[t - 1, s - 2])
            alpha[t, s] = logsumexp(paths) + log_probs[t, ext[s]]
    return alpha


def _backward_log(log_probs, ext, blank):
    """beta[t, s]: log-probability of completing the target from state s at t.

    The mirror image of the forward pass. A valid path must FINISH at either the
    final blank or the final label, which is why only those two are seeded.
    """
    T = len(log_probs)
    S = len(ext)
    beta = np.full((T, S), NEG_INF)
    beta[T - 1, S - 1] = 0.0
    if S > 1:
        beta[T - 1, S - 2] = 0.0
    for t in range(T - 2, -1, -1):
        for s in range(S):
            paths = [beta[t + 1, s] + log_probs[t + 1, ext[s]]]
            if s < S - 1:
                paths.append(beta[t + 1, s + 1] + log_probs[t + 1, ext[s + 1]])
            if (s < S - 2 and ext[s + 2] != blank
                    and ext[s] != ext[s + 2]):
                paths.append(beta[t + 1, s + 2] + log_probs[t + 1, ext[s + 2]])
            beta[t, s] = logsumexp(paths)
    return beta


def _ctc_single(log_probs, labels, blank):
    """Loss and gradient for one sequence. Returns (nll, dL/dlog_probs)."""
    T, K = log_probs.shape
    ext = _extend_target(labels, blank)
    S = len(ext)
    alpha = _forward_log(log_probs, ext, blank)
    beta = _backward_log(log_probs, ext, blank)

    # total probability: every path must end at the last blank or last label
    tail = [alpha[T - 1, S - 1]]
    if S > 1:
        tail.append(alpha[T - 1, S - 2])
    log_likelihood = logsumexp(tail)
    if not np.isfinite(log_likelihood):
        # no alignment exists (target longer than the input can express)
        return 0.0, np.zeros_like(log_probs)

    # posterior[t, k]: probability that label k is responsible for frame t.
    # alpha covers the past, beta the future; their product over the total is
    # the probability that a path passes through this state at this time.
    gamma = alpha + beta
    posterior = np.full((T, K), NEG_INF)
    for s in range(S):
        k = ext[s]
        posterior[:, k] = np.logaddexp(posterior[:, k], gamma[:, s])
    posterior = np.exp(posterior - log_likelihood)

    # the softmax-cross-entropy gradient, against an inferred soft target
    grad = np.exp(log_probs) - posterior
    return -log_likelihood, grad


class CTCLoss(Module):
    """CTC loss over log-probabilities, summing across every valid alignment.

    Parameters
    ----------
    blank : int
        Index of the blank symbol. Conventionally 0, which is why real
        vocabularies start at 1.
    reduction : {"mean", "sum", "none"}
        ``mean`` averages over sequences.

    Notes
    -----
    Input is ``(T, N, K)`` log-probabilities -- time-major, as every CTC
    implementation is, because the recursion walks time.

    CTC assumes each frame's output is conditionally independent given the input.
    That is false (letters depend on their neighbours) and is precisely why CTC
    models are paired with a language model at decoding time: CTC handles the
    alignment, the language model supplies what CTC's independence assumption
    threw away.
    """

    def __init__(self, blank=0, reduction="mean"):
        super().__init__()
        self.blank = blank
        self.reduction = reduction

    def forward(self, log_probs, targets, input_lengths=None,
                target_lengths=None):
        lp = Tensor._wrap(log_probs)
        data = lp.data
        T, N, K = data.shape
        if input_lengths is None:
            input_lengths = [T] * N
        targets = [np.asarray(t, dtype=np.int64) for t in targets]

        losses = np.empty(N)
        grads = np.zeros_like(data)
        for i in range(N):
            Ti = int(input_lengths[i])
            loss, g = _ctc_single(data[:Ti, i, :], targets[i], self.blank)
            losses[i] = loss
            grads[:Ti, i, :] = g

        if self.reduction == "mean":
            total = float(losses.mean())
            grads /= N
        elif self.reduction == "sum":
            total = float(losses.sum())
        else:
            total = losses

        def backward(g):
            if lp.requires_grad:
                lp._accumulate(grads * g)

        return Tensor._make(np.array(total), (lp,), backward, "ctc")


def ctc_greedy_decode(log_probs, blank=0):
    """Decode by taking the best label per frame, then collapsing.

    The cheap decoder: argmax each frame independently, merge runs, drop blanks.
    It finds the most likely PATH, which is not the most likely LABELLING -- many
    paths collapse to the same output, and a labelling can win in total while
    losing on every individual path. Beam search over collapsed prefixes fixes
    that, and is where a language model is folded in.
    """
    data = log_probs.data if isinstance(log_probs, Tensor) else np.asarray(log_probs)
    best = data.argmax(axis=-1)
    out = []
    for seq in best.T:
        collapsed = []
        prev = -1
        for k in seq:
            if k != prev and k != blank:
                collapsed.append(int(k))
            prev = k
        out.append(collapsed)
    return out


__all__ = ["CTCLoss", "ctc_greedy_decode"]
