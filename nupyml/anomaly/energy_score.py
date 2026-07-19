"""Turn a classifier's logits into an OOD score (Liu et al., 2020)."""
import numpy as np
from .deep_svdd import DeepSVDD
from .extended_isolation_forest import ExtendedIsolationForest
from .isolation_kernel import IsolationKernel


def energy_score(logits, temperature=1.0):
    """Turn a classifier's logits into an OOD score (Liu et al., 2020).

    A softmax hides how confident a network really is -- it always sums to 1, so an
    out-of-distribution input can still get a spuriously high max-probability. The
    free energy ``-T * logsumexp(logits / T)`` does not: it is LOW for in-
    distribution inputs (some class fires strongly) and HIGH for OOD inputs (no
    class does), and it is theoretically aligned with the data density in a way the
    softmax score is not. Needs only the trained logits -- no retraining, no OOD
    data. Higher score = more out-of-distribution.
    """
    logits = np.asarray(logits, dtype=float)
    a = logits / temperature
    m = a.max(axis=1, keepdims=True)                   # stable logsumexp of a
    lse = m.ravel() + np.log(np.exp(a - m).sum(axis=1))
    return -temperature * lse                          # free energy = OOD score


__all__ = ["energy_score"]
