"""Offline change-point detection algorithms.

All take a 1-D signal and return the indices where its behaviour changes. The
cost of a segment is its within-segment sum of squares (a mean-shift model); a
per-change penalty stops the trivial "split everywhere" solution.
"""
import numpy as np


def _segment_cost(cumsum, cumsq, i, j):
    """Sum of squared deviations from the mean over signal[i:j], in O(1).

    Precomputed prefix sums make each segment's cost a constant-time lookup --
    which is what lets the dynamic programs below run over all segmentations
    without recomputing means. cost = sum(x^2) - (sum x)^2 / length.
    """
    n = j - i
    if n <= 0:
        return 0.0
    s = cumsum[j] - cumsum[i]
    sq = cumsq[j] - cumsq[i]
    return sq - s * s / n


def pelt(signal, penalty=None):
    """PELT: the OPTIMAL segmentation under a per-change penalty, in ~O(n).

    THE OBJECTIVE
    -------------
    Minimise ``sum of segment costs + penalty * (number of change-points)``. The
    penalty is essential: without it, splitting at every point drives the cost to
    zero, so the penalty is what buys the "is this split worth it?" decision.

    THE ALGORITHM
    -------------
    A dynamic program computes the best segmentation of the first ``t`` points
    from the best segmentations of all earlier prefixes -- which is ``O(n^2)``.
    PELT's contribution is a PRUNING rule: a candidate last-change-point that can
    never beat the current best (its cost already exceeds the best plus the
    penalty) is discarded forever, and in practice this collapses the cost to
    linear. So it is exact (unlike greedy binary segmentation) AND fast, which is
    why PELT is the default offline change-point method.

    ``penalty`` defaults to the BIC-style ``2 * log(n) * variance``.

    Killick, Fearnhead & Eckley (2012). Returns sorted change-point indices.
    """
    x = np.asarray(signal, float)
    n = len(x)
    cumsum = np.concatenate([[0], np.cumsum(x)])
    cumsq = np.concatenate([[0], np.cumsum(x ** 2)])
    if penalty is None:
        penalty = 2 * np.log(n) * np.var(x)

    F = np.full(n + 1, np.inf)      # best total cost up to each point
    F[0] = -penalty
    last = [0] * (n + 1)            # the change-point before each position
    candidates = [0]                # prunable set of possible last-changes

    for t in range(1, n + 1):
        best_cost, best_s = np.inf, 0
        for s in candidates:
            cost = F[s] + _segment_cost(cumsum, cumsq, s, t) + penalty
            if cost < best_cost:
                best_cost, best_s = cost, s
        F[t] = best_cost
        last[t] = best_s
        # PELT pruning: drop candidates that can never become optimal later
        candidates = [s for s in candidates
                      if F[s] + _segment_cost(cumsum, cumsq, s, t) <= F[t]]
        candidates.append(t)

    # backtrack the change-points
    cps, t = [], n
    while t > 0:
        s = last[t]
        if s > 0:
            cps.append(s)
        t = s
    return sorted(cps)


def binary_segmentation(signal, penalty=None, max_cps=None):
    """Greedy segmentation: split at the single best point, then recurse.

    THE ALGORITHM
    -------------
    Find the ONE split that most reduces the cost of the whole signal; if that
    reduction beats the penalty, accept it and recurse independently on the left
    and right pieces. Stop when no split is worth its penalty (or ``max_cps`` is
    reached).

    The contrast with PELT: binary segmentation is fast and dead simple, but
    GREEDY -- an early split is committed and never revisited, so it can miss the
    optimal segmentation when change-points interact (two nearby changes can hide
    each other from the first, single-best-split search). PELT pays a little more
    to be exact. Binary segmentation is the pragmatic choice when the changes are
    well-separated.

    Returns sorted change-point indices.
    """
    x = np.asarray(signal, float)
    n = len(x)
    cumsum = np.concatenate([[0], np.cumsum(x)])
    cumsq = np.concatenate([[0], np.cumsum(x ** 2)])
    if penalty is None:
        penalty = 2 * np.log(n) * np.var(x)

    cps = []

    def recurse(start, end):
        if max_cps is not None and len(cps) >= max_cps:
            return
        if end - start < 2:
            return
        whole = _segment_cost(cumsum, cumsq, start, end)
        best_gain, best_k = 0.0, None
        for k in range(start + 1, end):
            gain = whole - (_segment_cost(cumsum, cumsq, start, k)
                            + _segment_cost(cumsum, cumsq, k, end))
            if gain > best_gain:
                best_gain, best_k = gain, k
        # accept the split only if its cost reduction beats the penalty
        if best_k is not None and best_gain > penalty:
            cps.append(best_k)
            recurse(start, best_k)
            recurse(best_k, end)

    recurse(0, n)
    return sorted(cps)


def cusum(signal, threshold=None, drift=0.0):
    """CUSUM: flag a mean shift from when the cumulative sum starts drifting.

    THE IDEA
    --------
    Accumulate the signal's deviations from its overall mean. While the signal is
    stationary this cumulative sum wanders near zero; once the mean shifts, it
    drifts steadily in one direction. The change-point is where the running
    cumulative sum reaches its extreme (its maximum absolute deviation), and a
    ``threshold`` decides whether that excursion is a real change or noise.

    It is the simplest change detector and the offline twin of Page-Hinkley in
    ``drift``. Best for a SINGLE mean shift in an otherwise stationary series; for
    multiple changes, PELT or binary segmentation is the tool.

    Returns the detected change-point index, or None if none exceeds the threshold.
    """
    x = np.asarray(signal, float)
    mean = x.mean()
    S = np.cumsum(x - mean - drift)         # cumulative deviation from the mean
    if threshold is None:
        threshold = 4 * x.std() * np.sqrt(len(x))
    peak = int(np.argmax(np.abs(S)))
    return peak if np.abs(S[peak]) > threshold else None


def bocpd(signal, hazard=1 / 100, mu0=0.0, kappa0=1.0, alpha0=1.0, beta0=1.0):
    """Bayesian Online Change-Point Detection: a posterior over the last change.

    THE IDEA
    --------
    Track the RUN LENGTH -- how many steps since the last change -- as a
    probability distribution, updated with every new point. Each step, the run
    length either grows by one (no change, probability ``1 - hazard``) or resets
    to zero (a change, probability ``hazard``), and the new data reweights those
    possibilities by how well each predicts it (a Normal-Gamma conjugate model
    supplies the predictive probability). A change-point is a step where the
    posterior mass suddenly collapses onto run-length zero.

    Unlike the point-estimate methods above, BOCPD gives calibrated UNCERTAINTY:
    "there was probably a change around here, with this confidence" -- and it is
    ONLINE, updating in constant time per step. The price is the conjugate-model
    assumption and more machinery.

    Adams & MacKay (2007). Returns, per time step, the most probable run length;
    a drop to ~0 marks a change.
    """
    x = np.asarray(signal, float)
    n = len(x)
    # run-length distribution; R[r] after t steps = P(run length = r)
    R = np.zeros(n + 1)
    R[0] = 1.0
    run_lengths = np.zeros(n, dtype=int)
    # Normal-Gamma sufficient statistics per run length
    mu = np.array([mu0]); kappa = np.array([kappa0])
    alpha = np.array([alpha0]); beta = np.array([beta0])

    from scipy import stats
    for t in range(n):
        # Student-t predictive probability of x[t] for each current run length
        scale = np.sqrt(beta * (kappa + 1) / (alpha * kappa))
        pred = stats.t.pdf(x[t], df=2 * alpha, loc=mu, scale=scale)

        R_prev = R[:t + 1].copy()
        growth = R_prev * pred * (1 - hazard)        # run continues
        cp = np.sum(R_prev * pred * hazard)          # run resets (a change)
        R[:t + 2] = 0
        R[1:t + 2] = growth
        R[0] = cp
        R[:t + 2] /= R[:t + 2].sum() + 1e-300
        run_lengths[t] = int(np.argmax(R[:t + 2]))

        # update the conjugate stats: extend existing runs, prepend the reset
        mu_new = np.concatenate([[mu0], (kappa * mu + x[t]) / (kappa + 1)])
        kappa_new = np.concatenate([[kappa0], kappa + 1])
        alpha_new = np.concatenate([[alpha0], alpha + 0.5])
        beta_new = np.concatenate([[beta0],
                                   beta + kappa * (x[t] - mu) ** 2 / (2 * (kappa + 1))])
        mu, kappa, alpha, beta = mu_new, kappa_new, alpha_new, beta_new

    return run_lengths


__all__ = ["pelt", "binary_segmentation", "cusum", "bocpd"]
