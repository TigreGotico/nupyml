"""Good-Turing smoothing: estimate the probability of the unseen from singletons."""
from collections import defaultdict


def good_turing_smoothing(counts):
    """Estimate probability of the UNSEEN from the count of singletons (Good, 1953).

    How much probability should a language model reserve for words it has never seen?
    Good-Turing's answer: look at how many events occurred exactly ONCE -- their share
    of the data estimates the total probability of everything unseen. Each observed
    count ``r`` is discounted to ``(r+1) * N_{r+1} / N_r`` so the freed-up mass funds
    the novelties. It underlies Katz back-off and is the classic missing-mass estimate.
    Returns ``(p0, discounted)`` -- unseen mass and a dict of smoothed probabilities.
    """
    counts = {k: int(v) for k, v in dict(counts).items() if v > 0}
    N = sum(counts.values())
    Nr = defaultdict(int)
    for c in counts.values():
        Nr[c] += 1
    p0 = Nr.get(1, 0) / N if N else 0.0                     # missing mass
    smoothed = {}
    for k, r in counts.items():
        star = (r + 1) * Nr.get(r + 1, 0) / Nr[r] if Nr.get(r + 1, 0) else r
        smoothed[k] = star
    tot = sum(smoothed.values())
    # renormalise the seen events to share (1 - p0)
    smoothed = {k: (1 - p0) * v / tot for k, v in smoothed.items()} if tot else smoothed
    return p0, smoothed


__all__ = ["good_turing_smoothing"]
