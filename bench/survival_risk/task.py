"""Survival risk ranking: order subjects by risk under CENSORING.

Survival data is (duration, event) per subject, where ``event=0`` means censored
(we only know they survived at least that long). The submission trains on the
censored training set and outputs a RISK SCORE per test subject (higher = fails
sooner). Scored by the concordance index -- the censoring-aware probability that,
for a comparable pair, the one who failed first was ranked riskier.
"""
import numpy as np

from nupyml.survival import concordance_index

KIND = "survival"
GOAL = ("Rank subjects by risk from censored survival data; scored by the "
        "concordance index.")
METRIC = "concordance_index"
HIGHER_IS_BETTER = True
MIN_SCORE = 0.70


def load():
    rng = np.random.RandomState(0)
    n, d = 600, 5
    X = rng.randn(n, d)
    risk = X @ np.array([1.2, -0.8, 0.5, 0.0, 0.0])   # true risk direction
    base = rng.exponential(scale=np.exp(-risk))        # higher risk -> shorter life
    cens = rng.exponential(scale=1.8, size=n)          # independent censoring
    t = np.minimum(base, cens)
    e = (base <= cens).astype(int)
    perm = rng.permutation(n)
    X, t, e = X[perm], t[perm], e[perm]
    n_tr = int(0.7 * n)
    return (X[:n_tr], t[:n_tr], e[:n_tr], X[n_tr:], t[n_tr:], e[n_tr:])


def metric(durations, events, risk_scores):
    return concordance_index(durations, events, risk_scores)
