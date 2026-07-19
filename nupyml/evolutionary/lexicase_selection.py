"""Select a parent CASE BY CASE, keeping specialists (Spector, 2012)."""
import numpy as np


def lexicase_selection(errors, rng=None):
    """Select a parent CASE BY CASE, keeping specialists (Spector, 2012).

    Averaging performance across training cases rewards jack-of-all-trades and
    quietly eliminates individuals that are the ONLY ones solving some hard case.
    Lexicase selection avoids the average entirely: shuffle the cases, then filter the
    candidate pool one case at a time, keeping only those with the BEST error on the
    current case, until a single individual remains. Because the case order is random
    each selection, specialists on rare-but-important cases regularly survive, which
    preserves the diversity that keeps evolution from stalling. ``errors`` is an
    (n_individuals, n_cases) array (lower = better); returns the selected index.
    """
    rng = rng if rng is not None else np.random.RandomState()
    if not isinstance(rng, np.random.RandomState):
        rng = np.random.RandomState(rng)
    errors = np.asarray(errors, float)
    candidates = np.arange(len(errors))
    cases = rng.permutation(errors.shape[1])
    for c in cases:
        best = errors[candidates, c].min()
        candidates = candidates[errors[candidates, c] <= best + 1e-12]
        if len(candidates) == 1:
            break
    return int(rng.choice(candidates))


__all__ = ["lexicase_selection"]
