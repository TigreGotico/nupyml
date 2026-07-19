"""Average the Brier score over a grid of times -- one summary of calibration."""
import numpy as np
from .brier_score import brier_score


def integrated_brier_score(durations, events, survival_pred_fn, times):
    """Average the Brier score over a grid of times -- one summary of calibration.

    ``survival_pred_fn(t)`` returns the predicted survival probabilities at time
    ``t`` for all subjects. Integrating over ``times`` (trapezoidally) gives the
    IBS, the standard scalar for comparing survival models' probabilistic accuracy.
    """
    times = np.asarray(times, float)
    bs = np.array([brier_score(durations, events, survival_pred_fn(t), t)
                   for t in times])
    return float(np.trapezoid(bs, times) / (times[-1] - times[0]))


__all__ = ["integrated_brier_score"]
