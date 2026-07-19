"""Cumulative/dynamic AUC at ``t``: can the risk score tell who fails by ``t``"""
import numpy as np
from .aalen_additive_hazards import AalenAdditiveHazards
from .aalen_johansen import AalenJohansen
from .weibull_aft import WeibullAFT
from .brier_score import brier_score
from .integrated_brier_score import integrated_brier_score


def time_dependent_auc(durations, events, risk_scores, t):
    """Cumulative/dynamic AUC at ``t``: can the risk score tell who fails by ``t``
    from who survives past it?

    Cases are subjects who experience the event at or before ``t``; controls are
    those known to survive beyond ``t``. The AUC is the probability a random case
    is ranked riskier than a random control -- a time-resolved discrimination
    measure (the concordance index integrated to a horizon).
    """
    t_arr = np.asarray(durations, float)
    e = np.asarray(events, int)
    r = np.asarray(risk_scores, float)
    cases = (t_arr <= t) & (e == 1)
    controls = t_arr > t
    if cases.sum() == 0 or controls.sum() == 0:
        return np.nan
    rc, ro = r[cases], r[controls]
    # P(case risk > control risk), ties count half
    comp = (rc[:, None] > ro[None, :]).sum() + 0.5 * (rc[:, None] == ro[None, :]).sum()
    return float(comp / (len(rc) * len(ro)))


__all__ = ["time_dependent_auc"]
