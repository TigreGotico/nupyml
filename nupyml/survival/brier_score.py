"""IPCW Brier score at time ``t``: calibrated squared error under censoring."""
import numpy as np


def _censoring_survival(durations, events):
    """Kaplan-Meier of the CENSORING distribution G(t) -- needed for IPCW."""
    t = np.asarray(durations, float)
    e = np.asarray(events, int)
    order = np.argsort(t)
    ts, G, g = np.unique(t), [], 1.0
    times, surv = [], []
    for u in ts:
        at_risk = (t >= u).sum()
        d_cens = ((t == u) & (e == 0)).sum()           # censoring "events"
        if at_risk > 0:
            g *= (1 - d_cens / at_risk)
        times.append(u); surv.append(g)
    return np.array(times), np.array(surv)


def brier_score(durations, events, survival_pred, t, train_durations=None,
                train_events=None):
    """IPCW Brier score at time ``t``: calibrated squared error under censoring.

    The Brier score is the mean squared error between the predicted survival
    probability at ``t`` and the true 0/1 outcome (survived past ``t`` or not).
    Under censoring some outcomes are unknown, so each known outcome is REWEIGHTED
    by the inverse probability of still being uncensored (IPCW) -- which corrects
    the bias that censored subjects would otherwise introduce. Lower is better.
    """
    t_arr = np.asarray(durations, float)
    e = np.asarray(events, int)
    S = np.asarray(survival_pred, float)               # predicted P(T>t) per subject
    gt, gs = _censoring_survival(
        train_durations if train_durations is not None else durations,
        train_events if train_events is not None else events)

    def G(u):
        k = np.searchsorted(gt, u, side="right") - 1
        return gs[k] if k >= 0 else 1.0

    score = 0.0
    for i in range(len(t_arr)):
        if t_arr[i] <= t and e[i] == 1:                # failed before t
            score += (0 - S[i]) ** 2 / max(G(t_arr[i]), 1e-6)
        elif t_arr[i] > t:                             # survived past t
            score += (1 - S[i]) ** 2 / max(G(t), 1e-6)
        # censored before t: outcome unknown, contributes 0 (weight handles it)
    return score / len(t_arr)


__all__ = ["brier_score"]
