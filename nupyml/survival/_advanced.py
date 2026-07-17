"""Survival analysis expansion: additive hazards, competing risks, parametric
AFT, and time-dependent evaluation.

Survival data is (duration, event) per subject, where ``event=0`` means the
observation was CENSORED (we only know the subject survived at least that long).
These join KM/Nelson-Aalen/Cox/RSF/AFT in the ``survival`` package.
"""
import numpy as np

from ..base import BaseEstimator
from ..utils import check_array


class AalenAdditiveHazards(BaseEstimator):
    """Additive hazards regression -- effects that CHANGE over time.

    THE CONTRAST WITH COX
    ---------------------
    Cox assumes each covariate multiplies the hazard by a constant factor for all
    time (proportional hazards). Aalen's model instead ADDS time-varying
    contributions::

        hazard(t | x) = b_0(t) + b_1(t) x_1 + ... + b_p(t) x_p

    so a covariate's effect can grow, fade, or reverse as time passes -- exactly
    what proportional hazards forbids. You recover the CUMULATIVE regression
    functions ``B_j(t)``, whose SLOPE at time t is the instantaneous effect; a
    flat stretch means the covariate stops mattering there.

    FIT
    ---
    At each event time, a weighted least-squares step on the at-risk subjects
    gives the increment ``dB``; accumulating those increments over event times is
    the estimator. No likelihood, no proportionality -- just a sequence of linear
    regressions down the risk set.

    Aalen (1989).
    """

    def fit(self, X, durations, events):
        X = check_array(X)
        durations = np.asarray(durations, float)
        events = np.asarray(events, int)
        n, p = X.shape
        Xd = np.column_stack([np.ones(n), X])           # intercept = baseline
        order = np.argsort(durations)
        self.times_ = []
        increments = []
        B = np.zeros(p + 1)
        for i in order:
            if events[i] != 1:
                continue
            at_risk = durations >= durations[i]         # still-alive subjects
            Xr = Xd[at_risk]
            dN = np.zeros(at_risk.sum())
            # the subject having the event is the one at (its position in Xr)
            risk_idx = np.where(at_risk)[0]
            dN[np.where(risk_idx == i)[0][0]] = 1.0
            # least-squares increment: (X'X)^-1 X' dN
            G = Xr.T @ Xr + 1e-6 * np.eye(p + 1)
            dB = np.linalg.solve(G, Xr.T @ dN)
            B = B + dB
            self.times_.append(durations[i])
            increments.append(B.copy())
        self.times_ = np.array(self.times_)
        self.cumulative_coef_ = np.array(increments) if increments else np.zeros((0, p + 1))
        return self

    def predict_cumulative_hazard(self, X, t=None):
        X = check_array(X)
        Xd = np.column_stack([np.ones(len(X)), X])
        if len(self.times_) == 0:
            return np.zeros(len(X))
        if t is None:
            t = self.times_[-1]
        k = np.searchsorted(self.times_, t, side="right") - 1
        k = max(0, k)
        return Xd @ self.cumulative_coef_[k]


class AalenJohansen(BaseEstimator):
    """Cumulative incidence under COMPETING RISKS (Aalen-Johansen estimator).

    THE COMPETING-RISKS PROBLEM
    ---------------------------
    When a subject can fail from several MUTUALLY EXCLUSIVE causes (die of cancer
    OR heart disease), a naive Kaplan-Meier for one cause -- treating the other
    causes as censoring -- OVERSTATES that cause's incidence, because it pretends
    the subjects lost to the other cause could still have failed from this one.
    The Aalen-Johansen cumulative incidence function fixes this::

        CIF_k(t) = sum_{t_i <= t} S(t_{i-1}) * (d_{k,i} / n_i)

    -- each cause-k event is weighted by the OVERALL survival up to just before
    it, so a subject already dead from another cause cannot contribute. The CIFs
    of all causes sum to ``1 - S(t)``, as they must.

    ``events`` here carries the CAUSE (0 = censored, 1..K = cause of failure).
    ``cif_`` maps each cause to its (times, incidence) curve.
    """

    def fit(self, durations, events):
        durations = np.asarray(durations, float)
        events = np.asarray(events, int)
        causes = [c for c in np.unique(events) if c != 0]
        uniq_t = np.unique(durations[events != 0])
        n = len(durations)
        S = 1.0                                         # overall survival
        cif = {c: 0.0 for c in causes}
        self.cif_ = {c: {"time": [], "incidence": []} for c in causes}
        self.times_ = uniq_t
        for t in uniq_t:
            at_risk = (durations >= t).sum()
            if at_risk == 0:
                continue
            d_total = ((durations == t) & (events != 0)).sum()
            for c in causes:
                d_c = ((durations == t) & (events == c)).sum()
                cif[c] += S * (d_c / at_risk)           # weighted by prior survival
                self.cif_[c]["time"].append(t)
                self.cif_[c]["incidence"].append(cif[c])
            S *= (1 - d_total / at_risk)                # update overall survival
        for c in causes:
            self.cif_[c] = {k: np.array(v) for k, v in self.cif_[c].items()}
        return self

    def predict_cif(self, cause, t):
        c = self.cif_[cause]
        if len(c["time"]) == 0:
            return 0.0
        k = np.searchsorted(c["time"], t, side="right") - 1
        return 0.0 if k < 0 else float(c["incidence"][k])


class WeibullAFT(BaseEstimator):
    """Parametric accelerated failure time with a Weibull baseline.

    THE AFT VIEW
    ------------
    Instead of modelling the hazard (Cox), model log-time directly::

        log T = x·beta + sigma * W,   W ~ standard Gumbel (Weibull baseline)

    so covariates ACCELERATE or decelerate the time to failure by a constant
    factor ``exp(x·beta)`` -- a description many people find more intuitive than a
    hazard ratio ("this treatment doubles expected survival time"). Being fully
    parametric, it extrapolates smoothly beyond the observed follow-up, where Cox
    and Kaplan-Meier simply stop.

    FIT
    ---
    Maximum likelihood over ``beta`` and ``log sigma``, with censored subjects
    contributing their survival ``S(t)`` and events their density ``f(t)``.
    Optimised with L-BFGS.
    """

    def __init__(self, max_iter=200):
        self.max_iter = max_iter

    def fit(self, X, durations, events):
        from scipy.optimize import minimize
        X = check_array(X)
        t = np.asarray(durations, float)
        e = np.asarray(events, int)
        logt = np.log(t + 1e-8)
        n, p = X.shape
        Xd = np.column_stack([np.ones(n), X])

        def nll(params):
            beta = params[:p + 1]
            sigma = np.exp(params[p + 1])
            z = (logt - Xd @ beta) / sigma
            log_S = -np.exp(z)                          # log survival
            log_f = z - np.log(sigma) - np.log(t + 1e-8) + log_S   # log density
            return -np.sum(e * log_f + (1 - e) * log_S)

        init = np.concatenate([np.zeros(p + 1), [0.0]])
        res = minimize(nll, init, method="L-BFGS-B",
                       options={"maxiter": self.max_iter})
        self.coef_ = res.x[:p + 1]
        self.sigma_ = float(np.exp(res.x[p + 1]))
        return self

    def predict_median(self, X):
        X = check_array(X)
        Xd = np.column_stack([np.ones(len(X)), X])
        # median of the Weibull AFT: T = exp(x·beta) * (ln 2)^sigma
        return np.exp(Xd @ self.coef_) * (np.log(2) ** self.sigma_)

    def predict_survival(self, X, t):
        X = check_array(X)
        Xd = np.column_stack([np.ones(len(X)), X])
        z = (np.log(t + 1e-8) - Xd @ self.coef_) / self.sigma_
        return np.exp(-np.exp(z))


# --- time-dependent evaluation --------------------------------------------

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


__all__ = ["AalenAdditiveHazards", "AalenJohansen", "WeibullAFT",
           "brier_score", "integrated_brier_score", "time_dependent_auc"]
