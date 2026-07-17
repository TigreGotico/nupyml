"""Kaplan-Meier, Nelson-Aalen, and Cox proportional hazards.

Throughout, the data is ``(durations, events)``: how long each subject was
observed, and whether the event actually happened (1) or the subject was censored
(0) at that time.
"""
import numpy as np
from scipy.optimize import minimize

from ..base import BaseEstimator
from ..utils import check_array


class KaplanMeier(BaseEstimator):
    """The survival curve ``S(t) = P(event has not happened by t)``, censored-aware.

    THE ESTIMATOR
    -------------
    Walk through the distinct event times in order. At each one, multiply the
    running survival by the fraction who made it through::

        S(t) = product over event times t_i <= t of  (1 - d_i / n_i)

    where ``d_i`` is how many had the event at ``t_i`` and ``n_i`` how many were
    still AT RISK just before it.

    Censoring is handled entirely through ``n_i``. A censored subject stays in the
    at-risk count right up until their censoring time, then silently drops out --
    it never triggers a step down in the curve, but it DOES swell the denominator
    for every earlier event, correctly lowering each event's estimated impact.
    That is the whole trick: a censored observation contributes the information it
    genuinely has ("survived at least this long") and no more. The curve steps
    down only at real events and is flat between them.
    """

    def fit(self, durations, events):
        durations = np.asarray(durations, float)
        events = np.asarray(events, int)
        times = np.unique(durations[events == 1])   # steps happen only at events

        survival = 1.0
        self.event_times_ = times
        self.survival_ = np.empty(len(times))
        self.at_risk_ = np.empty(len(times), dtype=int)
        var_sum = 0.0
        self.variance_ = np.empty(len(times))

        for i, t in enumerate(times):
            at_risk = np.sum(durations >= t)         # censored subjects count here
            n_events = np.sum((durations == t) & (events == 1))
            survival *= (1 - n_events / at_risk)
            self.survival_[i] = survival
            self.at_risk_[i] = at_risk
            # Greenwood's formula for the variance of the estimate
            if at_risk > n_events:
                var_sum += n_events / (at_risk * (at_risk - n_events))
            self.variance_[i] = survival ** 2 * var_sum

        self.median_survival_ = self._median()
        return self

    def _median(self):
        """The time the curve first drops to 0.5 -- the median survival time.

        Reported instead of the mean because the mean is often undefined: if the
        curve never reaches zero (some subjects censored at the end), the mean
        survival is infinite, while the median is still a finite, honest summary.
        """
        below = np.where(self.survival_ <= 0.5)[0]
        return float(self.event_times_[below[0]]) if len(below) else np.inf

    def predict(self, t):
        """Survival probability at time(s) ``t`` -- the step function, evaluated."""
        t = np.atleast_1d(np.asarray(t, float))
        out = np.ones(len(t))
        for i, ti in enumerate(t):
            past = self.event_times_ <= ti
            if past.any():
                out[i] = self.survival_[past][-1]     # last step at or before ti
        return out

    def confidence_interval(self, coverage=0.95):
        """Greenwood pointwise band around the survival curve."""
        from scipy.stats import norm
        z = norm.ppf(0.5 + coverage / 2)
        se = np.sqrt(self.variance_)
        return (np.clip(self.survival_ - z * se, 0, 1),
                np.clip(self.survival_ + z * se, 0, 1))


class NelsonAalen(BaseEstimator):
    """The cumulative hazard ``H(t)`` -- Kaplan-Meier's natural companion.

    Where Kaplan-Meier estimates survival by a PRODUCT, Nelson-Aalen estimates
    the cumulative hazard by a SUM::

        H(t) = sum over event times t_i <= t of  d_i / n_i

    The hazard is the instantaneous event rate, and its cumulative version is
    often the more natural thing to model and to compare -- its slope is the
    hazard rate, so a steepening curve means rising risk, read straight off the
    plot. The two are linked by ``S(t) ~ exp(-H(t))``, and Nelson-Aalen is the
    slightly better-behaved estimator in small samples, which is why it is
    preferred when the hazard itself is the quantity of interest.
    """

    def fit(self, durations, events):
        durations = np.asarray(durations, float)
        events = np.asarray(events, int)
        times = np.unique(durations[events == 1])

        self.event_times_ = times
        self.cumulative_hazard_ = np.empty(len(times))
        H = 0.0
        for i, t in enumerate(times):
            at_risk = np.sum(durations >= t)
            n_events = np.sum((durations == t) & (events == 1))
            H += n_events / at_risk               # a SUM, not a product
            self.cumulative_hazard_[i] = H
        return self

    def predict(self, t):
        t = np.atleast_1d(np.asarray(t, float))
        out = np.zeros(len(t))
        for i, ti in enumerate(t):
            past = self.event_times_ <= ti
            if past.any():
                out[i] = self.cumulative_hazard_[past][-1]
        return out


class CoxPH(BaseEstimator):
    """Cox proportional hazards: what covariates DO to the risk.

    THE MODEL
    ---------
    Each subject's hazard is a shared BASELINE hazard scaled by an exponential of
    their covariates::

        hazard_i(t) = baseline(t) * exp(beta . x_i)

    "Proportional" means the ratio of any two subjects' hazards is
    ``exp(beta . (x_i - x_j))`` -- CONSTANT over time. Subject i is, say, twice as
    likely to have the event as subject j, at every t. That single assumption is
    the model's power and its limit: it is what lets ``exp(beta)`` be reported as
    a clean "hazard ratio", and it is wrong whenever an effect grows or fades with
    time (a treatment that helps early and not late).

    THE SEMI-PARAMETRIC TRICK
    -------------------------
    The baseline hazard is left completely UNSPECIFIED -- any shape at all -- and
    yet ``beta`` is still estimated. The device is the PARTIAL likelihood: at each
    event, ask only "among those still at risk, how probable was it THIS subject
    who had the event, given the covariates?" That question compares subjects to
    each other at the same instant, so the baseline hazard -- common to everyone
    at that instant -- cancels out of every term. You get the covariate effects
    without ever estimating the nuisance curve. That cancellation is the whole
    reason Cox regression dominates its field.

    Cox (1972).
    """

    def __init__(self, l2=0.0, max_iter=100):
        self.l2 = l2
        self.max_iter = max_iter

    def _neg_partial_loglik(self, beta, X, order, event_mask):
        """Negative log partial likelihood, with subjects sorted by time.

        Because the data is sorted in ascending time, the risk set of an event
        (everyone with time >= this one) is a SUFFIX of the array, so the sum over
        risk sets is a reverse cumulative sum -- O(n) rather than O(n^2).
        """
        eta = X @ beta
        exp_eta = np.exp(eta)
        # reverse cumsum = sum over the risk set (all later-or-equal times)
        risk_sum = np.cumsum(exp_eta[::-1])[::-1]
        # each event contributes eta_i - log(sum of exp_eta over its risk set)
        ll = np.sum(eta[event_mask] - np.log(risk_sum[event_mask]))
        ll -= self.l2 * (beta @ beta)          # optional ridge on the coefficients
        return -ll

    def _grad(self, beta, X, order, event_mask):
        eta = X @ beta
        exp_eta = np.exp(eta)
        risk_sum = np.cumsum(exp_eta[::-1])[::-1]
        # weighted covariate sums over each risk set, again as a reverse cumsum
        weighted = np.cumsum((exp_eta[:, None] * X)[::-1], axis=0)[::-1]
        expected = weighted / risk_sum[:, None]
        grad = np.sum((X - expected)[event_mask], axis=0)
        grad -= 2 * self.l2 * beta
        return -grad

    def fit(self, X, durations, events):
        X = check_array(X)
        durations = np.asarray(durations, float)
        events = np.asarray(events, int)

        # sort by time so risk sets are suffixes -- the key to the O(n) likelihood
        order = np.argsort(durations)
        Xs = X[order]
        event_mask = events[order].astype(bool)
        self._mean = Xs.mean(axis=0)
        Xc = Xs - self._mean                    # centring aids the optimiser

        res = minimize(self._neg_partial_loglik, np.zeros(X.shape[1]),
                       args=(Xc, order, event_mask),
                       jac=self._grad, method="L-BFGS-B",
                       options={"maxiter": self.max_iter})
        self.coef_ = res.x
        self.hazard_ratios_ = np.exp(res.x)     # the interpretable output
        self.log_partial_likelihood_ = -res.fun

        # Breslow's baseline cumulative hazard, so absolute risks can be recovered
        self._baseline(Xc, durations[order], event_mask)
        return self

    def _baseline(self, Xc, times, event_mask):
        exp_eta = np.exp(Xc @ self.coef_)
        risk_sum = np.cumsum(exp_eta[::-1])[::-1]
        self.baseline_times_ = times[event_mask]
        increments = 1.0 / risk_sum[event_mask]
        self.baseline_cumhaz_ = np.cumsum(increments)

    def predict_risk(self, X):
        """The relative risk score ``exp(beta . x)`` -- higher means sooner."""
        X = check_array(X)
        return np.exp((X - self._mean) @ self.coef_)

    def predict_survival(self, X, t):
        """Absolute survival probability at time ``t``, per subject.

        Combines the fitted covariate effect with Breslow's estimated baseline:
        ``S(t | x) = exp(-H0(t) * exp(beta . x))``. This is where the semi-
        parametric model finally commits to a baseline shape, but only after
        ``beta`` was estimated without one.
        """
        X = check_array(X)
        risk = self.predict_risk(X)
        past = self.baseline_times_ <= t
        H0 = self.baseline_cumhaz_[past][-1] if past.any() else 0.0
        return np.exp(-H0 * risk)


__all__ = ["KaplanMeier", "NelsonAalen", "CoxPH"]
