"""Estimating treatment effects from observational data.

The data is ``(X, treatment, outcome)``: covariates, a binary treatment (1 or 0),
and the observed outcome. The target is the AVERAGE TREATMENT EFFECT -- how much
the treatment changes the outcome on average, had it been applied to everyone
versus no one.
"""
import numpy as np

from ..base import BaseEstimator, clone, check_is_fitted
from ..linear_model import LogisticRegression, LinearRegression
from ..utils import check_array, check_random_state


def propensity_score(X, treatment, model=None):
    """P(treated | covariates): each subject's probability of getting treatment.

    THE CENTRAL DEVICE OF OBSERVATIONAL CAUSAL INFERENCE
    ---------------------------------------------------
    Rosenbaum and Rubin's result: to remove confounding you do NOT need to match
    subjects on every covariate -- you only need to match on this ONE number.
    Two subjects with the same propensity score have, on average, the same
    covariate distribution, so comparing a treated and an untreated subject at the
    same propensity is like comparing within a tiny randomised trial.

    That collapse of a whole covariate vector to a single balancing score is what
    makes adjustment tractable in many dimensions, and it is the foundation both
    methods in this file build on.
    """
    X = check_array(X)
    treatment = np.asarray(treatment, int)
    model = LogisticRegression(max_iter=500) if model is None else clone(model)
    model.fit(X, treatment)
    p = model.predict_proba(X)[:, 1]
    # clip away from 0 and 1: a subject almost certain to be treated gets a huge
    # 1/p weight and can single-handedly dominate the estimate. This is the
    # notorious instability of propensity weighting, and clipping is the crude
    # standard guard against it
    return np.clip(p, 1e-3, 1 - 1e-3)


class InversePropensityWeighting(BaseEstimator):
    """Reweight the sample to mimic a randomised trial.

    THE IDEA
    --------
    In observational data the treated and untreated groups are unbalanced -- some
    kinds of people are far likelier to be treated. Weight each subject by the
    INVERSE of the probability of the treatment they actually received::

        weight = 1 / P(their treatment | covariates)

    A treated subject who was UNlikely to be treated is rare and stands in for
    many similar untreated people, so they get a large weight; a treated subject
    who was near-certain to be treated is over-represented and gets a small one.
    Summed up, the weights rebuild the population that a randomised trial would
    have produced -- balanced on the covariates -- from which the treatment effect
    reads off directly.

    THE WEAKNESS
    ------------
    It hinges entirely on the propensity model being right, and it is UNSTABLE:
    when some subjects have propensities near 0 or 1, their inverse weights
    explode and a handful of points dominate the estimate. High variance is the
    price of leaning on the propensity model alone -- which is exactly what
    ``DoublyRobust`` sets out to reduce.
    """

    def __init__(self, propensity_model=None):
        self.propensity_model = propensity_model

    def fit(self, X, treatment, outcome):
        X = check_array(X)
        treatment = np.asarray(treatment, int)
        outcome = np.asarray(outcome, float)

        self.propensity_ = propensity_score(X, treatment, self.propensity_model)
        p = self.propensity_
        # weighted mean outcome among the treated, and among the untreated; the
        # weights rebuild the balanced population for each arm
        treated = treatment == 1
        w = np.where(treated, 1 / p, 1 / (1 - p))
        mean_treated = np.sum(w[treated] * outcome[treated]) / np.sum(w[treated])
        mean_control = np.sum(w[~treated] * outcome[~treated]) / np.sum(w[~treated])
        self.ate_ = float(mean_treated - mean_control)
        return self


class DoublyRobust(BaseEstimator):
    """Two models, and unbiased if EITHER one is right.

    THE IDEA
    --------
    Combine two estimators of the treatment effect:

    * an OUTCOME model, predicting the outcome from covariates and treatment;
    * a PROPENSITY model, predicting who gets treated.

    The doubly-robust estimator is arranged so that its bias vanishes when EITHER
    model is correctly specified -- you do not need both. Two independent chances
    to be right, and you only need to convert one. That is a genuinely stronger
    guarantee than either method alone offers, and it is why doubly-robust
    estimators are the modern default.

    HOW THE TWO COMBINE
    -------------------
    Start from the outcome model's prediction of the effect, then add a
    propensity-weighted CORRECTION built from the outcome model's RESIDUALS::

        effect = outcome_model_prediction
                 + (treatment residual correction, weighted by 1/propensity)

    If the outcome model is right, its residuals are zero-mean and the correction
    vanishes -- the outcome model carries the estimate. If the outcome model is
    wrong but the propensity model is right, the weighting makes the correction
    exactly cancel the outcome model's bias. Either way the bias is removed, which
    is the small miracle the algebra delivers.

    This is the estimating equation behind modern DOUBLE MACHINE LEARNING; there
    the two models are flexible learners fit on separate folds.

    Robins, Rotnitzky & Zhao (1994).
    """

    def __init__(self, outcome_model=None, propensity_model=None):
        self.outcome_model = outcome_model
        self.propensity_model = propensity_model

    def fit(self, X, treatment, outcome):
        X = check_array(X)
        treatment = np.asarray(treatment, int)
        outcome = np.asarray(outcome, float)
        n = len(outcome)

        p = propensity_score(X, treatment, self.propensity_model)

        # outcome models, fitted separately on each arm, then used to predict the
        # COUNTERFACTUAL for everyone (what each subject's outcome would be under
        # each treatment)
        base = LinearRegression() if self.outcome_model is None \
            else self.outcome_model
        m1 = clone(base).fit(X[treatment == 1], outcome[treatment == 1])
        m0 = clone(base).fit(X[treatment == 0], outcome[treatment == 0])
        mu1 = m1.predict(X)          # predicted outcome under treatment, for all
        mu0 = m0.predict(X)          # predicted outcome under control, for all

        # the doubly-robust score, per subject: the outcome-model effect plus the
        # residual correction, weighted by inverse propensity. Averaging gives the
        # ATE, and either model being right makes this unbiased
        dr1 = mu1 + treatment * (outcome - mu1) / p
        dr0 = mu0 + (1 - treatment) * (outcome - mu0) / (1 - p)
        self.individual_effects_ = dr1 - dr0
        self.ate_ = float(np.mean(self.individual_effects_))
        self.propensity_ = p
        return self


class PropensityMatching(BaseEstimator):
    """Pair each treated subject with the untreated one most like them.

    The most intuitive adjustment: for every treated subject, find the untreated
    subject with the nearest PROPENSITY score and compare their outcomes directly.
    Averaging those paired differences estimates the effect on the treated. It is
    transparent -- you can inspect the actual pairs -- which is why it remains
    popular despite being less efficient than weighting, and it degrades
    gracefully when a good match simply does not exist (that treated subject has
    no comparable control, and the caliper drops them rather than forcing a bad
    pair).
    """

    def __init__(self, propensity_model=None, caliper=0.1):
        self.propensity_model = propensity_model
        self.caliper = caliper

    def fit(self, X, treatment, outcome):
        X = check_array(X)
        treatment = np.asarray(treatment, int)
        outcome = np.asarray(outcome, float)

        p = propensity_score(X, treatment, self.propensity_model)
        treated_idx = np.where(treatment == 1)[0]
        control_idx = np.where(treatment == 0)[0]
        control_p = p[control_idx]

        diffs, self.matches_ = [], []
        for i in treated_idx:
            gap = np.abs(control_p - p[i])
            j = np.argmin(gap)
            # the caliper: refuse a match that is not close enough, rather than
            # pairing a treated subject with a control who is nothing like them
            if gap[j] <= self.caliper:
                diffs.append(outcome[i] - outcome[control_idx[j]])
                self.matches_.append((i, int(control_idx[j])))
        self.att_ = float(np.mean(diffs)) if diffs else np.nan     # effect on treated
        self.n_matched_ = len(diffs)
        return self


def uplift_score(model_treated, model_control, X):
    """Per-subject uplift: how much the treatment changes THIS person's outcome.

    The average treatment effect answers "should we treat everyone?". Uplift asks
    the operational question: WHO should we treat? It predicts each individual's
    outcome under treatment and under control and returns the difference, so you
    can target the people the treatment actually helps -- and, importantly, avoid
    the "sleeping dogs" it would hurt. Two outcome models, one per arm, and the
    gap between their predictions is the uplift.
    """
    X = check_array(X)
    return model_treated.predict(X) - model_control.predict(X)


__all__ = ["InversePropensityWeighting", "DoublyRobust", "propensity_score",
           "PropensityMatching", "uplift_score"]
