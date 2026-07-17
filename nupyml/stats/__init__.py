"""Statistical inference: not just a prediction, but how sure we are of it.

THE GAP THIS FILLS
------------------
Everything else in this library gives POINT estimates -- a coefficient, a
prediction, a cluster label. This module answers the question a statistician asks
next: *how certain is that?* For a fitted coefficient it reports a standard
error, a t- or z-statistic, a p-value, and a confidence interval; for a whole
model it reports R^2, F-tests, AIC/BIC, and diagnostic tests of the assumptions
the inference rests on.

That shift -- from "what is the estimate" to "what does the estimate let me
CONCLUDE" -- is the whole point of ``statsmodels``, and it is what separates a
machine-learning library from a statistics one. A prediction library asks "does
it generalise?"; an inference library asks "is this effect real, and how big
could it plausibly be?".

WHERE THE UNCERTAINTY COMES FROM
--------------------------------
Every standard error here traces to one idea: the sampling distribution of the
estimator. Fit the same model on a different sample and the coefficients would
come out slightly different; the standard error is the spread of that
distribution, and everything else (t-stats, p-values, CIs) is read off it.

* For OLS it has a closed form: ``Var(beta) = sigma^2 (X'X)^-1``.
* For maximum-likelihood models (Logit, Poisson, ...) it is the inverse of the
  observed information -- the curvature of the log-likelihood at its peak. A
  sharply-peaked likelihood pins the parameter down (small SE); a flat one leaves
  it uncertain (large SE). This is the Fisher-information idea made concrete.

THE MODULES
-----------
* ``regression.py`` -- ``OLS``/``WLS`` with the full closed-form inference; GLM
  families and discrete-choice models (``Logit``, ``Probit``, ``Poisson``) by
  IRLS/MLE with the information-matrix covariance; robust (heteroskedasticity-
  consistent) standard errors; ``anova``; a printed ``.summary()``.
* ``diagnostics.py`` -- the tests that check whether the inference is even valid:
  stationarity (ADF, KPSS), autocorrelation (Durbin-Watson, Ljung-Box),
  heteroskedasticity (Breusch-Pagan, White), normality (Jarque-Bera), and
  Granger causality.

A WARNING
---------
Inference is only as trustworthy as its assumptions. A beautiful p-value from a
model whose residuals are autocorrelated or heteroskedastic is a confident lie --
which is exactly why ``diagnostics.py`` exists alongside ``regression.py``. Fit,
then check the assumptions, then believe the p-values, in that order.
"""
from .regression import OLS, WLS, GLM, Logit, Probit, Poisson, anova_lm
from .diagnostics import (durbin_watson, ljung_box, breusch_pagan, white_test,
                          jarque_bera, adf_test, kpss_test, granger_causality)

__all__ = [
    "OLS", "WLS", "GLM", "Logit", "Probit", "Poisson", "anova_lm",
    "durbin_watson", "ljung_box", "breusch_pagan", "white_test", "jarque_bera",
    "adf_test", "kpss_test", "granger_causality",
]
