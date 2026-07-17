# Statistics & inference

Most of the library predicts. These two packages are about *quantifying
uncertainty*: standard errors, p-values, confidence intervals, and coverage
guarantees. That is the line between a prediction library and a statistics one.

## `stats` — inference, not just point estimates

Every other regression in nupyml returns coefficients. `stats` returns
coefficients **with** the machinery to reason about them.

```python
from nupyml.stats import OLS

model = OLS().fit(X, y)
print(model.summary())     # coefficients, SEs, t-stats, p-values, CIs,
                           # R²/adj-R², F-test, AIC/BIC
```

- **Linear models**: `OLS`, `WLS`, `GLS` with a full `.summary()` — standard
  errors from the closed-form σ²(XᵀX)⁻¹ covariance, t-tests, confidence
  intervals, R²/adjusted-R², the overall F-test, AIC/BIC.
- **Generalized linear models**: Gaussian / Binomial / Poisson / Gamma families
  fit by IRLS, with the same inference from the Fisher information.
- **Discrete choice**: `Logit`, `Probit`, `Poisson` — maximum likelihood with
  the observed-information covariance.
- **Robust standard errors**: HC0–HC3 sandwich estimators, for when
  homoskedasticity does not hold.
- **ANOVA**: `anova_lm` for nested model comparison.

### Diagnostics (`stats.diagnostics`)

The tests that tell you whether a regression's assumptions actually hold:

| Test | Checks for |
|---|---|
| Durbin-Watson, Ljung-Box | autocorrelation in residuals |
| Breusch-Pagan, White | heteroskedasticity |
| Jarque-Bera | non-normal residuals |
| ADF, KPSS | (non-)stationarity — note the **opposite** null hypotheses |
| Granger causality | whether one series helps predict another |

These are validated against closed-form values and `scipy.stats`, not
statsmodels — keeping the numpy/scipy-only runtime, and keeping the test suite
runnable without a heavy optional dependency.

## `inference` — Bayesian and distribution-free uncertainty

### Sampling and optimization

- **MCMC**: `MetropolisHastings`, `GibbsSampler`, `HamiltonianMC`.
- **Bayesian optimization**: `BayesianOptimization` over a `GaussianProcessRegressor`.

### Conformal prediction

Wrap **any** model and get prediction intervals (or label sets) with a
*guaranteed* coverage rate under only the exchangeability assumption — no model
of the noise, and the guarantee survives a badly misspecified predictor.

- `ConformalRegressor` / `ConformalClassifier` — split-conformal intervals and
  label sets.
- `MondrianConformalRegressor` — group-conditional intervals that adapt width
  across regions.
- `ConformalizedQuantileRegression` (CQR) — conformalizes a pair of quantile
  regressors, so the band is **adaptive in width** *and* keeps the exact
  finite-sample coverage guarantee.
- `VennAbersCalibrator` — calibrated probabilities as an interval `[p0, p1]`,
  where the width is itself an honest signal of calibration certainty.
- `AdaptiveConformalInference` (ACI) — online conformal that keeps its long-run
  miss rate on target even when the stream **drifts** and exchangeability breaks.

```python
from nupyml.inference import ConformalRegressor
from nupyml.ensemble import RandomForestRegressor

cp = ConformalRegressor(RandomForestRegressor(), random_state=0).fit(X, y)
lo, hi = cp.predict_interval(X_new, coverage=0.9)   # ≥90% coverage, guaranteed
```

The contrast worth internalizing: a model that reports intervals from a *model
of the noise* (a Gaussian process, NGBoost) is only as calibrated as that model.
Conformal makes no such assumption — which is why it is the method to reach for
when the interval has to be trustworthy.
