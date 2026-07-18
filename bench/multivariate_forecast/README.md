# multivariate_forecast

**Kind:** forecast · **Metric:** RMSE (lower is better) · **Floor (ceiling):** 1.2

Three coupled series where each depends on the recent past of **all** of them
(a vector-autoregressive process). Forecast them jointly; using the cross-series
structure should beat forecasting each series alone.

`solve(y_history, horizon) -> (horizon, k) forecast`

Baselines: `var` (VAR(1)), `per_series_ar` (independent AR(1) each), `naive_last`.
VAR wins by exploiting the coupling.
