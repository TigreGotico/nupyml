# volatility_forecast

**Kind:** forecast · **Metric:** RMSE (lower is better) · **Floor (ceiling):** 3.6

Forecast the **variance** of a returns series. Returns are near-unpredictable in the
mean but their variance **clusters** (GARCH dynamics). Given a returns history,
predict the variance of the next `horizon` steps; scored against the realised
squared returns of the held-out window (a noisy but unbiased variance proxy).

`solve(y_history, horizon) -> variance_forecast`

Baselines: `garch` (GARCH(1,1)), `ewma` (RiskMetrics), `historical_var`
(unconditional variance). GARCH wins by tracking the volatility regime.
