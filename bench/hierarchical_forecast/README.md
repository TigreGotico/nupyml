# hierarchical_forecast

**Kind:** forecast · **Metric:** RMSE (lower is better) · **Floor (ceiling):** 1.6

Forecast a 4-part hierarchy and its total coherently. The history is `(T, 5)` (total
then the four coupled parts); the forecast is `(horizon, 5)`. Forecasting each series
alone is incoherent; reconciliation (bottom-up / MinT) restores coherence.

`solve(y_history, horizon) -> (horizon, 5) forecast`

Baselines: `reconciled` (independent AR + MinT), `base_independent` (AR per series),
`naive`.
