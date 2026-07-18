# timeseries_anomaly

**Kind:** clustering · **Metric:** ROC-AUC (higher is better) · **Floor:** 0.60

Flag the unusual timesteps in a seasonal series corrupted by spikes and level jumps.
Each timestep has local features (value, deviation from a rolling mean, local
volatility); return an anomaly score per timestep so injected anomalies rank high.

`solve(X) -> anomaly_scores`

Baselines: `isolation_forest`, `zscore`, `deviation`.
