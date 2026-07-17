# nupyml benchmark scoreboard

Best score per task. Regenerate with `python bench/harness.py`.

| Task | Metric | Best submission | Best score |
|------|--------|-----------------|-----------:|
| [blobs_clustering](blobs_clustering/SCOREBOARD.md) | `adjusted_rand_index` | `baseline_agglomerative` | 1.0000 |
| [diabetes_regression](diabetes_regression/SCOREBOARD.md) | `r2` | `baseline_ridge` | 0.3652 |
| [digits_classification](digits_classification/SCOREBOARD.md) | `accuracy` | `baseline_svc` | 0.9833 |
| [fraud_imbalanced](fraud_imbalanced/SCOREBOARD.md) | `macro_f1` | `baseline_gradient_boosting` | 0.7645 |
| [moons_classification](moons_classification/SCOREBOARD.md) | `accuracy` | `baseline_mlp` | 0.9700 |
| [sine_forecast](sine_forecast/SCOREBOARD.md) | `rmse` | `baseline_holt_winters` | 1.1988 |
