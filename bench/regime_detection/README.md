# regime_detection

**Kind:** clustering · **Metric:** adjusted Rand index (higher is better) · **Floor:** 0.30

Segment a one-dimensional series into its hidden regimes (a Markov-switching process
alternates between a low-mean/low-vol and a high-mean/high-vol state). Each timestep
is described by rolling statistics; cluster them back into the true regime, unsupervised.

`solve(X) -> labels`

Baselines: `kmeans`, `gmm`, `threshold` (level-based reference).
