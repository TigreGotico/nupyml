# image_segmentation

**Kind:** clustering · **Metric:** adjusted Rand index (higher is better) · **Floor:** 0.55

Segment a synthetic image into its colour regions. Each pixel is a feature vector of
its `(row, col)` position and intensity; cluster the pixels back into regions.

`solve(X) -> labels`

Baselines: `kmeans`, `gmm`, `agglomerative`.
