# blobs_clustering

**Goal:** recover the generating clusters **without labels**.

- **Kind:** clustering — `solve(X) -> labels`
- **Data:** `make_blobs(n_samples=1000, n_features=5, centers=5, cluster_std=1.2, random_state=0)`. The true assignment is held out for scoring only.
- **Metric:** adjusted Rand index (higher is better; invariant to how you number the clusters).
- **QA floor:** 0.70. Well-separated blobs — a strong entry reaches ~1.0.

Round, separated clusters are exactly k-means' home turf, so k-means, a Gaussian mixture, and Ward agglomeration all score near-perfectly. Harder cluster shapes would separate them; this task is the sanity baseline for the unsupervised family.
