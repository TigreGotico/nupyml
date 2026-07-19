# hyperbolic_hierarchy

**Goal:** embed a tree so that pairwise **graph distances** are preserved.

- **Kind:** supervised — `solve(X_train, y_train, X_test) -> y_pred`. Each row of `X` is a node pair `(i, j)`; `y` is its tree hop distance.
- **Data:** a balanced depth-4 binary tree. Training carries every edge plus half the non-edge pairs; the other half is held out.
- **Metric:** Spearman rank correlation between predicted and true distances (higher is better).
- **QA floor:** 0.55.

Trees embed with low distortion in **hyperbolic** space but poorly in Euclidean space. `LorentzEmbedding` and `PoincareEmbedding` clear the floor; reconstructing the graph and running BFS is a near-perfect reference.
