# node_classification

**Kind:** supervised · **Metric:** accuracy (higher is better) · **Floor:** 0.70

Label graph nodes by their community. A stochastic block model builds three
communities (dense within, sparse across); each node's features are its adjacency
row and its label is its community. Classify held-out nodes from connectivity alone
(a submission may embed the nodes first).

`solve(X_train, y_train, X_test) -> labels`

Baselines: `logistic`, `random_forest`, `knn`.
