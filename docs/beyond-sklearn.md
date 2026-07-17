# Beyond scikit-learn

These packages cover the scikit-learn-*adjacent* ecosystem — the families that
in practice live in statsmodels, pyod, river, gensim, POT, networkx,
imbalanced-learn, scikit-survival, category_encoders, scikit-image, and friends.
Each is a from-scratch, numpy/scipy-only implementation held to the property it
advertises.

(`stats` and `inference` have their own page: [Statistics & inference](statistics.md).)

## Probabilistic graphical models — `pgm`

Discrete Bayesian networks built on an explicit `Factor` algebra
(multiply / marginalize / reduce). Exact inference by **variable elimination**
and **belief propagation**; structure learning by **Chow-Liu** (maximum-mutual-
information spanning tree) and BIC-scored hill-climbing; ancestral sampling. The
count matrices *are* the model — the code makes that concrete.

## Anomaly detection & drift — `anomaly`, `drift`

- **Outliers** (`anomaly`): HBOS (histogram), ECOD / COPOD (empirical-CDF tails),
  KNN, CBLOF (cluster-based, catches clustered anomalies KNN misses), ABOD
  (angle-based, robust in high dimensions).
- **Concept drift** (`drift`): ADWIN (adaptive windowing), DDM / EDDM
  (error-rate and gap), Page-Hinkley (CUSUM), KSWIN (KS-test, catches variance
  shifts). Each answers "has the stream's distribution changed?" for a different
  kind of change.

## Topic models & text ranking — `topic`

LDA by collapsed Gibbs sampling, LSA (truncated SVD of tf-idf), BM25 ranking
(tf saturation + length normalization), Doc2Vec (PV-DBOW), and UMass coherence
for scoring topics.

## Multi-label & active learning — `multilabel`, `active`

- **Multi-label** (`multilabel`): a ladder of increasing correlation-awareness —
  `BinaryRelevance` → `ClassifierChain` → `LabelPowerset` → `RAkEL` → `MLkNN`.
- **Active learning** (`active`): uncertainty / margin / entropy sampling,
  query-by-committee, expected model change, and core-set selection — the
  tension between querying the *uncertain* points and the *diverse* ones.

## Optimal transport & metric learning — `optimal_transport`, `metric_learning`

- **OT**: entropic `sinkhorn`, exact 1-D `wasserstein` (sort-and-match),
  barycenters, OT-based domain adaptation, and the MMD / energy two-sample
  distances.
- **Metric learning**: ITML (LogDet), LFDA (locally-weighted LDA), RCA
  (whiten within-chunklet variation) — complementing the NCA/LMNN already in the
  core.

## Graphs & network embeddings — `graph`

PageRank and HITS, degree / closeness / betweenness / eigenvector centralities,
Louvain and label-propagation community detection, DeepWalk / node2vec
(random walks → word2vec), and the Weisfeiler-Lehman graph kernel. (For learned
graph representations, `gnn` has GCN / GraphSAGE / GAT.)

## Survival, change-point & copulas — `survival`, `changepoint`, `copula`

- **Survival** (`survival`): Kaplan-Meier, Cox proportional hazards, random
  survival forest (log-rank splits), AFT (log-time model), and the
  censoring-aware concordance index.
- **Change-point** (`changepoint`): PELT (exact via pruning), binary
  segmentation, CUSUM, and BOCPD (Bayesian online change-point via a run-length
  posterior).
- **Copulas** (`copula`): Sklar's theorem in practice — model the *shape* of
  dependence separately from the marginals. Gaussian (no tail dependence),
  Student-t (symmetric tails), and the Archimedean family — Clayton (lower),
  Gumbel (upper), Frank (none).

## Category encoders & feature engineering — `encoders`

Target encoders that bound the leakage plain target-encoding invites: WOE
(log-odds, the credit-scoring standard), James-Stein (shrink by the estimate's
standard error), M-estimate, and leave-one-out. Plus target-free binary and
count encoders, and the engineering transformers `Winsorizer`,
`RareLabelEncoder`, and `CyclicalEncoder` (sin/cos so hour 23 neighbors hour 0).

The advanced **feature selectors** live in `feature_selection`: `Boruta`
(shadow features), `mrmr` (relevance minus redundancy), `relieff` (near
hits/misses), and `StabilitySelection` (bootstrap consistency).

## Image & signal features — `image`

The pre-deep-learning vision toolkit, each descriptor invariant to a chosen
nuisance: HOG (shape, invariant to contrast), LBP (texture, invariant to
monotonic lighting), GLCM + Haralick statistics, and Gabor filter banks. Plus
structure tools: the integral image (constant-time box sums), connected-component
labelling, binary morphology, and the Hough line transform.

## Imbalance ensembles — `imbalance`

Beyond the resamplers (SMOTE, ADASYN, Borderline-SMOTE, Tomek links, NearMiss),
ensembles that fold balancing *into* the model so the majority is never
permanently discarded: `BalancedRandomForest` (each tree on a balanced
bootstrap), `RUSBoost` (undersample before each boosting round), and
`EasyEnsemble` (bag independent balanced learners).

## Recommenders — `recommend`

Latent-factor models — `MatrixFactorization`, `ALS`, `BPR`,
`FactorizationMachine`, and `SVDpp` (which also uses *which* items a user rated,
not just how) — alongside memory-based neighborhood CF: `UserBasedCF`,
`ItemBasedCF`, and `SLIM` (a learned sparse item-item model).

## AutoML-lite — `model_selection`

Alongside grid / random / halving search: `HyperbandSearchCV` (hedges across
successive-halving brackets) and `TPESearchCV` (Tree-structured Parzen Estimator
— models the good region and samples toward it).

## Other families already in the library

| Package | Contents |
|---|---|
| `timeseries` | Kalman / EKF / UKF / particle filters, ARIMA, Holt-Winters, STL |
| `sequence` | DTW, edit distances |
| `causal` | IPW, doubly-robust effect estimation |
| `explain` | SHAP, LIME, PDP, integrated gradients |
| `embed` | word2vec, GloVe, BPE, WordPiece |
| `gnn` | GCN, GraphSAGE, GAT |
| `rl` | multi-armed bandits, tabular RL |
| `evolutionary` | CMA-ES, evolution strategies |
| `search` | LSH, IVF, PQ; Bloom filter, count-min sketch, HyperLogLog, t-digest |
| `streaming` | FTRL, Hedge, Hoeffding trees |
| `patterns` | Apriori, FP-Growth, ECLAT |
| `nonparametric` | kernel regression and density tools |
