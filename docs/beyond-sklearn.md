# Beyond scikit-learn

These packages cover the scikit-learn-*adjacent* ecosystem, the families that
in practice live in statsmodels, pyod, river, gensim, POT, networkx,
imbalanced-learn, scikit-survival, category_encoders, scikit-image, and friends.
Each is a from-scratch, numpy/scipy-only implementation held to the property it
advertises.

(`stats` and `inference` have their own page: [Statistics & inference](statistics.md).)

## Probabilistic graphical models (`pgm`)

Discrete Bayesian networks built on an explicit `Factor` algebra
(multiply / marginalize / reduce). Exact inference by **variable elimination**
and **belief propagation**, approximate inference on loopy graphs by **loopy
belief propagation**, structure learning by **Chow-Liu** (maximum-mutual-
information spanning tree) and BIC-scored hill-climbing, ancestral sampling. The
`GaussianBayesianNetwork` handles continuous linear-Gaussian nodes with exact
closed-form conditioning, and the hidden semi-Markov model (`hmm`) adds explicit
state durations to the HMM. The count matrices *are* the model. The code makes
that concrete.

## Anomaly detection & drift (`anomaly`, `drift`)

- **Outliers** (`anomaly`): HBOS (histogram), ECOD / COPOD (empirical-CDF tails),
  KNN, CBLOF (cluster-based, catches clustered anomalies KNN misses), ABOD
  (angle-based, stable in high dimensions), LODA, feature bagging, half-space
  trees, Mahalanobis and PCA-reconstruction detectors. For OOD detection:
  the **extended isolation forest** (oblique cuts), the **isolation kernel**
  (data-dependent similarity), **Deep SVDD** (learned smallest-ball one-class),
  and the **energy score** (post-hoc OOD from a classifier's logits).
- **Concept drift** (`drift`): ADWIN (adaptive windowing), DDM / EDDM
  (error-rate and gap), Page-Hinkley (CUSUM), KSWIN (KS-test, catches variance
  shifts). Each answers "has the stream's distribution changed?" for a different
  kind of change.

## Topic models, ranking & language (`topic`, `text`)

- **Topics** (`topic`): LDA by collapsed Gibbs sampling, LSA (truncated SVD of
  tf-idf), BM25 ranking (tf saturation + length normalization), Doc2Vec
  (PV-DBOW), SIF sentence embeddings, and UMass coherence for scoring topics.
- **Language** (`text`): **Kneser-Ney** n-gram smoothing (continuation
  probabilities), the **unigram** (SentencePiece-style) subword tokenizer,
  **TextRank** / **LexRank** summarisation, **TextTiling** topic segmentation,
  **RAKE** and **PMI** keyphrase extraction, the **MEMM** / **HMM** /
  **averaged-perceptron** taggers, **beam search** decoding, the **Norvig spell
  corrector**, **Word Mover's Distance** (OT over word embeddings), and
  **BLEU** / **ROUGE** / **chrF** / **METEOR** generation metrics (with
  `LinearChainCRF` in `sequence`).

## Multi-label & active learning (`multilabel`, `active`)

- **Multi-label** (`multilabel`): a ladder of increasing correlation-awareness,
  `BinaryRelevance` → `ClassifierChain` → `LabelPowerset` → `RAkEL` → `MLkNN`.
- **Active learning** (`active`): uncertainty / margin / entropy sampling,
  query-by-committee, expected model change, and core-set selection, the
  tension between querying the *uncertain* points and the *diverse* ones.

## Optimal transport & metric learning (`optimal_transport`, `metric_learning`)

- **OT**: entropic `sinkhorn`, exact 1-D `wasserstein` (sort-and-match),
  **unbalanced** OT, **sliced** Wasserstein, **Gromov-Wasserstein** (structure-
  only matching), barycenters and barycentric mapping, OT-based domain
  adaptation, and the MMD / energy two-sample distances.
- **Metric learning**: ITML (LogDet), LFDA (locally-weighted LDA), RCA
  (whiten within-chunklet variation), complementing the NCA/LMNN already in the
  core.

## Graphs & network embeddings (`graph`)

PageRank and HITS, degree / closeness / betweenness / eigenvector centralities,
Louvain and label-propagation community detection, **link prediction** (common
neighbours, Jaccard, Adamic-Adar, Katz), Girvan-Newman, max-flow / min-cut, and
the Weisfeiler-Lehman / random-walk graph kernels. **Embeddings**: DeepWalk /
node2vec / LINE (random walks → word2vec), plus **struc2vec** (structural-role
embeddings) and **metapath2vec** (heterogeneous graphs) in `embed`, which also
has **LSH** approximate nearest neighbours and the feature-hashing trick. (For
learned graph representations, `gnn` has GCN / GraphSAGE / GAT.)

## Survival, change-point & copulas (`survival`, `changepoint`, `copula`)

- **Survival** (`survival`): Kaplan-Meier, Cox proportional hazards, random
  survival forest (log-rank splits), AFT / Weibull-AFT (log-time models), Aalen
  additive hazards, Aalen-Johansen competing risks, and censoring-aware
  evaluation (concordance, IPCW / integrated Brier, time-dependent AUC).
- **Change-point** (`changepoint`): PELT (exact via pruning), binary
  segmentation, CUSUM, and BOCPD (Bayesian online change-point via a run-length
  posterior).
- **Copulas** (`copula`): Sklar's theorem in practice, model the *shape* of
  dependence separately from the marginals. Gaussian (no tail dependence),
  Student-t (symmetric tails), and the Archimedean family, Clayton (lower),
  Gumbel (upper), Frank (none).

## Category encoders & feature engineering (`encoders`)

Target encoders that bound the leakage plain target-encoding invites: WOE
(log-odds, the credit-scoring standard), James-Stein (shrink by the estimate's
standard error), M-estimate, and leave-one-out. Plus target-free binary and
count encoders, and the engineering transformers `Winsorizer`,
`RareLabelEncoder`, `CyclicalEncoder` (sin/cos so hour 23 neighbors hour 0),
`MDLPDiscretizer` (supervised, entropy-driven cut points), and `DateTimeFeatures`
(calendar fields + their cyclical encodings from a timestamp).

The advanced **feature selectors** live in `feature_selection`: `Boruta`
(shadow features), `mrmr` (relevance minus redundancy), `relieff` (near
hits/misses), and `StabilitySelection` (bootstrap consistency).

## Image (`image`)

The pre-deep-learning vision toolkit, each descriptor invariant to a chosen
nuisance: HOG (shape, invariant to contrast), LBP (texture, invariant to
monotonic lighting), GLCM + Haralick statistics, and Gabor filter banks. Plus
structure tools: the integral image (constant-time box sums), connected-component
labelling, binary morphology, and the Hough line transform.

**Feature & geometry** (v2-v4): Harris & **FAST** corners, Canny edges, template
matching, Lucas-Kanade & **Horn-Schunck** (dense) optical flow, the chamfer
distance transform, non-maximum suppression, **LoG blob detection**, SLIC /
**Felzenszwalb** / **mean-shift** segmentation, Gaussian/Laplacian **pyramids**,
**watershed**, **active contours** (balloon snakes), **ORB** and **SIFT**
descriptors + matching, **bag of visual words**, content-aware **seam carving**,
**Hough circles**, **homography** (DLT + RANSAC), the **fundamental matrix**
(8-point + RANSAC, epipolar geometry), and the **Viola-Jones** Haar-cascade detector.

## Imbalance ensembles (`imbalance`)

Beyond the resamplers (SMOTE, ADASYN, Borderline-SMOTE, Tomek links, NearMiss),
ensembles that fold balancing *into* the model so the majority is never
permanently discarded: `BalancedRandomForest` (each tree on a balanced
bootstrap), `RUSBoost` (undersample before each boosting round), and
`EasyEnsemble` (bag independent balanced learners).

## Recommenders (`recommend`)

Latent-factor models: `MatrixFactorization`, `ALS`, `BPR`,
`FactorizationMachine`, and `SVDpp` (which also uses *which* items a user rated,
not just how), memory-based neighborhood CF: `UserBasedCF`, `ItemBasedCF`,
`SlopeOne`, co-clustering CF, and `SLIM` (a learned sparse item-item model), and
neural recommenders: `NeuralCF`, `DeepFM`, and `GRU4Rec` (session-based).

## AutoML-lite & meta-learning (`model_selection`, `meta`)

Alongside grid / random / halving search: `HyperbandSearchCV` (hedges across
successive-halving brackets), `BOHBSearchCV`, and `TPESearchCV` (Tree-structured
Parzen Estimator: models the good region and samples toward it). The `meta`
package adds few-shot learning, `PrototypicalNetwork`, `MatchingNetwork`, and
gradient-based meta-learners `Reptile` and `MAML` that learn an initialisation a
few steps can adapt to any task, plus `extract_meta_features` for algorithm
selection.

## Other families already in the library

| Package | Contents |
|---|---|
| `timeseries` | Kalman / EKF / UKF / particle filters, ARIMA, **SARIMA**, **VECM**, **dynamic factor model**, Holt-Winters, STL, Croston (+ **SBA**/**TSB**), Theta, SSA, SAX, DBA, **GARCH**, **VAR**, **AutoETS**, **Prophet-style**, **MSTL**, **TBATS**, **Bayesian structural TS**, **Kalman-EM**, hierarchical **reconciliation** (MinT), **N-BEATS**, **DeepAR** |
| `tsclass` | ROCKET, BOSS, shapelet transform, k-NN-DTW, matrix profile (motifs/discords) |
| `sequence` | DTW, edit distances, linear-chain CRF, structured perceptron |
| `causal` | IPW, `DoublyRobust`, instrumental variables, double-ML, T/X/**R-learners**, **Causal Forest**, **Regression Discontinuity**, synthetic control |
| `explain` | SHAP, LIME, PDP, integrated gradients, EBM/ALE/H-statistic, anchors, counterfactuals |
| `fairness` | group metrics (DP/EO/equal-opportunity), reweighing, correlation remover, exponentiated-gradient reduction, threshold optimizer |
| `embed` | word2vec, GloVe, **FastText**, **Poincaré**, StarSpace, item2vec, PPMI+SVD, BPE, WordPiece, **LSH**/**HNSW**/**IVFPQ** ANN, feature & **hash** embeddings, struc2vec, metapath2vec, **GraphWave**, **NetMF**, **GraRep**, **HOPE**, **personalized-PageRank** |
| `gnn` | GCN, GraphSAGE, GAT |
| `rl` | multi-armed bandits (v2), tabular RL, deep RL (DQN/PPO), continuous control (DDPG/TD3/SAC), GAE, MCTS |
| `evolutionary` | CMA-ES, ES, differential evolution (+ **SHADE**), PSO, genetic programming (+ **Cartesian**/**gene-expression**), symbolic regression, NSGA-II, **SPEA2**, island model, novelty search, MAP-Elites, **NEAT**, **MOEA/D**, grammatical GP, ant colony, **harmony search**, **memetic**, **lexicase**, **UMDA** / **compact-GA** (EDAs), **grey wolf** |
| `optimize` | LP simplex, QP, conjugate gradient, FISTA, ADMM, Nelder-Mead, **L-BFGS**, **Levenberg-Marquardt**, **Frank-Wolfe**, **Powell**, **cross-entropy method**, **trust-region Newton-CG**, **interior-point QP**, **SLSQP**, **augmented Lagrangian**, **SPSA**, **SVRG**/**SAGA**, **mirror descent**, **OWL-QN**/**proximal-Newton**, **basin-hopping** |
| `signal` | wavelets / DWT, STFT / spectrogram / mel / MFCC, EMD, peaks / envelope / ZCR |
| `tensor` / `matrix` | CP & Tucker decompositions, `RobustPCA`, soft-impute, singular-value thresholding |
| `ordinal` / `nonparametric` | proportional-odds & rank-ridge ordinal regression, kernel regression, GAMs / splines, density tools |
| `ranking` | NDCG / MAP / MRR / hit-rate metrics, LambdaMART, RankNet |
| `search` | LSH, IVF, PQ, Bloom filter, count-min sketch, HyperLogLog, t-digest |
| `streaming` | FTRL, Hedge, Hoeffding trees |
| `patterns` | Apriori, FP-Growth, ECLAT |

---
[← Statistics & inference](statistics.md) · [Home](index.md) · [The benchmark suite →](benchmarks.md)
