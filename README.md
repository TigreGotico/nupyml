# nupyml

A scikit-learn-like machine learning **and** neural network library whose only
runtime dependencies are **numpy** and **scipy**.

Everything is implemented from scratch: a tape-based reverse-mode autograd
engine powers the neural network stack, and the classic estimators follow the
familiar `fit` / `predict` / `transform` API with `get_params` / `set_params` /
`clone`, so `Pipeline` and the `*SearchCV` estimators work with every one of
them.

**This is written to be read.** The goal is a single place to study how these
algorithms actually work, with nothing hidden behind a compiled extension — if
you can read numpy, you can read every line of every model here. Optimizations
are not avoided, but they are explained rather than assumed: where the fast way
differs from the obvious way, the code says why.

```bash
pip install numpy scipy
pip install -e .
```

Full documentation lives in [`docs/`](docs/index.md): a getting-started guide,
a tour of every package, and a reading guide to the parts most worth studying.

## What's inside

### Neural networks (`nupyml.nn`, `nupyml.autograd`)

- **Autograd**: `Tensor` with topological-sort backprop; every op is
  gradient-checked against finite differences in the test suite.
- **Layers**: `Linear`, `Conv2d` (im2col), `MaxPool2d`/`AvgPool2d`,
  `BatchNorm1d/2d`, `LayerNorm`, `Dropout`, `Embedding`, `RNN`/`GRU`/`LSTM`,
  `MultiHeadAttention`, `TransformerEncoderLayer`, `PositionalEncoding`.
- **Losses**: MSE, MAE, Huber, cross-entropy (fused log-softmax), BCE,
  BCE-with-logits.
- **Optimizers**: SGD (+momentum/Nesterov), Adam, AdamW, RMSprop; StepLR /
  cosine / warmup schedulers; gradient clipping.
- **sklearn-style wrappers**: `MLPClassifier`, `MLPRegressor` with `adam`,
  `sgd` and `lbfgs` solvers, `early_stopping` and `warm_start`.
- Model persistence via `model.save("weights.npz")` / `model.load(...)`.

### Classic ML

| Package | Estimators |
|---|---|
| `linear_model` | LinearRegression, Ridge, Lasso, ElasticNet, LogisticRegression, Perceptron, SGD*; **robust**: Huber, Quantile (LP), TheilSen, RANSAC; **Bayesian**: BayesianRidge, ARDRegression; **GLMs**: Poisson, Gamma, Tweedie (log/identity links); **CV paths**: RidgeCV, LassoCV, ElasticNetCV, LogisticRegressionCV; OrthogonalMatchingPursuit |
| `tree` | DecisionTreeClassifier/Regressor — vectorized CART, cost-complexity pruning (`ccp_alpha`), native NaN routing, monotonic constraints |
| `ensemble` | RandomForest, ExtraTrees, Bagging, AdaBoost (SAMME), GradientBoosting, **HistGradientBoosting** (binned, second-order, NaN + categorical support), Voting, **Stacking** |
| `svm` | SVC (SMO, one-vs-one, `probability=True` via Platt), SVR, **NuSVC**, **NuSVR**, LinearSVC |
| `neighbors` | KNeighborsClassifier/Regressor, NearestNeighbors, KernelDensity |
| `naive_bayes` | Gaussian, Multinomial, Bernoulli, Complement, **Categorical** |
| `cluster` | KMeans (k-means++), MiniBatchKMeans, DBSCAN, Agglomerative, MeanShift, Spectral, **Birch**, **OPTICS**, **AffinityPropagation**, **HDBSCAN** |
| `mixture` | GaussianMixture (log-space EM), **BayesianGaussianMixture** (variational, Dirichlet / Dirichlet-process priors) |
| `decomposition` | PCA, TruncatedSVD, NMF, FastICA, KernelPCA, **SparsePCA**, **DictionaryLearning**, **SparseCoder**, **MiniBatchNMF**, **LatentDirichletAllocation** |
| `manifold` | TSNE (exact), **TSNEBarnesHut**, Isomap, MDS, LocallyLinearEmbedding, **SpectralEmbedding** |
| `gaussian_process` | GPR, GPC (Laplace); **composable kernels** — RBF (with ARD), Matérn, RationalQuadratic, ExpSineSquared, DotProduct, White, Constant, combined with `+` and `*` |
| `outlier` | **IsolationForest**, **LocalOutlierFactor**, **OneClassSVM**, **EllipticEnvelope** (MCD) |
| `semi_supervised` | **LabelPropagation**, **LabelSpreading**, **SelfTrainingClassifier** |
| `calibration` / `isotonic` | **CalibratedClassifierCV** (sigmoid + isotonic), **IsotonicRegression** |
| `multiclass` | **OneVsRest**, **OneVsOne**, **OutputCode**, **MultiOutput**\*, **ClassifierChain** |
| `feature_selection` | SelectKBest/Percentile/Fpr, VarianceThreshold, RFE, RFECV, SelectFromModel, SequentialFeatureSelector; `f_classif`, `f_regression`, `chi2`, `mutual_info_*` |
| `impute` | SimpleImputer, KNNImputer, IterativeImputer, MissingIndicator |
| `kernel_approximation` | Nystroem, RBFSampler, SkewedChi2Sampler, AdditiveChi2Sampler |
| `random_projection` | Gaussian / Sparse random projection, `johnson_lindenstrauss_min_dim` |
| `hmm` | GaussianHMM, MultinomialHMM (forward-backward, Viterbi, Baum-Welch) |
| `discriminant` | LDA (classifier + transform), QDA |
| `feature_extraction` | CountVectorizer, TfidfVectorizer (sparse, matches sklearn) |
| `preprocessing` | Standard/MinMax/MaxAbs/Robust scalers, Normalizer, Label/OneHot/Ordinal encoders, PolynomialFeatures, KBinsDiscretizer, Binarizer |
| `model_selection` | train_test_split, KFold, StratifiedKFold, GroupKFold, StratifiedGroupKFold, TimeSeriesSplit, ShuffleSplit, LeaveOneOut/POut, Repeated*, PredefinedSplit; cross_val_score/predict; GridSearchCV, RandomizedSearchCV, HalvingGridSearchCV; learning_curve, validation_curve, permutation_importance; string scorers |
| `pipeline` | Pipeline, make_pipeline, FeatureUnion, ColumnTransformer |
| `metrics` | 40+ metrics: accuracy, precision/recall/F1, ROC-AUC, PR curve, average precision, log-loss, Brier, MCC, Cohen's kappa, NDCG, calibration curve, MSE/MAE/R²/pinball/deviances, silhouette, Calinski-Harabasz, Davies-Bouldin, NMI/homogeneity/completeness/V-measure |
| `datasets` | make_moons/blobs/circles/classification/regression/spiral; **bundled** iris, wine, breast_cancer, digits, diabetes, linnerud |

**`sample_weight`** is supported across linear models, trees, forests, boosting,
naive Bayes, SVC, KMeans and the metrics. **`partial_fit`** (out-of-core) is
available on SGD\*, the naive Bayes family, the scalers and MiniBatchKMeans;
**`warm_start`** on forests, gradient boosting and MLPs.

### Beyond the core

The library also reaches into the scikit-learn-*adjacent* ecosystem — the
families sklearn itself leaves to statsmodels, pyod, gensim, POT, networkx and
friends. Each is documented in [`docs/`](docs/index.md).

| Package | What it adds |
|---|---|
| `stats` | Statistical **inference**, not just point estimates: OLS/WLS/GLM/Logit/Probit/Poisson with a `.summary()` (SEs, t/z, p-values, CIs, R²/AIC/BIC, HC robust SEs), ANOVA, and diagnostics (Durbin-Watson, Ljung-Box, Breusch-Pagan, White, Jarque-Bera, ADF, KPSS, Granger) |
| `inference` | MCMC (Metropolis/Gibbs/HMC), Bayesian optimization, and **conformal prediction** — split/Mondrian intervals, conformalized quantile regression, Venn-Abers probability intervals, adaptive conformal for drifting streams |
| `pgm` | Discrete **Bayesian networks**: factor algebra, variable elimination, belief propagation, Chow-Liu / BIC-hill-climb structure learning, ancestral sampling |
| `anomaly` / `drift` | HBOS, ECOD, COPOD, KNN, CBLOF, ABOD outlier detectors; ADWIN, DDM, EDDM, Page-Hinkley, KSWIN **concept-drift** detectors |
| `topic` | LDA (collapsed Gibbs), LSA, BM25 ranking, Doc2Vec, UMass coherence |
| `multilabel` / `active` | Binary relevance → classifier chains → label powerset → RAkEL → MLkNN; uncertainty/margin/entropy/QBC/expected-model-change/core-set **active learning** |
| `optimal_transport` / `metric_learning` | Sinkhorn, Wasserstein, barycenters, OT domain adaptation, MMD/energy distance; ITML, LFDA, RCA |
| `graph` | PageRank, HITS, centralities, Louvain and label-propagation communities, DeepWalk/node2vec, Weisfeiler-Lehman kernel |
| `changepoint` / `copula` / `survival` | PELT, BinSeg, CUSUM, BOCPD; Gaussian/t/Clayton/Gumbel/Frank copulas; Kaplan-Meier, Cox, random survival forest, AFT, concordance |
| `encoders` | Target encoders that bound leak — WOE, James-Stein, M-estimate, leave-one-out — plus binary/count encoders, winsorizer, rare-label, cyclical |
| `image` | HOG, LBP, GLCM/Haralick, Gabor descriptors; integral image, connected components, morphology, Hough transform |
| `imbalance` | SMOTE/ADASYN/Borderline/Tomek/NearMiss resampling **and** balanced ensembles (BalancedRandomForest, RUSBoost, EasyEnsemble) |
| `recommend` | Matrix factorization, ALS, BPR, factorization machines, SVD++, and neighborhood CF (user/item KNN, SLIM) |
| `timeseries` / `sequence` | Kalman/EKF/UKF/particle filters, ARIMA, Holt-Winters, STL; DTW and edit distances |
| `causal` / `explain` | IPW/doubly-robust effect estimation; SHAP, LIME, PDP, integrated gradients |
| `embed` / `gnn` | word2vec, GloVe, BPE, WordPiece; GCN, GraphSAGE, GAT |
| `rl` / `evolutionary` | Bandits and tabular RL; CMA-ES and evolution strategies |
| `search` / `streaming` / `patterns` | LSH/IVF/PQ + Bloom/CMS/HLL/t-digest; FTRL/Hedge/Hoeffding; Apriori/FP-Growth/ECLAT |

`model_selection` also carries the AutoML-lite searchers **Hyperband** and
**TPE** alongside grid/random/halving search.

## Quick start

```python
from nupyml.datasets import load_breast_cancer
from nupyml.model_selection import train_test_split, RandomizedSearchCV
from nupyml.pipeline import make_pipeline
from nupyml.preprocessing import StandardScaler
from nupyml.svm import SVC

X, y = load_breast_cancer(return_X_y=True)
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, stratify=y, random_state=0)

pipe = make_pipeline(StandardScaler(), SVC())
search = RandomizedSearchCV(pipe, {"svc__C": [0.1, 1, 10]}, n_iter=3, cv=5,
                            scoring="roc_auc", random_state=0).fit(Xtr, ytr)
print(search.best_params_, search.score(Xte, yte))
```

```python
import numpy as np
from nupyml import nn
from nupyml.autograd import Tensor

model = nn.Sequential(
    nn.Conv2d(1, 8, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
    nn.Flatten(), nn.Linear(8 * 14 * 14, 10),
)
opt = nn.Adam(model.parameters(), lr=1e-3)
loss_fn = nn.CrossEntropyLoss()
for xb, yb in nn.DataLoader(X_images, y_labels, batch_size=64):
    opt.zero_grad()
    loss_fn(model(Tensor(xb)), yb).backward()
    opt.step()
```

See `examples/` for runnable scripts, including a character-level transformer.

## Optional pandas integration

pandas is never required. If it is installed, transformers record
`feature_names_in_`, validate column names at `transform` time, and can emit
dataframes:

```python
StandardScaler().set_output(transform="pandas").fit_transform(df)
```

## Checking your own estimators

`check_estimator` runs the API contract against any estimator — the same suite
all 80 built-in estimators pass:

```python
from nupyml.utils.estimator_checks import check_estimator
check_estimator(MyEstimator())
```

## Testing

scikit-learn is a **dev-only** dependency used to cross-validate results.
Many estimators match it numerically (OLS, Ridge, Lasso, PCA, tf-idf, Gaussian
and Multinomial NB, CategoricalNB, isotonic regression, GLMs, cost-complexity
pruning, Nyström, GP kernels); the rest are compared on held-out accuracy.

```bash
pip install -e ".[dev]"
pytest test/
```

## Benchmarks

`bench/` is a small benchmark suite with a strict rule: solutions may use **only
nupyml** (plus numpy/scipy/stdlib). Each task is a folder with a goal, a dataset,
and a metric; every submission is a single script exposing a `solve(...)`
function. We ship baselines and an auto-ranked scoreboard, and the community can
add entries.

```bash
python bench/harness.py                    # run every task, refresh the boards
python bench/harness.py digits_classification   # a single task
```

Submissions run in isolated subprocesses with a timeout and a static import
check, so a crashing or rule-breaking entry cannot affect the run. The suite
also **doubles as end-to-end QA** — `test/test_bench.py` fails the build if any
baseline drops below its task's floor. See [`bench/README.md`](bench/README.md)
and [`bench/CONTRIBUTING.md`](bench/CONTRIBUTING.md).

## Performance

Being readable does not mean being slow. Fit time against scikit-learn on
5000x20 (scikit-learn is Cython and BLAS underneath, so it is the honest bar):

| estimator | nupyml | scikit-learn | ratio |
|---|---|---|---|
| Ridge | 0.001s | 0.002s | **0.4x** |
| GaussianNB | 0.003s | 0.003s | **0.8x** |
| PCA(10) | 0.001s | 0.001s | **0.8x** |
| LogisticRegression | 0.014s | 0.012s | 1.2x |
| DecisionTree | 0.257s | 0.078s | 3.3x |
| HistGradientBoosting(100) | 0.826s | 0.226s | 3.6x |
| KMeans(8) | 0.158s | 0.037s | 4.3x |
| SVC (rbf, 800 rows) | 0.017s | 0.003s | 4.8x |
| RandomForest(50) | 3.21s | 0.507s | 6.3x |

Nothing is asymptotically wrong, and a few things are faster. The optimizations
that get it there are documented where they live, because understanding *why*
they work is the point:

- **`nupyml/ensemble/_hist_gb.py`** — histogram trees. Why `np.bincount` beats
  `np.add.at` by ~7x, and the histogram subtraction trick (`parent - child ==
  sibling`) that halves the work at every level of the tree.
- **`nupyml/svm/__init__.py`** — SMO. Why the working set is two variables,
  what makes a "violating pair", why second-order selection beats picking a
  partner at random, and how the incremental gradient keeps each step O(n).
- **`nupyml/decomposition/__init__.py`** — PCA. Three routes to the same
  eigenvectors (`full`, `covariance_eigh`, `randomized`), what each costs, and
  the precision that `covariance_eigh` trades away by squaring the condition
  number.
- **`nupyml/cluster/__init__.py`** — k-means. Why the M-step is a scatter-add
  rather than a per-cluster mask, and what D^2 seeding buys.

## Limitations

Pure numpy/scipy means CPU-only and no Cython or hand-written kernels, so
scikit-learn keeps a constant-factor edge on the heaviest estimators. There is
no `n_jobs` — everything is single-threaded. ONNX export and other
external-ecosystem integrations are out of scope. The algorithms themselves are
complete implementations, not toys.
