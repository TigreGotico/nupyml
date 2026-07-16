# nupyml

A scikit-learn-like machine learning **and** neural network library whose only
runtime dependencies are **numpy** and **scipy**.

Everything is implemented from scratch: a tape-based reverse-mode autograd
engine powers the neural network stack, and the classic estimators follow the
familiar `fit` / `predict` / `transform` API with `get_params` / `set_params` /
`clone`, so `Pipeline` and the `*SearchCV` estimators work with every one of
them.

```bash
pip install numpy scipy
pip install -e .
```

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
all 78 built-in estimators pass:

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

## Limitations

Pure numpy/scipy means CPU-only and no Cython fast paths: large datasets
(hundreds of thousands of rows) or big deep networks will be slow relative to
sklearn/pytorch. There is no `n_jobs` — everything is single-threaded. ONNX
export and other external-ecosystem integrations are out of scope. The
algorithms themselves are complete implementations, not toys.
