# Classic ML

The scikit-learn-parity estimators. All follow the `fit`/`predict`/`transform`
contract and compose in pipelines and searches.

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
| `feature_selection` | SelectKBest/Percentile/Fpr, VarianceThreshold, RFE, RFECV, SelectFromModel, SequentialFeatureSelector; `f_classif`, `f_regression`, `chi2`, `mutual_info_*` (advanced selectors — Boruta, mRMR, ReliefF, stability selection — are in [Beyond scikit-learn](beyond-sklearn.md)) |
| `impute` | SimpleImputer, KNNImputer, IterativeImputer, MissingIndicator |
| `kernel_approximation` | Nystroem, RBFSampler, SkewedChi2Sampler, AdditiveChi2Sampler |
| `random_projection` | Gaussian / Sparse random projection, `johnson_lindenstrauss_min_dim` |
| `hmm` | GaussianHMM, MultinomialHMM (forward-backward, Viterbi, Baum-Welch) |
| `discriminant` | LDA (classifier + transform), QDA |
| `cross_decomposition` | PLS regression / canonical, CCA |
| `covariance` | Empirical, Ledoit-Wolf, OAS, graphical-lasso, MCD |
| `feature_extraction` | CountVectorizer, TfidfVectorizer (sparse, matches sklearn) |
| `preprocessing` | Standard/MinMax/MaxAbs/Robust scalers, Normalizer, Label/OneHot/Ordinal encoders, PolynomialFeatures, KBinsDiscretizer, Binarizer |
| `model_selection` | train_test_split, KFold, StratifiedKFold, GroupKFold, StratifiedGroupKFold, TimeSeriesSplit, ShuffleSplit, LeaveOneOut/POut, Repeated*, PredefinedSplit; cross_val_score/predict; GridSearchCV, RandomizedSearchCV, HalvingGridSearchCV, **HyperbandSearchCV**, **TPESearchCV**; learning_curve, validation_curve, permutation_importance; string scorers |
| `pipeline` | Pipeline, make_pipeline, FeatureUnion, ColumnTransformer |
| `metrics` | 40+ metrics: accuracy, precision/recall/F1, ROC-AUC, PR curve, average precision, log-loss, Brier, MCC, Cohen's kappa, NDCG, calibration curve, MSE/MAE/R²/pinball/deviances, silhouette, Calinski-Harabasz, Davies-Bouldin, NMI/homogeneity/completeness/V-measure |
| `datasets` | make_moons/blobs/circles/classification/regression/spiral; **bundled** iris, wine, breast_cancer, digits, diabetes, linnerud |

## Cross-cutting capabilities

- **`sample_weight`** — linear models, trees, forests, boosting, naive Bayes,
  SVC, KMeans, and the metrics.
- **`partial_fit`** (out-of-core) — SGD\*, the naive Bayes family, the scalers,
  and MiniBatchKMeans.
- **`warm_start`** — forests, gradient boosting, and MLPs.

## Numerical parity

Many estimators match scikit-learn numerically, not just in accuracy: OLS,
Ridge, Lasso, PCA, tf-idf, Gaussian / Multinomial / Categorical NB, isotonic
regression, GLMs, cost-complexity pruning, Nyström, and the GP kernels. The rest
are validated on held-out accuracy. See [Contributing](contributing.md) for how
that cross-validation is set up.
