# nupyml

A scikit-learn-like machine learning **and** neural network library whose only
runtime dependencies are **numpy** and **scipy**.

Everything is implemented from scratch: a tape-based reverse-mode autograd
engine powers the neural network stack, and the classic estimators follow the
familiar `fit` / `predict` / `transform` API with `get_params` / `set_params` /
`clone`, so `Pipeline` and `GridSearchCV` work with every estimator.

```bash
pip install numpy scipy
pip install -e .
```

## What's inside

### Neural networks (`nupyml.nn`, `nupyml.autograd`)

- **Autograd**: `Tensor` with topological-sort backprop; every op is
  gradient-checked against finite differences in the test suite.
- **Layers**: `Linear`, `Conv2d` (im2col), `MaxPool2d`/`AvgPool2d`,
  `BatchNorm1d/2d`, `LayerNorm`, `Dropout`, `Embedding`,
  `RNN`/`GRU`/`LSTM`, `MultiHeadAttention`, `TransformerEncoderLayer`,
  `PositionalEncoding`.
- **Losses**: MSE, MAE, Huber, cross-entropy (fused log-softmax), BCE,
  BCE-with-logits.
- **Optimizers**: SGD (+momentum/Nesterov), Adam, AdamW, RMSprop; StepLR /
  cosine / warmup schedulers; gradient clipping.
- **sklearn-style wrappers**: `MLPClassifier`, `MLPRegressor`.
- Model persistence via `model.save("weights.npz")` / `model.load(...)`.

### Classic ML

| Package | Estimators |
|---|---|
| `linear_model` | LinearRegression, Ridge, Lasso, ElasticNet (coordinate descent), LogisticRegression (L-BFGS), Perceptron, SGDClassifier/Regressor |
| `tree` | DecisionTreeClassifier/Regressor (vectorized CART) |
| `ensemble` | RandomForest, ExtraTrees, Bagging, AdaBoost (SAMME), GradientBoosting, **HistGradientBoosting** (binned, second-order), Voting |
| `svm` | SVC (SMO, one-vs-one), SVR (dual L-BFGS-B), LinearSVC (dual coordinate descent) |
| `neighbors` | KNeighborsClassifier/Regressor, NearestNeighbors, KernelDensity (cKDTree) |
| `naive_bayes` | Gaussian, Multinomial, Bernoulli, Complement |
| `cluster` | KMeans (k-means++), MiniBatchKMeans, DBSCAN, Agglomerative, MeanShift, SpectralClustering |
| `mixture` | GaussianMixture (log-space EM; full/diag/spherical; BIC/AIC, sampling) |
| `decomposition` | PCA, TruncatedSVD, NMF, FastICA, KernelPCA |
| `manifold` | TSNE (exact), Isomap, MDS (classical + SMACOF), LocallyLinearEmbedding |
| `gaussian_process` | GPR (Cholesky, marginal-likelihood optimization), GPC (Laplace), RBF/Matern kernels |
| `hmm` | GaussianHMM, MultinomialHMM (forward-backward, Viterbi, Baum-Welch) |
| `discriminant` | LDA (classifier + transform), QDA |
| `feature_extraction` | CountVectorizer, TfidfVectorizer (sparse, matches sklearn) |
| `preprocessing` | Standard/MinMax/MaxAbs/Robust scalers, Normalizer, Label/OneHot/Ordinal encoders, PolynomialFeatures, KBinsDiscretizer |
| `model_selection` | train_test_split (stratified), KFold, StratifiedKFold, LeaveOneOut, cross_val_score/predict, GridSearchCV |
| `pipeline` | Pipeline, make_pipeline, FeatureUnion, ColumnTransformer |
| `metrics` | accuracy, precision/recall/F1, ROC-AUC, log-loss, confusion matrix, classification report, MSE/MAE/R², silhouette, adjusted Rand |
| `datasets` | make_moons, make_blobs, make_circles, make_classification, make_regression, make_spiral |

## Quick start

```python
from nupyml.datasets import make_moons
from nupyml.model_selection import train_test_split, GridSearchCV
from nupyml.pipeline import make_pipeline
from nupyml.preprocessing import StandardScaler
from nupyml.svm import SVC

X, y = make_moons(500, noise=0.2, random_state=0)
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=0)

pipe = make_pipeline(StandardScaler(), SVC())
gs = GridSearchCV(pipe, {"svc__C": [0.1, 1, 10]}, cv=5).fit(Xtr, ytr)
print(gs.best_params_, gs.score(Xte, yte))
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

## Testing

scikit-learn is a **dev-only** dependency used to cross-validate results (OLS,
Ridge, Lasso, PCA, tf-idf and Gaussian NB match sklearn numerically; other
estimators are compared on held-out accuracy).

```bash
pip install -e ".[dev]"
pytest test/
```

## Limitations

Pure numpy/scipy means CPU-only and no Cython fast paths: large datasets
(hundreds of thousands of rows) or big deep networks will be slow relative to
sklearn/pytorch. The algorithms themselves are complete implementations, not
toys.
