# Getting started

## Install

```bash
pip install numpy scipy
pip install -e .            # from a checkout
```

For the test suite, install the dev extras (scikit-learn and pandas are
**test-only** — they are never imported at runtime):

```bash
pip install -e ".[dev]"
pytest test/
```

Python 3.9+ is supported.

## The two APIs

nupyml has two front doors that share the same style.

### 1. Estimators (the scikit-learn contract)

Every classic model is an estimator with `fit` / `predict` / `transform`,
parameters passed in `__init__`, and learned attributes ending in `_`:

```python
from nupyml.datasets import load_breast_cancer
from nupyml.model_selection import train_test_split, RandomizedSearchCV
from nupyml.pipeline import make_pipeline
from nupyml.preprocessing import StandardScaler
from nupyml.svm import SVC

X, y = load_breast_cancer(return_X_y=True)
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, stratify=y,
                                      random_state=0)

pipe = make_pipeline(StandardScaler(), SVC())
search = RandomizedSearchCV(pipe, {"svc__C": [0.1, 1, 10]}, n_iter=3, cv=5,
                            scoring="roc_auc", random_state=0).fit(Xtr, ytr)
print(search.best_params_, search.score(Xte, yte))
```

Because the contract is uniform, `Pipeline`, `FeatureUnion`,
`ColumnTransformer`, cross-validation, and every `*SearchCV` estimator work with
any model — including ones you write yourself, as long as they pass
`check_estimator` (see [Contributing](contributing.md)).

### 2. Autograd (the neural-network stack)

The neural side is a small PyTorch-shaped API over a tape-based reverse-mode
autograd engine:

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

If you prefer the estimator API for a net, `nn.MLPClassifier` /
`nn.MLPRegressor` wrap a multilayer perceptron behind `fit`/`predict`.

See [`examples/`](../examples) for runnable scripts, including a character-level
transformer.

## Optional pandas integration

pandas is never required. If it is installed, transformers record
`feature_names_in_`, validate column names at `transform` time, and can emit
dataframes:

```python
StandardScaler().set_output(transform="pandas").fit_transform(df)
```

## Where to go next

- Predicting from tabular data → [Classic ML](classic-ml.md)
- Building a network → [Neural networks](neural-networks.md)
- You need error bars, p-values, or guaranteed coverage →
  [Statistics & inference](statistics.md)
- Graphs, text, anomalies, recommenders, images →
  [Beyond scikit-learn](beyond-sklearn.md)
