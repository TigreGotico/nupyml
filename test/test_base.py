import numpy as np
import pytest

from nupyml.base import BaseEstimator, clone, check_is_fitted
from nupyml.preprocessing import (
    StandardScaler, MinMaxScaler, RobustScaler, Normalizer, LabelEncoder,
    LabelBinarizer, OneHotEncoder, PolynomialFeatures, KBinsDiscretizer,
)
from nupyml.metrics import (
    accuracy_score, f1_score, confusion_matrix, log_loss, roc_auc_score,
    mean_squared_error, r2_score, silhouette_score, adjusted_rand_score,
)
from nupyml.model_selection import (
    train_test_split, KFold, StratifiedKFold, cross_val_score, ParameterGrid,
)
from nupyml.datasets import make_blobs, make_moons, make_classification


class Dummy(BaseEstimator):
    def __init__(self, a=1, b="x"):
        self.a = a
        self.b = b


def test_get_set_params_and_clone():
    d = Dummy(a=5)
    assert d.get_params() == {"a": 5, "b": "x"}
    d.set_params(b="y")
    assert d.b == "y"
    c = clone(d)
    assert c.get_params() == d.get_params() and c is not d
    with pytest.raises(ValueError):
        d.set_params(nope=1)


def test_check_is_fitted():
    d = Dummy()
    with pytest.raises(RuntimeError):
        check_is_fitted(d)
    d.coef_ = 1
    check_is_fitted(d)


def test_standard_scaler_roundtrip():
    X = np.random.RandomState(0).normal(5, 3, size=(50, 4))
    s = StandardScaler()
    Xt = s.fit_transform(X)
    assert np.allclose(Xt.mean(axis=0), 0, atol=1e-12)
    assert np.allclose(Xt.std(axis=0), 1, atol=1e-12)
    assert np.allclose(s.inverse_transform(Xt), X)


def test_minmax_scaler():
    X = np.random.RandomState(0).uniform(-3, 7, size=(30, 3))
    Xt = MinMaxScaler().fit_transform(X)
    assert np.allclose(Xt.min(axis=0), 0) and np.allclose(Xt.max(axis=0), 1)


def test_robust_scaler_and_normalizer():
    X = np.random.RandomState(1).normal(size=(40, 3))
    Xt = RobustScaler().fit_transform(X)
    assert np.allclose(np.median(Xt, axis=0), 0, atol=1e-12)
    Xn = Normalizer().fit_transform(X)
    assert np.allclose(np.linalg.norm(Xn, axis=1), 1)


def test_label_encoder():
    le = LabelEncoder()
    y = np.array(["b", "a", "c", "a"])
    yt = le.fit_transform(y)
    assert list(le.classes_) == ["a", "b", "c"]
    assert np.array_equal(le.inverse_transform(yt), y)
    with pytest.raises(ValueError):
        le.transform(["z"])


def test_label_binarizer_multiclass_and_binary():
    lb = LabelBinarizer()
    Y = lb.fit_transform([0, 1, 2, 1])
    assert Y.shape == (4, 3) and np.array_equal(Y.sum(axis=1), np.ones(4))
    assert np.array_equal(lb.inverse_transform(Y), [0, 1, 2, 1])
    Y2 = LabelBinarizer().fit_transform([0, 1, 0])
    assert Y2.shape == (3, 1)


def test_onehot_encoder():
    X = np.array([["a", "x"], ["b", "y"], ["a", "y"]])
    Xt = OneHotEncoder().fit_transform(X)
    assert Xt.shape == (3, 4) and np.allclose(Xt.sum(axis=1), 2)


def test_polynomial_features():
    X = np.array([[2.0, 3.0]])
    Xt = PolynomialFeatures(degree=2).fit_transform(X)
    # 1, x1, x2, x1^2, x1x2, x2^2
    assert np.allclose(sorted(Xt[0]), sorted([1, 2, 3, 4, 6, 9]))


def test_kbins():
    X = np.arange(20, dtype=float)[:, None]
    Xt = KBinsDiscretizer(n_bins=4).fit_transform(X)
    assert set(np.unique(Xt)) == {0, 1, 2, 3}


def test_metrics_classification():
    y = [0, 1, 1, 0, 1]
    p = [0, 1, 0, 0, 1]
    assert accuracy_score(y, p) == 0.8
    assert np.array_equal(confusion_matrix(y, p), [[2, 0], [1, 2]])
    assert abs(f1_score(y, p) - 0.8) < 1e-12
    assert roc_auc_score([0, 0, 1, 1], [0.1, 0.4, 0.35, 0.8]) == 0.75
    assert log_loss([0, 1], [[0.9, 0.1], [0.1, 0.9]]) < 0.2


def test_metrics_regression():
    assert mean_squared_error([1, 2], [1, 4]) == 2.0
    assert r2_score([1, 2, 3], [1, 2, 3]) == 1.0
    assert r2_score([1, 2, 3], [2, 2, 2]) == 0.0


def test_clustering_metrics():
    X, y = make_blobs(n_samples=60, centers=3, cluster_std=0.3, random_state=0)
    assert silhouette_score(X, y) > 0.7
    assert adjusted_rand_score(y, y) == 1.0
    perm = (y + 1) % 3
    assert adjusted_rand_score(y, perm) == 1.0


def test_train_test_split_stratified():
    X, y = make_classification(n_samples=100, random_state=0)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y,
                                          random_state=0)
    assert len(Xte) in (20, 21)
    # class balance preserved
    assert abs(ytr.mean() - yte.mean()) < 0.1


def test_kfold_partitions():
    X = np.arange(23)
    seen = []
    for tr, te in KFold(n_splits=5, shuffle=True, random_state=0).split(X):
        assert len(set(tr) & set(te)) == 0
        seen.extend(te)
    assert sorted(seen) == list(range(23))


def test_stratified_kfold():
    y = np.array([0] * 10 + [1] * 10)
    for tr, te in StratifiedKFold(n_splits=5).split(np.zeros((20, 1)), y):
        assert (y[te] == 0).sum() == 2 and (y[te] == 1).sum() == 2


def test_parameter_grid():
    grid = ParameterGrid({"a": [1, 2], "b": ["x", "y", "z"]})
    assert len(grid) == 6 and len(list(grid)) == 6


def test_datasets_shapes():
    X, y = make_moons(200, noise=0.1, random_state=0)
    assert X.shape == (200, 2) and set(y) == {0, 1}
    X, y = make_blobs(90, centers=3, random_state=0)
    assert X.shape == (90, 2) and len(set(y)) == 3
