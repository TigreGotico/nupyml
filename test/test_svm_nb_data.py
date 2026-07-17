import numpy as np
import pytest
import scipy.sparse as sp

import sklearn.svm as sksvm
import sklearn.naive_bayes as sknb
import sklearn.datasets as skd

from nupyml.svm import SVC, SVR, NuSVC, NuSVR, LinearSVC
from nupyml.naive_bayes import CategoricalNB
from nupyml.cluster import KMeans
from nupyml.neighbors import KNeighborsClassifier, KNeighborsRegressor
from nupyml.datasets import (load_iris, load_wine, load_breast_cancer,
                             load_digits, load_diabetes, load_linnerud,
                             make_moons, make_regression, make_blobs,
                             make_classification)
from nupyml.metrics import adjusted_rand_score, brier_score_loss
from nupyml.model_selection import train_test_split

rng = np.random.RandomState(0)


# ---------------------------------------------------------------------------
# SVR: the 2n-variable dual produces genuinely sparse solutions
# ---------------------------------------------------------------------------

def test_svr_matches_sklearn():
    X, y = make_regression(n_samples=120, n_features=3, noise=5.0, random_state=0)
    ours = SVR(C=100).fit(X, y)
    ref = sksvm.SVR(C=100).fit(X, y)
    assert ours.score(X, y) == pytest.approx(ref.score(X, y), abs=0.01)


def test_svr_epsilon_controls_sparsity():
    """A wider tube must drop support vectors. The collapsed |beta| dual cannot
    show this: a quasi-Newton solver never drives beta to exact zero."""
    X, y = make_regression(n_samples=100, n_features=2, noise=5.0, random_state=1)
    scale = np.abs(y).mean()
    fracs = [len(SVR(C=100, epsilon=e * scale).fit(X, y).support_) / len(y)
             for e in (0.01, 0.5, 2.0)]
    assert fracs[0] > fracs[1] > fracs[2]
    assert fracs[2] < 0.5


def test_svr_exposes_support_attributes():
    X, y = make_regression(n_samples=80, n_features=2, noise=5.0, random_state=2)
    svr = SVR(C=10, epsilon=5.0).fit(X, y)
    assert svr.support_vectors_.shape == (len(svr.support_), 2)
    assert svr.dual_coef_.shape == (1, len(svr.support_))
    assert np.isfinite(svr.intercept_)


def test_svr_sine():
    r = np.random.RandomState(3)
    X = np.sort(r.uniform(0, 6, size=(120, 1)), axis=0)
    y = np.sin(X).ravel() + r.normal(scale=0.05, size=120)
    assert SVR(C=10, epsilon=0.05).fit(X, y).score(X, y) > 0.95


# ---------------------------------------------------------------------------
# NuSVC / NuSVR
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("nu", [0.3, 0.5])
def test_nusvc_matches_sklearn(nu):
    X, y = make_moons(200, noise=0.2, random_state=0)
    ours = NuSVC(nu=nu, random_state=0).fit(X, y)
    ref = sksvm.NuSVC(nu=nu).fit(X, y)
    assert ours.score(X, y) >= ref.score(X, y) - 0.02


@pytest.mark.parametrize("nu", [0.3, 0.5])
def test_nusvc_nu_lower_bounds_support_vector_fraction(nu):
    X, y = make_moons(200, noise=0.2, random_state=1)
    clf = NuSVC(nu=nu, random_state=0).fit(X, y)
    frac = len(clf.support_vectors_) / len(X)
    # that is what nu means: at least this fraction are support vectors
    assert frac >= nu * 0.9


def test_nusvc_multiclass_and_probability():
    X, y = make_blobs(n_samples=210, centers=3, cluster_std=1.0, random_state=2)
    clf = NuSVC(nu=0.3, probability=True, random_state=0).fit(X, y)
    assert clf.score(X, y) > 0.9
    proba = clf.predict_proba(X)
    assert proba.shape == (210, 3)
    assert np.allclose(proba.sum(axis=1), 1)


@pytest.mark.parametrize("nu", [0.3, 0.5, 0.8])
def test_nusvr_nu_lower_bounds_support_fraction(nu):
    """nu lower-bounds the support-vector fraction. The fraction is a step
    function of epsilon, so the search must land on the admissible side."""
    X, y = make_regression(n_samples=120, n_features=3, noise=5.0, random_state=3)
    reg = NuSVR(nu=nu, C=100).fit(X, y)
    assert reg.sv_fraction_ >= nu - 1e-9
    assert reg.sv_fraction_ < nu + 0.15      # and not wildly over
    assert reg.score(X, y) > 0.9


def test_nusvr_larger_nu_widens_the_support_set():
    X, y = make_regression(n_samples=120, n_features=3, noise=5.0, random_state=3)
    fracs = [NuSVR(nu=nu, C=100).fit(X, y).sv_fraction_ for nu in (0.3, 0.6, 0.9)]
    assert fracs[0] < fracs[1] < fracs[2]
    # a bigger nu means a narrower tube
    eps = [NuSVR(nu=nu, C=100).fit(X, y).epsilon_ for nu in (0.3, 0.9)]
    assert eps[0] > eps[1]


def test_nusvr_comparable_to_sklearn():
    X, y = make_regression(n_samples=120, n_features=3, noise=5.0, random_state=4)
    ours = NuSVR(nu=0.5, C=100).fit(X, y)
    ref = sksvm.NuSVR(nu=0.5, C=100).fit(X, y)
    assert ours.score(X, y) >= ref.score(X, y) - 0.05


# ---------------------------------------------------------------------------
# SVC probability
# ---------------------------------------------------------------------------

def test_svc_probability_is_calibrated():
    X, y = make_moons(400, noise=0.35, random_state=5)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.4, random_state=0)
    clf = SVC(probability=True, random_state=0).fit(Xtr, ytr)
    proba = clf.predict_proba(Xte)
    assert np.allclose(proba.sum(axis=1), 1)
    assert ((proba >= 0) & (proba <= 1)).all()
    # Platt-scaled margins beat a raw squashed decision value on Brier score
    from nupyml.utils import sigmoid
    raw = SVC(random_state=0).fit(Xtr, ytr)
    naive = sigmoid(raw.decision_function(Xte))
    assert brier_score_loss(yte, proba[:, 1]) <= brier_score_loss(yte, naive)


def test_svc_predict_proba_requires_probability_flag():
    X, y = make_moons(100, noise=0.2, random_state=6)
    clf = SVC(random_state=0).fit(X, y)
    with pytest.raises(AttributeError, match="probability=True"):
        clf.predict_proba(X)


def test_svc_predict_log_proba():
    X, y = make_moons(150, noise=0.2, random_state=7)
    clf = SVC(probability=True, random_state=0).fit(X, y)
    assert np.allclose(np.exp(clf.predict_log_proba(X)), clf.predict_proba(X))


# ---------------------------------------------------------------------------
# CategoricalNB
# ---------------------------------------------------------------------------

def test_categorical_nb_matches_sklearn():
    r = np.random.RandomState(8)
    X = r.randint(0, 4, size=(200, 3))
    y = (X[:, 0] > 1).astype(int)
    ours = CategoricalNB().fit(X, y)
    ref = sknb.CategoricalNB().fit(X, y)
    assert np.allclose(ours.predict_proba(X), ref.predict_proba(X))
    assert np.array_equal(ours.predict(X), ref.predict(X))


def test_categorical_nb_no_ordinal_assumption():
    """Category codes carry no order: a model that assumed one would fail."""
    r = np.random.RandomState(9)
    X = r.randint(0, 4, size=(400, 1))
    # class depends on category membership, not on the code's magnitude
    y = np.isin(X[:, 0], [0, 2]).astype(int)
    assert CategoricalNB().fit(X, y).score(X, y) > 0.95


def test_categorical_nb_alpha_smooths_unseen_categories():
    X = np.array([[0], [0], [1], [1]])
    y = np.array([0, 0, 1, 1])
    nb = CategoricalNB(alpha=1.0).fit(X, y)
    # every category keeps non-zero probability under both classes
    assert np.isfinite(nb.category_log_prob_[0]).all()
    assert (np.exp(nb.category_log_prob_[0]) > 0).all()


def test_categorical_nb_rejects_negative():
    X = np.array([[-1], [0]])
    with pytest.raises(ValueError, match="non-negative"):
        CategoricalNB().fit(X, np.array([0, 1]))


def test_categorical_nb_min_categories():
    X = np.array([[0], [1], [0], [1]])
    y = np.array([0, 1, 0, 1])
    nb = CategoricalNB(min_categories=5).fit(X, y)
    assert nb.n_categories_[0] == 5
    assert nb.category_log_prob_[0].shape[1] == 5


# ---------------------------------------------------------------------------
# sparse input
# ---------------------------------------------------------------------------

def test_kmeans_accepts_sparse():
    X, y = make_blobs(n_samples=200, centers=3, cluster_std=0.5, random_state=10)
    Xs = sp.csr_matrix(np.abs(X))
    dense = KMeans(n_clusters=3, random_state=0).fit(np.abs(X))
    sparse = KMeans(n_clusters=3, random_state=0).fit(Xs)
    assert adjusted_rand_score(dense.labels_, sparse.labels_) == 1.0
    assert sparse.predict(Xs).shape == (200,)
    assert sparse.transform(Xs).shape == (200, 3)


def test_knn_accepts_sparse():
    X, y = make_blobs(n_samples=200, centers=3, cluster_std=0.6, random_state=11)
    Xs = sp.csr_matrix(np.abs(X))
    dense = KNeighborsClassifier().fit(np.abs(X), y)
    sparse = KNeighborsClassifier().fit(Xs, y)
    assert np.array_equal(dense.predict(np.abs(X)), sparse.predict(Xs))
    reg = KNeighborsRegressor().fit(Xs, y.astype(float))
    assert reg.predict(Xs).shape == (200,)


# ---------------------------------------------------------------------------
# bundled datasets
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("ours,ref", [
    (load_iris, skd.load_iris),
    (load_wine, skd.load_wine),
    (load_breast_cancer, skd.load_breast_cancer),
    (load_digits, skd.load_digits),
    (load_diabetes, skd.load_diabetes),
])
def test_bundled_dataset_matches_sklearn(ours, ref):
    a, b = ours(), ref()
    assert np.allclose(a.data, b.data)
    assert np.array_equal(a.target, b.target)
    assert len(a.feature_names) == a.data.shape[1]


def test_load_returns_X_y():
    X, y = load_iris(return_X_y=True)
    assert X.shape == (150, 4) and y.shape == (150,)


def test_digits_images():
    d = load_digits()
    assert d.images.shape == (1797, 8, 8)
    assert np.allclose(d.images.reshape(1797, 64), d.data)


def test_bunch_attribute_and_key_access():
    d = load_wine()
    assert d.data is d["data"]
    assert len(d.target_names) == 3
    with pytest.raises(AttributeError):
        d.nonexistent


def test_bundled_data_is_usable_end_to_end():
    X, y = load_breast_cancer(return_X_y=True)
    from nupyml.pipeline import make_pipeline
    from nupyml.preprocessing import StandardScaler
    from nupyml.linear_model import LogisticRegression
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=0)
    pipe = make_pipeline(StandardScaler(), LogisticRegression()).fit(Xtr, ytr)
    assert pipe.score(Xte, yte) > 0.9


# ---------------------------------------------------------------------------
# SMO solver: second-order working-set selection
# ---------------------------------------------------------------------------

def test_smo_converges_in_few_iterations():
    """Second-order working-set selection reaches the KKT tolerance in a few
    hundred pair updates. Selecting the partner at random instead needs orders
    of magnitude more steps for the same problem."""
    X, y = make_classification(n_samples=800, n_features=20, n_informative=8,
                               random_state=0)
    clf = SVC(random_state=0).fit(X, y)
    assert clf.n_iter_ < 5000
    assert clf.score(X, y) >= sksvm.SVC().fit(X, y).score(X, y) - 0.01


def test_smo_is_deterministic_without_a_seed():
    """WSS3 picks the working set analytically, so no RNG is involved."""
    X, y = make_moons(300, noise=0.25, random_state=0)
    a = SVC().fit(X, y)
    b = SVC().fit(X, y)
    assert np.array_equal(a.predict(X), b.predict(X))
    assert np.allclose(a.decision_function(X), b.decision_function(X))


def test_smo_matches_sklearn_decision_function():
    """Same dual problem, same optimum: the margins should agree closely."""
    X, y = make_moons(300, noise=0.25, random_state=1)
    ours = SVC(kernel="rbf", C=1.0, gamma=0.5).fit(X, y)
    ref = sksvm.SVC(kernel="rbf", C=1.0, gamma=0.5).fit(X, y)
    assert np.corrcoef(ours.decision_function(X),
                       ref.decision_function(X))[0, 1] > 0.999
    assert (ours.predict(X) == ref.predict(X)).mean() > 0.99


def test_smo_respects_the_box_constraint():
    X, y = make_moons(200, noise=0.4, random_state=2)
    C = 0.5
    clf = SVC(C=C, random_state=0).fit(X, y)
    coef = clf._models[(0, 1)][1]      # alpha_i * y_i
    assert np.abs(coef).max() <= C + 1e-6


def test_smo_scales_to_larger_problems():
    """Each step costs O(n) rather than O(n^2), so doubling n stays cheap."""
    import time
    X, y = make_classification(n_samples=2000, n_features=20, n_informative=8,
                               random_state=3)
    start = time.perf_counter()
    clf = SVC(random_state=0).fit(X, y)
    elapsed = time.perf_counter() - start
    assert elapsed < 5.0
    assert clf.score(X, y) > 0.95
