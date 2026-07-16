import numpy as np
import pytest

from nupyml.svm import SVC, SVR, LinearSVC
from nupyml.hmm import GaussianHMM, MultinomialHMM
from nupyml.manifold import TSNE, Isomap, MDS, LocallyLinearEmbedding
from nupyml.gaussian_process import (GaussianProcessRegressor,
                                     GaussianProcessClassifier, RBF, Matern,
                                     ConstantTimes)
from nupyml.discriminant import (LinearDiscriminantAnalysis,
                                 QuadraticDiscriminantAnalysis)
from nupyml.datasets import make_moons, make_blobs, make_circles
from nupyml.model_selection import train_test_split
from nupyml.metrics import adjusted_rand_score


# ---------------------------------------------------------------------------
# SVM
# ---------------------------------------------------------------------------

def test_svc_rbf_moons():
    X, y = make_moons(200, noise=0.15, random_state=0)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=0)
    clf = SVC(kernel="rbf", C=1.0, random_state=0).fit(Xtr, ytr)
    assert clf.score(Xte, yte) > 0.9
    assert len(clf.support_vectors_) > 0


def test_svc_linear_separable():
    X, y = make_blobs(n_samples=100, centers=2, cluster_std=0.5, random_state=0)
    clf = SVC(kernel="linear", random_state=0).fit(X, y)
    assert clf.score(X, y) == 1.0


def test_svc_multiclass_ovo():
    X, y = make_blobs(n_samples=200, centers=3, cluster_std=0.8, random_state=1)
    clf = SVC(kernel="rbf", random_state=0).fit(X, y)
    assert clf.score(X, y) > 0.95


def test_svc_string_labels():
    X, y = make_moons(120, noise=0.1, random_state=1)
    labels = np.array(["a", "b"])[y]
    clf = SVC(random_state=0).fit(X, labels)
    assert set(clf.predict(X)) <= {"a", "b"}


def test_svr_sine():
    rng = np.random.RandomState(0)
    X = np.sort(rng.uniform(0, 6, size=(120, 1)), axis=0)
    y = np.sin(X).ravel() + rng.normal(scale=0.05, size=120)
    reg = SVR(C=10.0, epsilon=0.05).fit(X, y)
    assert reg.score(X, y) > 0.95


def test_linear_svc():
    X, y = make_blobs(n_samples=200, centers=2, cluster_std=1.0, random_state=2)
    clf = LinearSVC(C=1.0, random_state=0).fit(X, y)
    assert clf.score(X, y) > 0.95
    X3, y3 = make_blobs(n_samples=240, centers=3, cluster_std=0.8, random_state=3)
    clf3 = LinearSVC(random_state=0).fit(X3, y3)
    assert clf3.score(X3, y3) > 0.9


# ---------------------------------------------------------------------------
# HMM
# ---------------------------------------------------------------------------

def test_gaussian_hmm_recovers_states():
    rng = np.random.RandomState(0)
    # two well-separated emission regimes with sticky transitions
    T = 600
    states = np.zeros(T, dtype=int)
    for t in range(1, T):
        states[t] = states[t - 1] if rng.uniform() < 0.95 else 1 - states[t - 1]
    means = np.array([[-3.0], [3.0]])
    X = means[states] + rng.normal(scale=0.7, size=(T, 1))
    hmm = GaussianHMM(n_components=2, random_state=0).fit(X)
    decoded = hmm.predict(X)
    # label permutation invariant comparison
    acc = max((decoded == states).mean(), (decoded == 1 - states).mean())
    assert acc > 0.95
    assert np.allclose(hmm.transmat_.sum(axis=1), 1)
    # transitions should be sticky
    assert hmm.transmat_.diagonal().min() > 0.85
    ll = hmm.score(X)
    assert np.isfinite(ll)
    proba = hmm.predict_proba(X)
    assert np.allclose(proba.sum(axis=1), 1)


def test_gaussian_hmm_sample_roundtrip():
    rng = np.random.RandomState(1)
    hmm = GaussianHMM(n_components=2, random_state=0)
    hmm.startprob_ = np.array([1.0, 0.0])
    hmm.transmat_ = np.array([[0.9, 0.1], [0.1, 0.9]])
    hmm.means_ = np.array([[-5.0], [5.0]])
    hmm.covars_ = np.array([[0.5], [0.5]])
    X, states = hmm.sample(300, random_state=2)
    hmm2 = GaussianHMM(n_components=2, random_state=0).fit(X)
    decoded = hmm2.predict(X)
    acc = max((decoded == states).mean(), (decoded == 1 - states).mean())
    assert acc > 0.95


def test_multinomial_hmm():
    rng = np.random.RandomState(2)
    T = 800
    states = np.zeros(T, dtype=int)
    for t in range(1, T):
        states[t] = states[t - 1] if rng.uniform() < 0.9 else 1 - states[t - 1]
    emis = np.array([[0.8, 0.15, 0.05], [0.05, 0.15, 0.8]])
    X = np.array([rng.choice(3, p=emis[s]) for s in states])
    hmm = MultinomialHMM(n_components=2, random_state=0).fit(X)
    decoded = hmm.predict(X)
    acc = max((decoded == states).mean(), (decoded == 1 - states).mean())
    assert acc > 0.85


def test_hmm_multiple_sequences():
    rng = np.random.RandomState(3)
    X = np.vstack([rng.normal(-2, 0.5, size=(100, 1)),
                   rng.normal(2, 0.5, size=(100, 1))])
    hmm = GaussianHMM(n_components=2, random_state=0).fit(X, lengths=[100, 100])
    means = np.sort(hmm.means_.ravel())
    assert np.allclose(means, [-2, 2], atol=0.3)


# ---------------------------------------------------------------------------
# manifold
# ---------------------------------------------------------------------------

def test_tsne_separates_blobs():
    X, y = make_blobs(n_samples=120, centers=3, cluster_std=0.5, random_state=4)
    emb = TSNE(perplexity=15, max_iter=400, random_state=0).fit_transform(X)
    assert emb.shape == (120, 2)
    from nupyml.cluster import KMeans
    labels = KMeans(n_clusters=3, random_state=0).fit(emb).labels_
    assert adjusted_rand_score(y, labels) > 0.9


def test_isomap_swiss_roll_unrolls():
    rng = np.random.RandomState(5)
    t = 1.5 * np.pi * (1 + 2 * rng.uniform(size=300))
    X = np.column_stack([t * np.cos(t), 10 * rng.uniform(size=300),
                         t * np.sin(t)])
    emb = Isomap(n_components=2, n_neighbors=8).fit_transform(X)
    # first isomap coordinate should track the roll parameter t
    corr = abs(np.corrcoef(emb[:, 0], t)[0, 1])
    assert corr > 0.9


def test_mds_preserves_distances():
    rng = np.random.RandomState(6)
    X = rng.normal(size=(60, 5))
    for metric in ["classical", "smacof"]:
        emb = MDS(n_components=5, metric=metric, random_state=0).fit_transform(X)
        from scipy.spatial.distance import pdist
        corr = np.corrcoef(pdist(X), pdist(emb))[0, 1]
        assert corr > 0.95, metric


def test_lle_runs_and_embeds():
    rng = np.random.RandomState(7)
    t = np.linspace(0, 4 * np.pi, 200)
    X = np.column_stack([np.cos(t), np.sin(t), t / 5]) \
        + rng.normal(scale=0.01, size=(200, 3))
    emb = LocallyLinearEmbedding(n_components=2, n_neighbors=10).fit_transform(X)
    assert emb.shape == (200, 2)
    assert np.all(np.isfinite(emb))


# ---------------------------------------------------------------------------
# gaussian process
# ---------------------------------------------------------------------------

def test_gpr_interpolates_sine():
    rng = np.random.RandomState(8)
    X = np.sort(rng.uniform(0, 6, size=(40, 1)), axis=0)
    y = np.sin(X).ravel()
    gp = GaussianProcessRegressor(alpha=1e-8, random_state=0).fit(X, y)
    Xq = np.linspace(0.5, 5.5, 50)[:, None]
    mean, std = gp.predict(Xq, return_std=True)
    assert np.max(np.abs(mean - np.sin(Xq).ravel())) < 0.05
    # uncertainty grows away from data
    m2, s2 = gp.predict(np.array([[20.0]]), return_std=True)
    assert s2[0] > std.mean()


def test_gpr_matern():
    rng = np.random.RandomState(9)
    X = np.sort(rng.uniform(0, 5, size=(50, 1)), axis=0)
    y = np.sin(2 * X).ravel() + rng.normal(scale=0.05, size=50)
    gp = GaussianProcessRegressor(
        kernel=ConstantTimes(Matern(1.0, nu=2.5), 1.0), alpha=1e-3,
        random_state=0).fit(X, y)
    assert gp.score(X, y) > 0.95


def test_gpc_moons():
    X, y = make_moons(150, noise=0.15, random_state=10)
    clf = GaussianProcessClassifier().fit(X, y)
    assert clf.score(X, y) > 0.9
    proba = clf.predict_proba(X)
    assert np.allclose(proba.sum(axis=1), 1)
    assert (proba > 0).all() and (proba < 1).all()


# ---------------------------------------------------------------------------
# discriminant analysis
# ---------------------------------------------------------------------------

def test_lda_matches_sklearn():
    import sklearn.discriminant_analysis as skda
    X, y = make_blobs(n_samples=300, centers=3, cluster_std=1.5, random_state=11)
    ours = LinearDiscriminantAnalysis().fit(X, y)
    ref = skda.LinearDiscriminantAnalysis().fit(X, y)
    assert (ours.predict(X) == ref.predict(X)).mean() > 0.98
    Xt = ours.transform(X)
    assert Xt.shape == (300, 2)


def test_qda_beats_lda_on_curved_boundary():
    X, y = make_circles(300, noise=0.1, factor=0.4, random_state=12)
    lda_acc = LinearDiscriminantAnalysis().fit(X, y).score(X, y)
    qda_acc = QuadraticDiscriminantAnalysis().fit(X, y).score(X, y)
    assert qda_acc > lda_acc
    assert qda_acc > 0.9
