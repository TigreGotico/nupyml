import numpy as np
import pytest

from nupyml.neighbors import (KNeighborsClassifier, KNeighborsRegressor,
                              NearestNeighbors, KernelDensity)
from nupyml.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB, ComplementNB
from nupyml.cluster import (KMeans, MiniBatchKMeans, DBSCAN,
                            AgglomerativeClustering, MeanShift, SpectralClustering)
from nupyml.decomposition import PCA, TruncatedSVD, NMF, FastICA, KernelPCA
from nupyml.mixture import GaussianMixture
from nupyml.metrics import adjusted_rand_score
from nupyml.datasets import make_blobs, make_moons, make_circles, make_regression
from nupyml.model_selection import train_test_split


def test_knn_classifier():
    X, y = make_blobs(n_samples=300, centers=3, cluster_std=1.0, random_state=0)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=0)
    clf = KNeighborsClassifier(n_neighbors=5).fit(Xtr, ytr)
    import sklearn.neighbors as sknn
    ref = sknn.KNeighborsClassifier(n_neighbors=5).fit(Xtr, ytr)
    assert clf.score(Xte, yte) >= ref.score(Xte, yte) - 1e-12
    proba = clf.predict_proba(Xte)
    assert np.allclose(proba.sum(axis=1), 1)
    wclf = KNeighborsClassifier(n_neighbors=5, weights="distance").fit(Xtr, ytr)
    assert wclf.score(Xtr, ytr) == 1.0  # exact-match weighting memorizes train


def test_knn_regressor():
    X, y = make_regression(n_samples=300, n_features=3, noise=0.5, random_state=0)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=0)
    reg = KNeighborsRegressor(n_neighbors=7).fit(Xtr, ytr)
    assert reg.score(Xte, yte) > 0.5


def test_nearest_neighbors():
    X = np.array([[0.0, 0], [1, 0], [0, 1], [10, 10]])
    nn = NearestNeighbors(n_neighbors=2).fit(X)
    dist, idx = nn.kneighbors([[0.1, 0.1]])
    assert idx[0, 0] == 0
    rn = nn.radius_neighbors([[0.0, 0.0]], radius=1.5)
    assert set(rn[0]) == {0, 1, 2}


def test_kernel_density():
    rng = np.random.RandomState(0)
    X = rng.normal(0, 1, size=(500, 1))
    kde = KernelDensity(bandwidth=0.4).fit(X)
    logp = kde.score_samples(np.array([[0.0], [5.0]]))
    assert logp[0] > logp[1]
    # near true density at 0: 1/sqrt(2pi) ~ 0.399
    assert abs(np.exp(logp[0]) - 0.399) < 0.08
    samples = kde.sample(200, random_state=0)
    assert abs(samples.mean()) < 0.3


def test_gaussian_nb_matches_sklearn():
    import sklearn.naive_bayes as sknb
    X, y = make_blobs(n_samples=300, centers=3, cluster_std=2.0, random_state=1)
    ours = GaussianNB().fit(X, y)
    ref = sknb.GaussianNB().fit(X, y)
    assert np.allclose(ours.theta_, ref.theta_)
    assert ours.score(X, y) == pytest.approx(ref.score(X, y), abs=0.01)
    assert np.allclose(ours.predict_proba(X), ref.predict_proba(X), atol=1e-6)


def test_multinomial_nb_matches_sklearn():
    import sklearn.naive_bayes as sknb
    rng = np.random.RandomState(0)
    X = rng.poisson(2, size=(200, 30)).astype(float)
    y = (X[:, :5].sum(axis=1) > X[:, 5:10].sum(axis=1)).astype(int)
    ours = MultinomialNB().fit(X, y)
    ref = sknb.MultinomialNB().fit(X, y)
    assert np.allclose(ours.feature_log_prob_, ref.feature_log_prob_)
    assert np.array_equal(ours.predict(X), ref.predict(X))


def test_bernoulli_and_complement_nb():
    import sklearn.naive_bayes as sknb
    rng = np.random.RandomState(1)
    X = (rng.uniform(size=(200, 20)) > 0.6).astype(float)
    y = (X[:, :3].sum(axis=1) > 1).astype(int)
    ours = BernoulliNB().fit(X, y)
    ref = sknb.BernoulliNB().fit(X, y)
    assert np.array_equal(ours.predict(X), ref.predict(X))
    oursC = ComplementNB().fit(X + 0.0, y)
    refC = sknb.ComplementNB().fit(X, y)
    assert (oursC.predict(X) == refC.predict(X)).mean() > 0.95


def test_kmeans_recovers_blobs():
    X, y = make_blobs(n_samples=300, centers=4, cluster_std=0.6, random_state=2)
    km = KMeans(n_clusters=4, random_state=0).fit(X)
    assert adjusted_rand_score(y, km.labels_) > 0.95
    assert km.inertia_ > 0
    pred = km.predict(X)
    assert np.array_equal(pred, km.labels_)
    assert km.transform(X).shape == (300, 4)


def test_minibatch_kmeans():
    X, y = make_blobs(n_samples=500, centers=3, cluster_std=0.5, random_state=3)
    km = MiniBatchKMeans(n_clusters=3, random_state=0).fit(X)
    assert adjusted_rand_score(y, km.labels_) > 0.9


def test_dbscan_moons():
    X, y = make_moons(300, noise=0.05, random_state=0)
    db = DBSCAN(eps=0.2, min_samples=5).fit(X)
    labels = db.labels_
    core_mask = labels != -1
    assert adjusted_rand_score(y[core_mask], labels[core_mask]) > 0.95


def test_dbscan_noise_detection():
    X, _ = make_blobs(n_samples=100, centers=1, cluster_std=0.3, random_state=0)
    X = np.vstack([X, [[50, 50]]])
    db = DBSCAN(eps=1.0, min_samples=5).fit(X)
    assert db.labels_[-1] == -1


def test_agglomerative():
    X, y = make_blobs(n_samples=150, centers=3, cluster_std=0.5, random_state=4)
    ag = AgglomerativeClustering(n_clusters=3).fit(X)
    assert adjusted_rand_score(y, ag.labels_) > 0.95


def test_meanshift():
    X, y = make_blobs(n_samples=200, centers=3, cluster_std=0.4,
                      center_box=(-8, 8), random_state=8)
    ms = MeanShift(bandwidth=2.0).fit(X)
    assert adjusted_rand_score(y, ms.labels_) > 0.8


def test_spectral_circles():
    X, y = make_circles(200, noise=0.05, factor=0.4, random_state=0)
    sc = SpectralClustering(n_clusters=2, affinity="nearest_neighbors",
                            n_neighbors=10, random_state=0).fit(X)
    assert adjusted_rand_score(y, sc.labels_) > 0.9
    # kmeans cannot separate concentric circles
    km = KMeans(n_clusters=2, random_state=0).fit(X)
    assert adjusted_rand_score(y, km.labels_) < 0.5


def test_pca_matches_sklearn():
    import sklearn.decomposition as skd
    rng = np.random.RandomState(0)
    X = rng.normal(size=(100, 6)) @ rng.normal(size=(6, 6))
    ours = PCA(n_components=3).fit(X)
    ref = skd.PCA(n_components=3).fit(X)
    # components equal up to sign
    for i in range(3):
        assert (np.allclose(ours.components_[i], ref.components_[i], atol=1e-8)
                or np.allclose(ours.components_[i], -ref.components_[i], atol=1e-8))
    assert np.allclose(ours.explained_variance_ratio_,
                       ref.explained_variance_ratio_)
    Xt = ours.transform(X)
    assert np.allclose(ours.transform(ours.inverse_transform(Xt)), Xt, atol=1e-8)


def test_pca_variance_fraction():
    rng = np.random.RandomState(1)
    X = np.column_stack([rng.normal(scale=10, size=200),
                         rng.normal(scale=1, size=200),
                         rng.normal(scale=0.1, size=200)])
    p = PCA(n_components=0.95).fit(X)
    assert p.n_components_ <= 2


def test_truncated_svd():
    rng = np.random.RandomState(2)
    X = rng.normal(size=(50, 20))
    Xt = TruncatedSVD(n_components=5).fit_transform(X)
    assert Xt.shape == (50, 5)
    import scipy.sparse as sp
    Xs = sp.csr_matrix(np.abs(X))
    Xt2 = TruncatedSVD(n_components=5).fit_transform(Xs)
    assert Xt2.shape == (50, 5)


def test_nmf_reconstruction():
    rng = np.random.RandomState(3)
    W0 = np.abs(rng.normal(size=(60, 4)))
    H0 = np.abs(rng.normal(size=(4, 30)))
    X = W0 @ H0
    nmf = NMF(n_components=4, max_iter=2000, random_state=0)
    W = nmf.fit_transform(X)
    rel_err = nmf.reconstruction_err_ / np.linalg.norm(X)
    assert rel_err < 0.05
    assert (W >= 0).all() and (nmf.components_ >= 0).all()


def test_fastica_unmixes_sources():
    rng = np.random.RandomState(4)
    t = np.linspace(0, 8, 1000)
    s1 = np.sin(2 * t)
    s2 = np.sign(np.sin(3 * t))
    S = np.column_stack([s1, s2])
    A = np.array([[1.0, 0.5], [0.5, 1.0]])
    X = S @ A.T
    ica = FastICA(n_components=2, random_state=0)
    S_hat = ica.fit_transform(X)
    # each recovered component should correlate strongly with one source
    corr = np.abs(np.corrcoef(S.T, S_hat.T)[:2, 2:])
    assert corr.max(axis=1).min() > 0.95


def test_kernel_pca_separates_circles():
    X, y = make_circles(200, noise=0.05, factor=0.3, random_state=1)
    Xt = KernelPCA(n_components=2, kernel="rbf", gamma=4.0).fit(X).transform(X)
    # 1D threshold on first kPCA component should separate classes well
    from nupyml.linear_model import LogisticRegression
    acc = LogisticRegression().fit(Xt, y).score(Xt, y)
    assert acc > 0.95


def test_gmm_recovers_mixture():
    rng = np.random.RandomState(5)
    X = np.vstack([rng.normal([-3, 0], 0.5, size=(150, 2)),
                   rng.normal([3, 3], 1.0, size=(150, 2))])
    y = np.r_[np.zeros(150), np.ones(150)]
    gm = GaussianMixture(n_components=2, random_state=0).fit(X)
    assert adjusted_rand_score(y, gm.predict(X)) > 0.95
    assert np.allclose(gm.predict_proba(X).sum(axis=1), 1)
    means = gm.means_[np.argsort(gm.means_[:, 0])]
    assert np.allclose(means[0], [-3, 0], atol=0.3)
    assert np.allclose(means[1], [3, 3], atol=0.4)


@pytest.mark.parametrize("cov_type", ["full", "diag", "spherical"])
def test_gmm_covariance_types(cov_type):
    X, y = make_blobs(n_samples=300, centers=3, cluster_std=0.7, random_state=6)
    gm = GaussianMixture(n_components=3, covariance_type=cov_type,
                         random_state=0).fit(X)
    assert adjusted_rand_score(y, gm.predict(X)) > 0.9


def test_gmm_sample_and_bic():
    X, _ = make_blobs(n_samples=200, centers=2, cluster_std=0.5, random_state=7)
    gm = GaussianMixture(n_components=2, random_state=0).fit(X)
    samples, comps = gm.sample(100, random_state=0)
    assert samples.shape == (100, 2)
    gm1 = GaussianMixture(n_components=1, random_state=0).fit(X)
    assert gm.bic(X) < gm1.bic(X)  # 2 components fit better
