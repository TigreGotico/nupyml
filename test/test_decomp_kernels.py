import numpy as np
import pytest

import sklearn.kernel_approximation as skka
import sklearn.random_projection as skrp
import sklearn.decomposition as skd
import sklearn.gaussian_process.kernels as skgpk

from nupyml.random_projection import (GaussianRandomProjection,
                                      SparseRandomProjection,
                                      johnson_lindenstrauss_min_dim)
from nupyml.kernel_approximation import (RBFSampler, Nystroem,
                                         SkewedChi2Sampler, AdditiveChi2Sampler)
from nupyml.decomposition import (SparsePCA, DictionaryLearning, SparseCoder,
                                  MiniBatchNMF, LatentDirichletAllocation, PCA)
from nupyml.gaussian_process import (GaussianProcessRegressor, RBF, Matern,
                                     WhiteKernel, ConstantKernel,
                                     RationalQuadratic, ExpSineSquared,
                                     DotProduct)
from nupyml.linear_model import Ridge, LogisticRegression
from nupyml.svm import SVC
from nupyml.pipeline import make_pipeline
from nupyml.datasets import (make_moons, make_classification, make_regression,
                             make_blobs)

rng = np.random.RandomState(0)


# ---------------------------------------------------------------------------
# random projection
# ---------------------------------------------------------------------------

def test_jl_min_dim_matches_sklearn():
    for n in (100, 10000):
        assert (johnson_lindenstrauss_min_dim(n, eps=0.1)
                == skrp.johnson_lindenstrauss_min_dim(n, eps=0.1))


def test_gaussian_random_projection_preserves_distances():
    from scipy.spatial.distance import pdist
    # clustered data: iid gaussian points have concentrated pairwise distances
    # (std/mean ~ 0.05), so there is no real distance structure to preserve
    X, _ = make_blobs(n_samples=100, n_features=200, centers=5, random_state=0)
    Xt = GaussianRandomProjection(n_components=150, random_state=0).fit_transform(X)
    assert Xt.shape == (100, 150)
    d_orig, d_proj = pdist(X), pdist(Xt)
    assert np.corrcoef(d_orig, d_proj)[0, 1] > 0.95
    assert abs(np.median(d_proj / d_orig) - 1.0) < 0.1


def test_sparse_random_projection_is_sparse_and_preserves_distances():
    from scipy.spatial.distance import pdist
    X, _ = make_blobs(n_samples=80, n_features=200, centers=4, random_state=0)
    srp = SparseRandomProjection(n_components=150, random_state=0).fit(X)
    zero_fraction = (srp.components_ == 0).mean()
    assert zero_fraction > 0.8
    Xt = srp.transform(X)
    assert np.corrcoef(pdist(X), pdist(Xt))[0, 1] > 0.95
    assert abs(np.median(pdist(Xt) / pdist(X)) - 1.0) < 0.1


def test_random_projection_auto_components():
    X = rng.normal(size=(500, 1000))
    grp = GaussianRandomProjection(eps=0.5, random_state=0).fit(X)
    assert grp.n_components_ == johnson_lindenstrauss_min_dim(500, eps=0.5)


# ---------------------------------------------------------------------------
# kernel approximation
# ---------------------------------------------------------------------------

def test_rbf_sampler_approximates_rbf_kernel():
    from scipy.spatial.distance import cdist
    X = rng.normal(size=(60, 4))
    gamma = 0.5
    exact = np.exp(-gamma * cdist(X, X, "sqeuclidean"))
    Xt = RBFSampler(gamma=gamma, n_components=2000, random_state=0).fit_transform(X)
    approx = Xt @ Xt.T
    assert np.abs(exact - approx).max() < 0.1


def test_rbf_sampler_matches_sklearn_shape_and_quality():
    X = rng.normal(size=(50, 5))
    ours = RBFSampler(gamma=1.0, n_components=100, random_state=0).fit_transform(X)
    ref = skka.RBFSampler(gamma=1.0, n_components=100, random_state=0).fit_transform(X)
    assert ours.shape == ref.shape
    from scipy.spatial.distance import cdist
    exact = np.exp(-1.0 * cdist(X, X, "sqeuclidean"))
    err_ours = np.abs(exact - ours @ ours.T).mean()
    err_ref = np.abs(exact - ref @ ref.T).mean()
    assert err_ours < err_ref * 2


def test_nystroem_approximates_kernel():
    from scipy.spatial.distance import cdist
    X = rng.normal(size=(80, 4))
    gamma = 0.3
    exact = np.exp(-gamma * cdist(X, X, "sqeuclidean"))
    Xt = Nystroem(gamma=gamma, n_components=60, random_state=0).fit_transform(X)
    approx = Xt @ Xt.T
    assert np.abs(exact - approx).mean() < 0.01
    ref = skka.Nystroem(gamma=gamma, n_components=60,
                        random_state=0).fit_transform(X)
    assert np.allclose(np.abs(exact - approx).mean(),
                       np.abs(exact - ref @ ref.T).mean(), rtol=1e-6)


def test_nystroem_exact_when_all_points_used():
    from scipy.spatial.distance import cdist
    X = rng.normal(size=(30, 3))
    ny = Nystroem(gamma=0.5, n_components=30, random_state=0).fit(X)
    Xt = ny.transform(X)
    exact = np.exp(-0.5 * cdist(X, X, "sqeuclidean"))
    # with every point in the basis the approximation is exact
    assert np.allclose(Xt @ Xt.T, exact, atol=1e-6)


def test_kernel_approx_makes_linear_model_nonlinear():
    X, y = make_moons(400, noise=0.2, random_state=0)
    linear = LogisticRegression().fit(X, y).score(X, y)
    approx = make_pipeline(RBFSampler(gamma=2.0, n_components=300,
                                      random_state=0),
                           LogisticRegression()).fit(X, y).score(X, y)
    assert approx > linear + 0.05
    kernel_svm = SVC(kernel="rbf", gamma=2.0, random_state=0).fit(X, y).score(X, y)
    assert approx > kernel_svm - 0.05


def test_nystroem_pipeline_matches_kernel_svm():
    X, y = make_moons(300, noise=0.2, random_state=1)
    approx = make_pipeline(Nystroem(gamma=2.0, n_components=150, random_state=0),
                           LogisticRegression()).fit(X, y).score(X, y)
    assert approx > 0.9


def test_additive_chi2_sampler():
    X = np.abs(rng.normal(size=(40, 5)))
    Xt = AdditiveChi2Sampler(sample_steps=2).fit_transform(X)
    ref = skka.AdditiveChi2Sampler(sample_steps=2).fit_transform(X)
    assert Xt.shape == ref.shape
    with pytest.raises(ValueError, match="non-negative"):
        AdditiveChi2Sampler().fit(-X)


def test_skewed_chi2_sampler():
    X = np.abs(rng.normal(size=(40, 4)))
    Xt = SkewedChi2Sampler(n_components=50, random_state=0).fit_transform(X)
    assert Xt.shape == (40, 50)
    assert np.isfinite(Xt).all()


# ---------------------------------------------------------------------------
# sparse decomposition
# ---------------------------------------------------------------------------

def test_sparse_pca_components_are_sparse():
    # data built from a few sparse atoms
    atoms = np.zeros((3, 20))
    atoms[0, :5] = 1.0
    atoms[1, 5:10] = 1.0
    atoms[2, 10:15] = 1.0
    code = np.abs(rng.normal(size=(100, 3)))
    X = code @ atoms + rng.normal(scale=0.05, size=(100, 20))
    spca = SparsePCA(n_components=3, alpha=1.0, random_state=0).fit(X)
    dense = PCA(n_components=3).fit(X)
    sparsity = (np.abs(spca.components_) < 1e-6).mean()
    dense_sparsity = (np.abs(dense.components_) < 1e-6).mean()
    assert sparsity > dense_sparsity
    assert spca.transform(X).shape == (100, 3)


def test_dictionary_learning_recovers_atoms():
    atoms = np.zeros((4, 16))
    for i in range(4):
        atoms[i, i * 4:(i + 1) * 4] = 1.0
    atoms /= np.linalg.norm(atoms, axis=1, keepdims=True)
    code = np.zeros((200, 4))
    for i in range(200):
        code[i, rng.choice(4, size=2, replace=False)] = rng.uniform(1, 3, size=2)
    X = code @ atoms
    dl = DictionaryLearning(n_components=4, alpha=0.1, max_iter=200,
                            random_state=0).fit(X)
    assert dl.components_.shape == (4, 16)
    # each true atom should be matched by some learned atom (up to sign)
    sim = np.abs(dl.components_ @ atoms.T)
    assert sim.max(axis=0).min() > 0.9
    codes = dl.transform(X)
    # every sample was built from 2 of the 4 atoms, so half the code is zero
    assert (np.abs(codes) < 1e-8).mean() > 0.3


def test_sparse_coder_with_fixed_dictionary():
    D = np.eye(5)
    X = np.array([[3.0, 0, 0, 0, 0], [0, 0, 2.0, 0, 0]])
    codes = SparseCoder(D, transform_alpha=0.1).fit().transform(X)
    assert codes.shape == (2, 5)
    assert np.argmax(np.abs(codes[0])) == 0
    assert np.argmax(np.abs(codes[1])) == 2


def test_minibatch_nmf_reconstructs():
    W0 = np.abs(rng.normal(size=(300, 3)))
    H0 = np.abs(rng.normal(size=(3, 12)))
    X = W0 @ H0
    nmf = MiniBatchNMF(n_components=3, max_iter=200, random_state=0).fit(X)
    W = nmf.transform(X)
    rel = np.linalg.norm(X - W @ nmf.components_) / np.linalg.norm(X)
    assert rel < 0.2
    assert (nmf.components_ >= 0).all() and (W >= 0).all()
    with pytest.raises(ValueError, match="non-negative"):
        MiniBatchNMF().fit(-X)


def test_lda_recovers_topics():
    # two disjoint vocabularies -> two clean topics
    r = np.random.RandomState(3)
    n_docs, vocab = 120, 10
    X = np.zeros((n_docs, vocab))
    true_topic = r.randint(0, 2, n_docs)
    for i, t in enumerate(true_topic):
        words = np.arange(0, 5) if t == 0 else np.arange(5, 10)
        X[i, words] = r.poisson(5, size=5)
    lda = LatentDirichletAllocation(n_components=2, max_iter=20,
                                    random_state=0).fit(X)
    assert lda.components_.shape == (2, vocab)
    doc_topics = lda.transform(X)
    assert np.allclose(doc_topics.sum(axis=1), 1)
    # documents of the same true topic must share a dominant learned topic
    assigned = doc_topics.argmax(axis=1)
    from nupyml.metrics import adjusted_rand_score
    assert adjusted_rand_score(true_topic, assigned) > 0.8


def test_lda_topic_word_distribution_is_concentrated():
    r = np.random.RandomState(4)
    X = np.zeros((100, 8))
    for i in range(100):
        words = np.arange(0, 4) if i % 2 == 0 else np.arange(4, 8)
        X[i, words] = r.poisson(4, size=4)
    lda = LatentDirichletAllocation(n_components=2, max_iter=20,
                                    random_state=0).fit(X)
    norm = lda.components_ / lda.components_.sum(axis=1, keepdims=True)
    # each topic should put most mass on its own half of the vocabulary
    assert max(norm[0, :4].sum(), norm[0, 4:].sum()) > 0.8


# ---------------------------------------------------------------------------
# GP kernel algebra
# ---------------------------------------------------------------------------

def test_kernel_arithmetic_builds_composites():
    k = ConstantKernel(2.0) * RBF(1.5) + WhiteKernel(0.1)
    X = rng.normal(size=(10, 2))
    K = k(X, X)
    assert K.shape == (10, 10)
    # theta collects every sub-kernel hyperparameter in log space
    assert k.n_dims == 3
    assert np.allclose(np.exp(k.theta), [2.0, 1.5, 0.1])


def test_kernel_theta_roundtrip():
    k = ConstantKernel(1.0) * Matern(2.0, nu=2.5) + WhiteKernel(0.5)
    k.theta = np.log([3.0, 4.0, 0.2])
    assert np.allclose(np.exp(k.theta), [3.0, 4.0, 0.2])
    assert k.k1.k1.constant_value == pytest.approx(3.0)
    assert k.k1.k2.length_scale == pytest.approx(4.0)
    assert k.k2.noise_level == pytest.approx(0.2)


def test_white_kernel_only_on_diagonal():
    X = rng.normal(size=(8, 2))
    Y = rng.normal(size=(5, 2))
    wk = WhiteKernel(0.3)
    assert np.allclose(wk(X, X), 0.3 * np.eye(8))
    assert np.allclose(wk(X, Y), 0.0)      # no noise shared across sets
    assert np.allclose(wk.diag(X), 0.3)


def test_ard_rbf_learns_per_dimension_scales():
    # y depends on x0 only; ARD should stretch the length scale of x1
    n = 120
    X = rng.uniform(-3, 3, size=(n, 2))
    y = np.sin(X[:, 0])
    kernel = ConstantKernel(1.0) * RBF(np.array([1.0, 1.0]))
    gp = GaussianProcessRegressor(kernel=kernel, alpha=1e-6,
                                  random_state=0).fit(X, y)
    ls = gp.kernel_.k2.length_scale
    assert len(ls) == 2
    assert ls[1] > ls[0] * 3     # the irrelevant dimension is scaled away


def test_kernel_matches_sklearn_values():
    X = rng.normal(size=(12, 3))
    pairs = [
        (RBF(1.7), skgpk.RBF(1.7)),
        (Matern(1.2, nu=1.5), skgpk.Matern(1.2, nu=1.5)),
        (Matern(0.8, nu=2.5), skgpk.Matern(0.8, nu=2.5)),
        (RationalQuadratic(1.3, 2.0), skgpk.RationalQuadratic(1.3, 2.0)),
        (ExpSineSquared(1.1, 2.5), skgpk.ExpSineSquared(1.1, 2.5)),
        (DotProduct(0.9), skgpk.DotProduct(0.9)),
    ]
    for ours, ref in pairs:
        assert np.allclose(ours(X, X), ref(X), atol=1e-8), type(ours).__name__


def test_gp_with_white_kernel_handles_noise():
    X = np.sort(rng.uniform(0, 5, size=(60, 1)), axis=0)
    y = np.sin(2 * X).ravel() + rng.normal(scale=0.3, size=60)
    kernel = ConstantKernel(1.0) * RBF(1.0) + WhiteKernel(0.1)
    gp = GaussianProcessRegressor(kernel=kernel, alpha=1e-8,
                                  random_state=0).fit(X, y)
    # the learned noise level should reflect the injected noise variance
    assert 0.02 < gp.kernel_.k2.noise_level < 0.5
    assert gp.score(X, y) > 0.7


def test_gp_periodic_kernel_extrapolates():
    X = np.linspace(0, 6, 60)[:, None]
    y = np.sin(2 * np.pi * X.ravel())
    kernel = ConstantKernel(1.0) * ExpSineSquared(1.0, 1.0)
    gp = GaussianProcessRegressor(kernel=kernel, alpha=1e-6,
                                  random_state=0).fit(X, y)
    Xq = np.linspace(6, 7, 20)[:, None]
    pred = gp.predict(Xq)
    # a periodic kernel keeps tracking the signal beyond the training range
    assert np.abs(pred - np.sin(2 * np.pi * Xq.ravel())).max() < 0.3


def test_kernel_repr_and_clone_independence():
    k = ConstantKernel(2.0) * RBF(1.0)
    c = k.clone()
    c.theta = np.log([5.0, 5.0])
    assert k.k1.constant_value == 2.0     # clone must not alias the original
    assert "RBF" in repr(k)
