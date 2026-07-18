"""H5: robust PCA and matrix completion.

Robust PCA must recover a low-rank matrix hidden under sparse gross corruption
(where ordinary PCA fails); matrix completion must fill missing entries of a
low-rank matrix; SVT must shrink the rank.
"""
import numpy as np
import pytest

from nupyml.matrix import singular_value_threshold, RobustPCA, SoftImpute


def test_svt_shrinks_rank():
    rng = np.random.RandomState(0)
    X = rng.randn(20, 15)
    thr = singular_value_threshold(X, tau=2.0)
    # thresholding singular values drops the small ones -> lower rank
    assert np.linalg.matrix_rank(thr, tol=1e-6) < np.linalg.matrix_rank(X)


def test_robust_pca_separates_low_rank_from_sparse():
    rng = np.random.RandomState(0)
    m, n, r = 40, 30, 3
    L0 = rng.randn(m, r) @ rng.randn(r, n)            # true low-rank part
    S0 = np.zeros((m, n))
    idx = rng.rand(m, n) < 0.1                        # 10% gross corruption
    S0[idx] = rng.randn(idx.sum()) * 5
    rp = RobustPCA().fit(L0 + S0)
    assert np.linalg.norm(rp.low_rank_ - L0) / np.linalg.norm(L0) < 0.05
    assert np.linalg.matrix_rank(rp.low_rank_, tol=1e-3) == r
    # the sparse part is genuinely sparse and lands on the corrupted entries
    assert (np.abs(rp.sparse_) > 1e-3).mean() < 0.2


def test_robust_pca_beats_plain_pca_under_corruption():
    rng = np.random.RandomState(1)
    m, n, r = 30, 25, 2
    L0 = rng.randn(m, r) @ rng.randn(r, n)
    S0 = np.zeros((m, n)); S0[rng.rand(m, n) < 0.08] = rng.randn() * 8
    M = L0 + S0
    rp = RobustPCA().fit(M)
    # plain rank-r SVD truncation is dragged off by the corruption
    U, s, Vt = np.linalg.svd(M, full_matrices=False)
    pca_L = (U[:, :r] * s[:r]) @ Vt[:r]
    assert (np.linalg.norm(rp.low_rank_ - L0)
            < np.linalg.norm(pca_L - L0))


def test_soft_impute_completes_missing_entries():
    rng = np.random.RandomState(0)
    m, n, r = 40, 30, 3
    full = rng.randn(m, r) @ rng.randn(r, n)
    M = full.copy()
    miss = rng.rand(m, n) < 0.4
    M[miss] = np.nan
    comp = SoftImpute(lam=2.0, max_iter=300).fit_transform(M)
    assert not np.any(np.isnan(comp))
    # missing entries recovered, observed entries preserved
    assert np.linalg.norm((comp - full)[miss]) / np.linalg.norm(full[miss]) < 0.3
    assert np.allclose(comp[~miss], full[~miss], atol=1e-6)
