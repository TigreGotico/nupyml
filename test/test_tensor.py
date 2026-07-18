"""H4: tensor decompositions -- CP, Tucker, tensor-train, non-negative CP.

Each recovers a tensor with the structure it targets: CP a low-CP-rank tensor,
Tucker a low-multilinear-rank tensor, TT any tensor exactly at full rank, and NTF
a non-negative low-rank tensor with non-negative factors.
"""
import numpy as np
import pytest

from nupyml.tensor import (unfold, fold, mode_dot, cp_decomposition, cp_to_tensor,
                          tucker_decomposition, tucker_to_tensor, tensor_train,
                          tt_to_tensor, non_negative_cp)


def _relerr(A, B):
    return np.linalg.norm(A - B) / np.linalg.norm(A)


def test_unfold_fold_roundtrip():
    rng = np.random.RandomState(0)
    T = rng.randn(4, 3, 5)
    for mode in range(3):
        assert np.allclose(fold(unfold(T, mode), mode, T.shape), T)


def test_cp_recovers_low_rank_tensor():
    rng = np.random.RandomState(0)
    factors_true = [rng.randn(d, 3) for d in (6, 5, 4)]
    T = cp_to_tensor(factors_true)
    factors = cp_decomposition(T, rank=3, n_iter=300, random_state=0)
    assert _relerr(T, cp_to_tensor(factors)) < 1e-3


def test_tucker_recovers_low_multilinear_rank():
    rng = np.random.RandomState(0)
    core0 = rng.randn(2, 2, 2)
    Us = [np.linalg.qr(rng.randn(d, 2))[0] for d in (6, 5, 4)]
    T = core0
    for n, U in enumerate(Us):
        T = mode_dot(T, U, n)
    core, factors = tucker_decomposition(T, [2, 2, 2])
    assert core.shape == (2, 2, 2)                     # compressed core
    assert _relerr(T, tucker_to_tensor(core, factors)) < 1e-8


def test_tensor_train_is_exact_at_full_rank():
    rng = np.random.RandomState(0)
    T = rng.randn(4, 3, 5, 2)
    cores = tensor_train(T, max_rank=100)
    assert len(cores) == 4
    assert cores[0].shape[0] == 1 and cores[-1].shape[-1] == 1   # boundary bonds
    assert _relerr(T, tt_to_tensor(cores)) < 1e-8


def test_tensor_train_truncation_compresses():
    rng = np.random.RandomState(0)
    # a genuinely low-TT-rank tensor: outer structure
    T = np.einsum("i,j,k->ijk", rng.randn(5), rng.randn(4), rng.randn(6))
    cores = tensor_train(T, max_rank=1)
    assert _relerr(T, tt_to_tensor(cores)) < 1e-6      # rank-1 suffices here


def test_non_negative_cp_has_nonnegative_factors():
    rng = np.random.RandomState(0)
    P = [np.abs(rng.randn(d, 3)) for d in (6, 5, 4)]
    T = cp_to_tensor(P)
    factors = non_negative_cp(T, rank=3, n_iter=400, random_state=0)
    assert all((f >= 0).all() for f in factors)
    assert _relerr(T, cp_to_tensor(factors)) < 0.05
