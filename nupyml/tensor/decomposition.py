"""Tensor decompositions -- the multi-way generalisation of PCA / NMF.

A matrix decomposition (SVD, NMF) factors a 2-way array. Real data is often
MULTI-way: users x items x time, or subjects x voxels x conditions. Flattening it
to a matrix loses the multi-way structure; tensor decompositions preserve it.
Four classic factorizations here: CP, Tucker, tensor-train, and non-negative CP.
"""
import numpy as np


# --- helpers --------------------------------------------------------------

def unfold(tensor, mode):
    """Matricise a tensor along ``mode`` (mode-n unfolding)."""
    return np.moveaxis(tensor, mode, 0).reshape(tensor.shape[mode], -1)


def fold(matrix, mode, shape):
    """Inverse of :func:`unfold`."""
    full = [shape[mode]] + [s for i, s in enumerate(shape) if i != mode]
    return np.moveaxis(matrix.reshape(full), 0, mode)


def mode_dot(tensor, matrix, mode):
    """n-mode product: multiply ``tensor`` by ``matrix`` along ``mode``."""
    new_shape = list(tensor.shape)
    new_shape[mode] = matrix.shape[0]
    return fold(matrix @ unfold(tensor, mode), mode, new_shape)


def khatri_rao(matrices):
    """Column-wise Kronecker product of a list of matrices (shared #columns)."""
    r = matrices[0].shape[1]
    out = matrices[0]
    for m in matrices[1:]:
        out = (out[:, None, :] * m[None, :, :]).reshape(-1, r)
    return out


# --- CP / PARAFAC ---------------------------------------------------------

def cp_decomposition(tensor, rank, n_iter=100, tol=1e-8, random_state=None):
    """CP / PARAFAC: a tensor as a sum of ``rank`` rank-1 tensors, via ALS.

    THE MODEL
    ---------
    Write the tensor as ``sum_r a_r ∘ b_r ∘ c_r`` -- rank-1 outer products, one set
    of factor vectors per mode. Unlike matrix rank, this is UNIQUE under mild
    conditions (no rotation ambiguity), so the factors are interpretable: each
    component is one latent pattern present across all modes simultaneously.

    THE FIT (alternating least squares)
    -----------------------------------
    Fixing all factor matrices but one turns the problem into a linear least-
    squares solve for that one (via the mode-n unfolding and a Khatri-Rao product).
    Cycle through the modes until convergence. Returns the list of factor matrices;
    ``cp_to_tensor`` reconstructs.
    """
    from ..utils import check_random_state
    rng = check_random_state(random_state)
    shape = tensor.shape
    factors = [rng.standard_normal((d, rank)) for d in shape]
    prev_err = np.inf
    norm_t = np.linalg.norm(tensor)
    for _ in range(n_iter):
        for n in range(len(shape)):
            # V = Hadamard product of Gram matrices of the OTHER factors
            V = np.ones((rank, rank))
            kr_parts = []
            for m in range(len(shape)):
                if m != n:
                    V *= factors[m].T @ factors[m]
                    kr_parts.append(factors[m])
            kr = khatri_rao(kr_parts)                 # increasing mode order matches unfold
            factors[n] = unfold(tensor, n) @ kr @ np.linalg.pinv(V)
        err = np.linalg.norm(tensor - cp_to_tensor(factors)) / (norm_t + 1e-12)
        if abs(prev_err - err) < tol:
            break
        prev_err = err
    return factors


def cp_to_tensor(factors):
    """Reconstruct a tensor from its CP factor matrices."""
    shape = [f.shape[0] for f in factors]
    kr = khatri_rao(factors[1:])
    return fold(factors[0] @ kr.T, 0, shape)


# --- Tucker / HOSVD -------------------------------------------------------

def tucker_decomposition(tensor, ranks):
    """Tucker / HOSVD: a CORE tensor times a factor matrix per mode.

    Tucker generalises PCA to tensors: each mode gets an orthogonal basis (the top
    singular vectors of its unfolding), and a small CORE tensor holds the
    interactions between the modes' components. Unlike CP (a strict sum of rank-1
    terms), Tucker allows FULL interaction among components -- more flexible, at the
    cost of CP's uniqueness. ``ranks`` is the retained rank per mode. Returns
    ``(core, factors)``; ``tucker_to_tensor`` reconstructs.
    """
    factors = []
    for n in range(tensor.ndim):
        U, _, _ = np.linalg.svd(unfold(tensor, n), full_matrices=False)
        factors.append(U[:, :ranks[n]])               # leading left singular vecs
    core = tensor.copy()
    for n, U in enumerate(factors):
        core = mode_dot(core, U.T, n)                  # project onto each basis
    return core, factors


def tucker_to_tensor(core, factors):
    out = core
    for n, U in enumerate(factors):
        out = mode_dot(out, U, n)
    return out


# --- tensor train ---------------------------------------------------------

def tensor_train(tensor, max_rank):
    """TT-SVD: factor a tensor into a CHAIN of 3-way cores (Oseledets, 2011).

    A d-way tensor has exponentially many entries in d -- the curse of
    dimensionality. Tensor-train writes it as a chain of small 3-way cores
    ``G_1 ... G_d`` contracted along shared bond indices, so storage grows only
    LINEARLY in d (given bounded ranks). Built by a sequence of SVDs: reshape,
    factor, carry the remainder forward. This is what makes very high-order
    tensors (and the quantum states / neural-net weights they model) tractable.
    Returns the list of cores; ``tt_to_tensor`` reconstructs.
    """
    shape = tensor.shape
    d = len(shape)
    cores = []
    r_prev = 1
    C = tensor.reshape(1, -1)
    for k in range(d - 1):
        C = C.reshape(r_prev * shape[k], -1)
        U, s, Vt = np.linalg.svd(C, full_matrices=False)
        r = min(max_rank, len(s))
        U, s, Vt = U[:, :r], s[:r], Vt[:r]
        cores.append(U.reshape(r_prev, shape[k], r))
        C = np.diag(s) @ Vt
        r_prev = r
    cores.append(C.reshape(r_prev, shape[-1], 1))
    return cores


def tt_to_tensor(cores):
    shape = [c.shape[1] for c in cores]
    out = cores[0]                                     # (1, n0, r0)
    for c in cores[1:]:
        # contract the shared bond dimension
        out = np.tensordot(out, c, axes=([out.ndim - 1], [0]))
    return out.reshape(shape)


# --- non-negative CP ------------------------------------------------------

def non_negative_cp(tensor, rank, n_iter=200, random_state=None):
    """Non-negative CP: CP with all factors >= 0, by MULTIPLICATIVE updates.

    When the data is non-negative (counts, intensities, spectra), forcing the
    factors non-negative -- as NMF does for matrices -- yields PARTS-BASED,
    additive components you can actually interpret (no cancelling positives and
    negatives). Fit by the same multiplicative update rule as NMF, applied per
    mode: each factor is scaled by a ratio of the positive data term to the
    reconstruction term, so it stays non-negative and the error decreases
    monotonically. Requires a non-negative tensor.
    """
    from ..utils import check_random_state
    rng = check_random_state(random_state)
    shape = tensor.shape
    factors = [np.abs(rng.standard_normal((d, rank))) + 1e-2 for d in shape]
    for _ in range(n_iter):
        for n in range(len(shape)):
            kr_parts = [factors[m] for m in range(len(shape)) if m != n]
            kr = khatri_rao(kr_parts)
            Xn = unfold(tensor, n)
            numer = Xn @ kr
            denom = factors[n] @ (kr.T @ kr) + 1e-12
            factors[n] *= numer / denom                # multiplicative -> stays >= 0
    return factors


__all__ = ["unfold", "fold", "mode_dot", "khatri_rao",
           "cp_decomposition", "cp_to_tensor",
           "tucker_decomposition", "tucker_to_tensor",
           "tensor_train", "tt_to_tensor", "non_negative_cp"]
