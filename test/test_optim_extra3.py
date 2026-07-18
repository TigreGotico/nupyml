"""I6: optimizers v3 -- Shampoo, LARS, LARC, NovoGrad, Adan.

Held to their defining properties: Shampoo preconditions a matrix (and falls back
to a diagonal on vectors); LARS is invariant to gradient scale (its step tracks the
weight, not the gradient, magnitude); LARC's clip lets it fine-converge where LARS
cannot; NovoGrad keeps a single scalar second moment per layer; Adan uses the
gradient difference. Each is checked to actually descend a quadratic bowl.
"""
import numpy as np
import pytest

from nupyml.nn.module import Parameter
from nupyml.nn import Shampoo, LARS, LARC, NovoGrad, Adan


def _descend(Opt, shape=(4, 3), steps=400, **kw):
    rng = np.random.RandomState(0)
    target = rng.randn(*shape)
    p = Parameter(rng.randn(*shape))
    start = np.linalg.norm(p.data - target)
    opt = Opt([p], **kw)
    for _ in range(steps):
        p.grad = 2.0 * (p.data - target)             # grad of ||w - target||^2
        opt.step()
    return start, np.linalg.norm(p.data - target)


def test_shampoo_preconditions_matrix_and_falls_back_on_vector():
    _, end = _descend(Shampoo, shape=(4, 3), lr=0.5)
    assert end < 1e-4                                 # matrix path converges
    _, end1d = _descend(Shampoo, shape=(5,), lr=0.5)
    assert end1d < 1e-4                               # 1-D Adagrad fallback converges


def test_lars_is_invariant_to_gradient_scale():
    # LARS steps a fixed fraction of the WEIGHT norm, independent of grad magnitude:
    # scaling the gradient by 100x leaves the update essentially unchanged.
    rng = np.random.RandomState(0)
    w = rng.randn(6, 4)
    g = rng.randn(6, 4)

    def one_update(scale):
        p = Parameter(w.copy())
        opt = LARS([p], lr=0.1, momentum=0.0, weight_decay=0.0)
        p.grad = g * scale
        opt.step()
        return w - p.data                            # the applied update

    u1 = one_update(1.0)
    u100 = one_update(100.0)
    assert np.allclose(u1, u100, atol=1e-6)          # scale-invariant


def test_larc_clip_enables_fine_convergence_lars_cannot():
    # Same problem, same rate: LARC's clipped trust ratio converges far tighter,
    # because near the optimum its step shrinks with the gradient while LARS's
    # fixed-fraction step keeps oscillating.
    _, lars_end = _descend(LARS, lr=0.2, weight_decay=0.0)
    _, larc_end = _descend(LARC, lr=0.2, weight_decay=0.0)
    assert larc_end < 1e-4
    assert larc_end < lars_end * 1e-3


def test_novograd_uses_scalar_second_moment_per_layer():
    _, end = _descend(NovoGrad, lr=0.1, weight_decay=0.0)
    assert end < 1e-2
    # the second-moment state is one scalar per parameter, not a full tensor
    rng = np.random.RandomState(0)
    p = Parameter(rng.randn(4, 3))
    opt = NovoGrad([p], lr=0.1)
    p.grad = np.ones((4, 3))
    opt.step()
    assert np.isscalar(opt._v[0]) or np.ndim(opt._v[0]) == 0


def test_adan_tracks_gradient_difference_and_descends():
    _, end = _descend(Adan, lr=0.2, weight_decay=0.0)
    assert end < 0.1
    # after a step the previous gradient is remembered (the difference term needs it)
    rng = np.random.RandomState(0)
    p = Parameter(rng.randn(3, 2))
    opt = Adan([p], lr=0.1)
    p.grad = np.ones((3, 2))
    opt.step()
    assert opt._prev_g[0] is not None and np.allclose(opt._prev_g[0], 1.0)


def test_adan_decoupled_weight_decay_shrinks_weights():
    # with zero gradient, decoupled decay alone must pull weights toward zero
    p = Parameter(np.ones((3, 3)) * 5.0)
    opt = Adan([p], lr=0.1, weight_decay=0.5)
    for _ in range(5):
        p.grad = np.zeros((3, 3))
        opt.step()
    assert np.all(np.abs(p.data) < 5.0)
