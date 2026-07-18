"""H2: optimizer zoo v2 -- AdaBelief, Adamax, Yogi, AdaBound, Adafactor, PCGrad,
NaturalGradient.

The adaptive-Adam variants must reach a convex minimum; Adafactor must optimise a
matrix parameter with its factored memory; PCGrad must remove the conflicting
component between two gradients; NaturalGradient must fit logistic regression (its
probabilistic home turf).
"""
import numpy as np
import pytest

from nupyml.autograd import Tensor
from nupyml.nn import (Parameter, AdaBelief, Adamax, Yogi, AdaBound, Adafactor,
                      pcgrad, NaturalGradient)


def _minimise(make_opt, steps=500):
    target = np.array([3.0, -2.0, 1.0])
    w = Parameter(np.zeros(3))
    opt = make_opt([w])
    for _ in range(steps):
        opt.zero_grad()
        ((w - Tensor(target)) ** 2).sum().backward()
        opt.step()
    return np.linalg.norm(w.data - target)


@pytest.mark.parametrize("make_opt", [
    lambda p: AdaBelief(p, lr=0.1),
    lambda p: Adamax(p, lr=0.1),
    lambda p: Yogi(p, lr=0.1),
    lambda p: AdaBound(p, lr=0.05),
], ids=["adabelief", "adamax", "yogi", "adabound"])
def test_adam_variant_reaches_minimum(make_opt):
    assert _minimise(make_opt) < 0.05


def test_adafactor_optimises_a_matrix_parameter():
    rng = np.random.RandomState(0)
    target = rng.randn(5, 4)
    w = Parameter(np.zeros((5, 4)))
    opt = Adafactor([w], lr=0.1)
    for _ in range(600):
        opt.zero_grad()
        ((w - Tensor(target)) ** 2).sum().backward()
        opt.step()
    assert np.linalg.norm(w.data - target) < 0.1
    # the factored state really is per-row + per-col (sublinear memory)
    st = opt._state[0]
    assert "r" in st and "c" in st and "v" not in st


def test_pcgrad_removes_the_conflicting_component():
    g1 = np.array([1.0, 0.0])
    g2 = np.array([-0.5, 1.0])                        # conflicts with g1 (dot < 0)
    combined = pcgrad([g1, g2])
    # projecting out the conflict then summing gives [0.8, 0.4] + [0.0, 1.0]
    assert np.allclose(combined, [0.8, 1.4])
    # non-conflicting gradients pass through as a plain sum
    assert np.allclose(pcgrad([np.array([1.0, 0.0]), np.array([0.0, 1.0])]),
                       [1.0, 1.0])


def test_natural_gradient_fits_logistic_regression():
    rng = np.random.RandomState(0)
    X = rng.randn(200, 4)
    w_true = np.array([1.5, -1.0, 0.5, 0.0])
    y = (X @ w_true + 0.2 * rng.randn(200) > 0).astype(float)
    w = Parameter(np.zeros(4))
    opt = NaturalGradient([w], lr=0.5, damping=0.1)
    Xt, yt = Tensor(X), Tensor(y)
    first = None
    for i in range(300):
        opt.zero_grad()
        p = (Xt @ w).sigmoid()
        loss = -(yt * (p + 1e-7).log() + (1 - yt) * ((1 - p) + 1e-7).log()).mean()
        if i == 0:
            first = float(loss.data)
        loss.backward()
        opt.step()
    assert float(loss.data) < 0.3 < first
    assert ((X @ w.data > 0) == y).mean() > 0.9
