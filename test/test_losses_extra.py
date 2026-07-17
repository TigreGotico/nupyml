"""G1: additional modern losses.

Each loss is (a) gradient-checked against finite differences and (b) held to the
property it advertises -- class-balanced weights saturate, balanced-softmax cancels
the prior, poly tunes the leading CE term, focal-Tversky trades FP vs FN, GHM
harmonises by gradient density, Wing is log-then-linear, Poisson NLL handles
counts, and SupCon rewards same-class embeddings being close.
"""
import numpy as np
import pytest

from nupyml.autograd import Tensor
from nupyml import nn


def _gradcheck(loss_fn, x, *args, tol=1e-4):
    t = Tensor(x.copy(), requires_grad=True)
    loss_fn(t, *args).backward()
    ana = t.grad.copy()
    num = np.zeros_like(x)
    eps = 1e-5
    for idx in np.ndindex(*x.shape):
        xp = x.copy(); xp[idx] += eps
        xm = x.copy(); xm[idx] -= eps
        num[idx] = (loss_fn(Tensor(xp), *args).data
                    - loss_fn(Tensor(xm), *args).data) / (2 * eps)
    return np.max(np.abs(ana - num))


@pytest.fixture
def clf():
    rng = np.random.RandomState(0)
    return rng.randn(6, 4), rng.randint(0, 4, 6)


# --- gradient checks ------------------------------------------------------

def test_classification_losses_gradients(clf):
    logits, y = clf
    for loss in [nn.ClassBalancedLoss([10, 20, 5, 50], gamma=2.0),
                 nn.BalancedSoftmaxLoss([10, 20, 5, 50]),
                 nn.PolyLoss(1.0)]:
        assert _gradcheck(loss, logits, y) < 1e-4


def test_regression_and_binary_losses_gradients():
    rng = np.random.RandomState(1)
    zb, yb = rng.randn(20), rng.randint(0, 2, 20).astype(float)
    assert _gradcheck(nn.FocalTverskyLoss(), zb, yb) < 1e-4
    assert _gradcheck(nn.GHMLoss(), zb, yb) < 1e-4
    assert _gradcheck(nn.WingLoss(), rng.randn(15) * 3, rng.randn(15) * 3) < 1e-4
    counts = np.abs(rng.poisson(2, 15)).astype(float)
    assert _gradcheck(nn.PoissonNLLLoss(), rng.randn(15), counts) < 1e-4


def test_supcon_gradient():
    rng = np.random.RandomState(2)
    assert _gradcheck(nn.SupConLoss(0.2), rng.randn(8, 5), rng.randint(0, 3, 8)) < 1e-4


# --- properties -----------------------------------------------------------

def test_class_balanced_weights_saturate():
    # a class with 10x more samples gets LESS than 10x less weight (effective number)
    cb = nn.ClassBalancedLoss([100, 1000], beta=0.999)
    ratio = cb.weights[0] / cb.weights[1]            # rare / common weight
    assert 1 < ratio < 10                            # not the naive 10x of 1/freq


def test_balanced_softmax_reduces_to_ce_when_balanced(clf):
    """With equal class counts the log-prior term is constant, so balanced
    softmax IS plain cross-entropy; under imbalance it genuinely differs."""
    logits, y = clf
    bs_bal = nn.BalancedSoftmaxLoss([25, 25, 25, 25])(Tensor(logits), y).data
    ce = nn.CrossEntropyLoss()(Tensor(logits), y).data
    assert abs(bs_bal - ce) < 1e-9
    bs_imb = nn.BalancedSoftmaxLoss([1000, 10, 100, 5])(Tensor(logits), y).data
    assert abs(bs_imb - ce) > 1e-3                   # the prior adjustment bites


def test_poly_loss_reduces_to_ce_at_zero_epsilon(clf):
    logits, y = clf
    poly0 = nn.PolyLoss(0.0)(Tensor(logits), y).data
    ce = nn.CrossEntropyLoss()(Tensor(logits), y).data
    assert abs(poly0 - ce) < 1e-9
    # positive epsilon adds the (1 - p_true) correction -> larger loss
    assert nn.PolyLoss(1.0)(Tensor(logits), y).data > ce


def test_focal_tversky_beta_penalises_false_negatives():
    # prediction misses positives (false negatives): low prob where target is 1
    logits = np.array([-3.0, -3.0, -3.0, 0.0])       # under-predicts the positives
    y = np.array([1.0, 1.0, 1.0, 0.0])
    high_fn = nn.FocalTverskyLoss(alpha=0.1, beta=0.9)(Tensor(logits), y).data
    low_fn = nn.FocalTverskyLoss(alpha=0.9, beta=0.1)(Tensor(logits), y).data
    assert high_fn > low_fn                          # beta punishes missed positives


def test_wing_loss_is_log_small_linear_large():
    wl = nn.WingLoss(w=10.0, eps=2.0)
    small = wl(Tensor(np.array([0.5])), np.array([0.0])).data
    large = wl(Tensor(np.array([50.0])), np.array([0.0])).data
    # large-error branch is |x| - C ; small-error is the log region
    assert large == pytest.approx(50.0 - wl.C, abs=1e-6)
    assert small == pytest.approx(10.0 * np.log(1 + 0.5 / 2.0), abs=1e-6)


def test_poisson_nll_minimised_at_true_rate():
    # log_input: rate = exp(input); loss minimal when exp(input) == mean(target)
    counts = np.array([3.0, 3.0, 3.0, 3.0])
    good = nn.PoissonNLLLoss()(Tensor(np.log(np.array([3.0, 3.0, 3.0, 3.0]))), counts).data
    bad = nn.PoissonNLLLoss()(Tensor(np.log(np.array([1.0, 1.0, 1.0, 1.0]))), counts).data
    assert good < bad


def test_supcon_lower_when_same_class_embeddings_cluster():
    rng = np.random.RandomState(0)
    labels = np.array([0, 0, 1, 1])
    # clustered: same-label embeddings nearly identical
    clustered = np.array([[1, 0], [1, 0], [0, 1], [0, 1]], float) + 0.01 * rng.randn(4, 2)
    scattered = rng.randn(4, 2)
    loss = nn.SupConLoss(0.1)
    assert loss(Tensor(clustered), labels).data < loss(Tensor(scattered), labels).data
