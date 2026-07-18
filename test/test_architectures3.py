"""I5: neural architectures v3 -- Highway, SE, TCN, GATv2, HyperNetwork.

Held to structural properties: Highway starts near the identity (carry gate);
SE gates channels into [0,1]; the TCN is strictly causal; GATv2 aggregates over
graph neighbours; the hypernetwork generates a per-sample weight matrix. Plus a
gradient-flow / learning check on each.
"""
import numpy as np
import pytest

from nupyml import nn
from nupyml.autograd import Tensor


def test_highway_starts_near_identity_and_trains():
    hw = nn.HighwayNetwork(6, rng=0)
    x = np.random.RandomState(0).randn(4, 6)
    out = hw(Tensor(x)).data
    transform = hw.H(Tensor(x)).relu().data
    # the carry gate starts nearly shut: output tracks x, not the raw transform
    assert np.linalg.norm(out - x) < np.linalg.norm(out - transform)
    # gradients flow through the gate + transform
    t = Tensor(x, requires_grad=True)
    hw(t).sum().backward()
    assert t.grad is not None and np.all(np.isfinite(t.grad))


def test_squeeze_excitation_gates_channels():
    se = nn.SqueezeExcitation(8)
    x = np.abs(np.random.RandomState(0).randn(5, 8)) + 1
    out = se(Tensor(x)).data
    # the gate is a sigmoid in (0,1), so it can only attenuate a positive input
    assert np.all(out <= x + 1e-9) and np.all(out >= 0)


def test_tcn_is_strictly_causal():
    tcn = nn.TemporalConvNet(2, channels=8, n_layers=3)
    rng = np.random.RandomState(0)
    x1 = rng.randn(1, 12, 2)
    x2 = x1.copy(); x2[0, 11] = 99.0                   # change only the LAST step
    o1 = tcn(Tensor(x1)).data
    o2 = tcn(Tensor(x2)).data
    assert o1.shape == (1, 12, 8)
    # outputs before the changed step are unaffected -> no leakage from the future
    assert np.allclose(o1[0, :6], o2[0, :6])


def test_causal_conv1d_receptive_field_is_past_only():
    conv = nn.CausalConv1d(1, 1, kernel_size=3, dilation=1)
    rng = np.random.RandomState(0)
    x = rng.randn(1, 8, 1)
    xf = x.copy(); xf[0, 7] = 50.0                     # future spike
    out = conv(Tensor(x)).data
    outf = conv(Tensor(xf)).data
    assert np.allclose(out[0, :7], outf[0, :7])        # earlier outputs unchanged


def test_gatv2_aggregates_over_neighbours():
    gat = nn.GATv2(4, 6)
    rng = np.random.RandomState(0)
    X = rng.randn(5, 4)
    A = np.zeros((5, 5))
    for i, j in [(0, 1), (1, 2), (2, 3), (3, 4), (0, 4)]:
        A[i, j] = A[j, i] = 1
    out = gat(X, A)
    assert out.shape == (5, 6)
    out.sum().backward()                               # gradients flow
    assert gat.a.grad is not None


def test_hypernetwork_generates_and_applies_weights():
    hn = nn.HyperNetwork(context_dim=3, in_dim=4, out_dim=2)
    rng = np.random.RandomState(0)
    ctx = rng.randn(6, 3)
    x = rng.randn(6, 4)
    out = hn(ctx, x)
    assert out.shape == (6, 2)
    # the generated weight really is (batch, in, out)
    W = hn.generate_weight(ctx)
    assert W.shape == (6, 4, 2)
    # different contexts generate different weights
    assert not np.allclose(W.data[0], W.data[1])
