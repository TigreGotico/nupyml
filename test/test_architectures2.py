"""H1: neural architectures v2 -- sets, graphs, efficient attention, ODE, SSM, KAN.

Each is held to its structural property: DeepSets/SetTransformer are permutation-
invariant and learn a set function; GIN distinguishes non-isomorphic graphs;
linear attention and Neural ODE train with flowing gradients; the state-space
model carries information across a long sequence; KAN's learnable splines fit a
nonlinearity a single linear layer cannot.
"""
import numpy as np
import pytest

from nupyml import nn
from nupyml.autograd import Tensor


def _fit(model, forward, target, params, steps=200, lr=0.01):
    opt = nn.Adam(params, lr=lr)
    first = None
    for i in range(steps):
        opt.zero_grad()
        loss = ((forward() - Tensor(target)) ** 2).mean()
        if i == 0:
            first = float(loss.data)
        loss.backward()
        opt.step()
    return first, float(loss.data)


# --- DeepSets / SetTransformer -------------------------------------------

def test_deepsets_learns_set_sum_and_is_permutation_invariant():
    rng = np.random.RandomState(0)
    X = rng.uniform(0, 1, (200, 5, 3))
    y = X.sum(axis=(1, 2))                             # a symmetric target
    ds = nn.DeepSets(3, hidden=32, out_dim=1)
    first, last = _fit(ds, lambda: ds(X)[:, 0], y, ds.parameters(), steps=250)
    assert last < first and last < 0.05
    # shuffling a set's elements leaves the output unchanged
    Xp = X[:1].copy(); Xp[0] = X[0, [4, 3, 2, 1, 0]]
    assert np.allclose(ds(X[:1]).data, ds(Xp).data, atol=1e-5)


def test_set_transformer_is_permutation_invariant():
    rng = np.random.RandomState(0)
    st = nn.SetTransformer(3, embed_dim=16, out_dim=2)
    X = rng.randn(1, 6, 3)
    Xp = X[:, [5, 0, 3, 1, 4, 2]]
    assert st(X).shape == (1, 2)
    assert np.allclose(st(X).data, st(Xp).data, atol=1e-5)


# --- GIN ------------------------------------------------------------------

def test_gin_distinguishes_non_isomorphic_graphs():
    gin = nn.GIN(4, hidden=8, n_layers=2)
    X = np.ones((6, 4))                                # identical node features
    # graph A: a STAR (degrees 5,1,1,1,1,1); graph B: a 6-CYCLE (all degree 2).
    # Different degree sequences -> WL-distinguishable -> GIN must separate them.
    # (Two 2-regular graphs, e.g. triangles vs cycle, are WL-INDISTINGUISHABLE and
    #  GIN correctly cannot tell them apart -- so we don't test that.)
    Aa = np.zeros((6, 6))
    for j in range(1, 6):
        Aa[0, j] = Aa[j, 0] = 1
    Ab = np.zeros((6, 6))
    for i in range(6):
        Ab[i, (i + 1) % 6] = Ab[(i + 1) % 6, i] = 1
    ea = gin.graph_embedding(X, Aa).data
    eb = gin.graph_embedding(X, Ab).data
    assert not np.allclose(ea, eb)                     # different embeddings
    assert gin(X, Aa).shape == (6, 8)


# --- linear attention -----------------------------------------------------

def test_linear_attention_shape_and_gradient():
    la = nn.LinearAttention(16)
    x = Tensor(np.random.RandomState(0).randn(2, 12, 16), requires_grad=True)
    out = la(x)
    assert out.shape == (2, 12, 16)
    out.sum().backward()
    assert x.grad is not None and np.all(np.isfinite(x.grad))


# --- Neural ODE -----------------------------------------------------------

def test_neural_ode_trains_a_mapping():
    rng = np.random.RandomState(0)
    X = rng.randn(150, 3)
    # a smooth target reachable by evolving the state; wrap ODE + linear head
    node = nn.NeuralODE(3, hidden=16, n_steps=5)
    head = nn.Linear(3, 1)
    y = np.tanh(X.sum(axis=1))
    params = node.parameters() + list(head.parameters())
    first, last = _fit(node, lambda: head(node(X))[:, 0], y, params, steps=200)
    assert last < first                                # it learns something


# --- state-space model ----------------------------------------------------

def test_state_space_model_has_long_range_memory():
    """A copy task: the output at the LAST step must recall the FIRST input --
    information carried across the whole sequence by the linear recurrence."""
    rng = np.random.RandomState(0)
    T = 30
    u = rng.randn(200, T, 1)
    y = u[:, 0, 0]                                     # target = the first input
    ssm = nn.StateSpaceModel(1, state_dim=16, out_dim=1)
    first, last = _fit(ssm, lambda: ssm(u)[:, -1, 0], y, ssm.parameters(),
                       steps=300, lr=0.02)
    assert last < 0.5 * first                          # learns to carry it forward


# --- KAN ------------------------------------------------------------------

def test_kan_fits_a_nonlinearity_a_linear_layer_cannot():
    rng = np.random.RandomState(0)
    X = rng.uniform(-2, 2, (300, 1))
    y = np.sin(3 * X[:, 0])                            # a linear map cannot fit this
    kan = nn.KAN(1, 1, n_basis=10)
    first, last = _fit(kan, lambda: kan(X)[:, 0], y, kan.parameters(),
                       steps=300, lr=0.05)
    assert last < 0.01                                 # learnable splines fit it
    # a single linear layer, by contrast, cannot
    lin = nn.Linear(1, 1)
    _, lin_last = _fit(lin, lambda: lin(Tensor(X))[:, 0], y, list(lin.parameters()),
                       steps=300, lr=0.05)
    assert last < lin_last
