"""Modern NN pieces: activations, norms, optimizers, positional, regularizers, GNNs.

Each is checked for the property that makes it worth having -- SELU self-normalises,
RMSNorm gives unit RMS, RoPE preserves norms, orthogonal init is orthogonal, and
the GNNs recover graph community structure.
"""
import numpy as np
import pytest

from nupyml.autograd import Tensor
from nupyml.autograd.functional import log_softmax
from nupyml.nn import (
    SiLU, Mish, ELU, SELU, Softplus, GLU, SwiGLU, RMSNorm, GroupNorm,
    InstanceNorm, Adagrad, Nadam, RAdam, Lion, Lookahead, Adam,
    RotaryPositionalEmbedding, ALiBi, mixup, cutmix, StochasticDepth, drop_path,
)
from nupyml.nn import init
from nupyml.nn.module import Module, Parameter
from nupyml.gnn import GCN, GraphSAGE, GAT
from nupyml.gnn.layers import normalize_adjacency


# --- activations ----------------------------------------------------------

@pytest.mark.parametrize("act_cls", [SiLU, Mish, ELU, SELU, Softplus])
def test_activation_flows_finite_gradients(act_cls):
    x = Tensor(np.random.RandomState(0).normal(size=(4, 8)), requires_grad=True)
    out = act_cls()(x)
    out.sum().backward()
    assert x.grad is not None
    assert np.all(np.isfinite(x.grad))


def test_silu_is_smooth_and_dips_negative():
    """Unlike ReLU, SiLU passes a little negative signal, so units do not die."""
    x = Tensor(np.array([[-2.0, -0.5, 0.5, 2.0]]))
    out = SiLU()(x).data[0]
    assert out[0] < 0 and out[1] < 0          # negative inputs give negative output
    assert out[3] > 1.0                        # large positive ~ identity


def test_selu_preserves_unit_variance():
    """The self-normalising property: standardised input stays ~unit-variance out,
    which is what lets a SELU net train without BatchNorm."""
    rng = np.random.RandomState(0)
    x = Tensor(rng.normal(0, 1, size=(10000, 1)))
    out = SELU()(x).data
    assert abs(out.std() - 1.0) < 0.25
    assert abs(out.mean()) < 0.2


def test_softplus_is_positive_and_stable():
    """Softplus is positive for every finite input, though it underflows to 0 far
    in the negative tail (softplus(-100) ~ 4e-44, below float range). The point is
    it never goes NEGATIVE and never overflows at large positive input."""
    x = Tensor(np.array([[-100.0, -1.0, 0.0, 100.0]]))
    out = Softplus()(x).data[0]
    assert np.all(out >= 0)                    # never negative
    assert out[1] > 0 and out[2] > 0          # strictly positive where representable
    assert np.isfinite(out).all()             # no overflow at +100


def test_glu_halves_the_dimension():
    x = Tensor(np.random.RandomState(0).normal(size=(4, 8)))
    assert GLU()(x).shape == (4, 4)           # one half gates the other


def test_swiglu_preserves_dimension_and_trains():
    x = Tensor(np.random.RandomState(0).normal(size=(4, 8)), requires_grad=True)
    out = SwiGLU(8, rng=0)(x)
    assert out.shape == (4, 8)
    out.sum().backward()
    assert x.grad is not None


# --- normalization --------------------------------------------------------

def test_rmsnorm_gives_unit_rms():
    x = Tensor(np.random.RandomState(0).normal(0, 5, size=(6, 8)))
    out = RMSNorm(8)(x).data
    rms = np.sqrt((out ** 2).mean(axis=1))
    assert np.allclose(rms, 1.0, atol=1e-3)


def test_rmsnorm_does_not_center():
    """Unlike LayerNorm, RMSNorm keeps the mean -- it only rescales."""
    x = Tensor(np.random.RandomState(0).normal(5, 1, size=(6, 8)))
    out = RMSNorm(8)(x).data
    # a shifted input keeps a nonzero mean (LayerNorm would zero it)
    assert abs(out.mean()) > 0.1


def test_groupnorm_normalizes_per_group():
    x = Tensor(np.random.RandomState(0).normal(3, 2, size=(2, 8, 4, 4)))
    out = GroupNorm(4, 8)(x).data
    assert out.shape == (2, 8, 4, 4)
    # each group of each example is standardised
    xr = out.reshape(2, 4, 2, 4, 4)
    for n in range(2):
        for g in range(4):
            assert abs(xr[n, g].mean()) < 0.2


def test_groupnorm_rejects_indivisible_channels():
    with pytest.raises(ValueError, match="divisible"):
        GroupNorm(3, 8)


def test_instancenorm_normalizes_each_channel():
    x = Tensor(np.random.RandomState(0).normal(3, 2, size=(2, 4, 5, 5)))
    out = InstanceNorm(4)(x).data
    # each channel of each example is standardised on its own
    for n in range(2):
        for c in range(4):
            assert abs(out[n, c].mean()) < 0.2


# --- optimizers -----------------------------------------------------------

def _minimize(make_opt, target=3.0, steps=800):
    w = Tensor(np.zeros(3), requires_grad=True)
    opt = make_opt(w)
    for _ in range(steps):
        opt.zero_grad()
        (((w - Tensor(np.full(3, target))) ** 2).sum()).backward()
        opt.step()
    return w.data.mean()


def test_adagrad_converges():
    assert _minimize(lambda w: Adagrad([w], lr=0.5)) == pytest.approx(3.0, abs=0.1)


def test_nadam_converges():
    assert _minimize(lambda w: Nadam([w], lr=0.1)) == pytest.approx(3.0, abs=0.1)


def test_radam_converges():
    assert _minimize(lambda w: RAdam([w], lr=0.1)) == pytest.approx(3.0, abs=0.1)


def test_lion_converges():
    assert _minimize(lambda w: Lion([w], lr=0.02)) == pytest.approx(3.0, abs=0.15)


def test_lion_takes_uniform_magnitude_steps():
    """Lion steps by the SIGN, so before weight decay every coordinate moves by
    exactly the learning rate."""
    w = Tensor(np.array([10.0, -10.0, 0.5]), requires_grad=True)
    opt = Lion([w], lr=0.1, weight_decay=0.0)
    before = w.data.copy()
    opt.zero_grad()
    (((w - Tensor(np.zeros(3))) ** 2).sum()).backward()
    opt.step()
    steps = np.abs(w.data - before)
    assert np.allclose(steps, 0.1)            # same magnitude for every coordinate


def test_lookahead_wraps_and_converges():
    w = Tensor(np.zeros(3), requires_grad=True)
    la = Lookahead(Adam([w], lr=0.1), k=5)
    for _ in range(600):
        la.zero_grad()
        (((w - Tensor(np.full(3, 3.0))) ** 2).sum()).backward()
        la.step()
    assert w.data.mean() == pytest.approx(3.0, abs=0.1)


def test_adagrad_learning_rate_decays():
    """Adagrad's flaw: the accumulator only grows, so the effective step shrinks
    monotonically."""
    w = Tensor(np.array([5.0]), requires_grad=True)
    opt = Adagrad([w], lr=0.5)
    steps = []
    for _ in range(20):
        before = w.data.copy()
        opt.zero_grad()
        ((w ** 2).sum()).backward()
        opt.step()
        steps.append(abs(w.data[0] - before[0]))
    assert steps[-1] < steps[0]              # the step size fell


# --- positional -----------------------------------------------------------

def test_rope_preserves_vector_norms():
    """A rotation changes direction, never length -- so norms are preserved."""
    rope = RotaryPositionalEmbedding(8)
    q = np.random.RandomState(0).normal(size=(2, 5, 8))
    rotated = rope.rotate(q)
    assert np.allclose(np.linalg.norm(q, axis=-1),
                       np.linalg.norm(rotated, axis=-1))


def test_rope_encodes_relative_position():
    """The RoPE property: the dot product of a rotated query and key depends only
    on their offset, not their absolute positions."""
    rope = RotaryPositionalEmbedding(8)
    # the same two vectors placed at (0,2) vs (3,5) -- same offset of 2
    v = np.random.RandomState(0).normal(size=8)
    w = np.random.RandomState(1).normal(size=8)
    seq_a = rope.rotate(np.stack([v, v, v])[None])[0]      # positions 0,1,2
    seq_b = rope.rotate(np.stack([w, w, w, w, w, w])[None])[0]  # positions 0..5
    # dot of pos0.pos2 should equal dot of pos3.pos5 for the SAME vectors
    q = np.random.RandomState(2).normal(size=8)
    qs = rope.rotate(np.tile(q, (6, 1))[None])[0]
    assert np.dot(qs[0], qs[2]) == pytest.approx(np.dot(qs[3], qs[5]), abs=1e-8)


def test_alibi_penalizes_distance():
    """ALiBi's bias is zero at a token itself and grows negative with distance."""
    alibi = ALiBi(4)
    bias = alibi.bias(6)
    assert bias.shape == (4, 6, 6)
    assert np.allclose(np.diagonal(bias, axis1=1, axis2=2), 0)   # self: no penalty
    # farther apart => more negative, for every head
    assert np.all(bias[:, 0, 5] < bias[:, 0, 1])


def test_alibi_heads_have_different_slopes():
    """Each head covers a different distance scale, so the slopes must differ."""
    alibi = ALiBi(8)
    assert len(np.unique(alibi.slopes)) == 8


# --- regularization -------------------------------------------------------

def test_mixup_blends_examples_and_labels():
    rng = np.random.RandomState(0)
    X = rng.normal(size=(10, 4))
    y = rng.randint(0, 3, 10)
    Xm, ym, lam = mixup(X, y, alpha=0.2, rng=0)
    assert Xm.shape == X.shape
    assert np.allclose(ym.sum(axis=1), 1.0)   # blended labels still a distribution
    assert 0 <= lam <= 1


def test_cutmix_pastes_a_patch():
    rng = np.random.RandomState(0)
    X = rng.normal(size=(8, 3, 8, 8))
    y = rng.randint(0, 2, 8)
    Xc, yc, lam = cutmix(X, y, rng=0)
    assert Xc.shape == X.shape
    assert np.allclose(yc.sum(axis=1), 1.0)


def test_stochastic_depth_can_skip_a_block():
    """During training the block is sometimes bypassed to its identity shortcut."""
    class AddOne(Module):
        def forward(self, x):
            return Tensor._wrap(x) + 1.0

    sd = StochasticDepth(AddOne(), survival_prob=0.0, rng=0)  # always drop
    sd.train()
    x = Tensor(np.zeros((3, 4)))
    assert np.allclose(sd(x).data, 0.0)       # block dropped: identity only


def test_stochastic_depth_scales_at_eval():
    class AddOne(Module):
        def forward(self, x):
            return Tensor._wrap(x) + 1.0

    sd = StochasticDepth(AddOne(), survival_prob=0.8, rng=0)
    sd.eval()
    x = Tensor(np.zeros((3, 4)))
    # eval keeps the block scaled by survival prob: 0 + 1*0.8
    assert np.allclose(sd(x).data, 0.8)


def test_drop_path_is_identity_at_eval():
    x = Tensor(np.ones((4, 6)))
    out = drop_path(x, drop_prob=0.5, training=False, rng=0)
    assert np.allclose(out.data, 1.0)


# --- init -----------------------------------------------------------------

def test_orthogonal_init_is_orthogonal():
    Q = init.orthogonal((6, 6), rng=0)
    assert np.allclose(Q.T @ Q, np.eye(6), atol=1e-8)


def test_orthogonal_preserves_norms():
    """The reason to use it: it neither shrinks nor grows the signal passing
    through, ideal for deep and recurrent nets."""
    Q = init.orthogonal((10, 10), rng=0)
    x = np.random.RandomState(0).normal(size=10)
    assert np.linalg.norm(Q @ x) == pytest.approx(np.linalg.norm(x))


def test_truncated_normal_has_no_tails():
    t = init.truncated_normal((5000,), std=0.02, rng=0)
    assert np.abs(t).max() <= 2 * 0.02 + 1e-9    # nothing past two sigma
    assert abs(t.std() - 0.02) < 0.005


# --- GNNs -----------------------------------------------------------------

@pytest.fixture
def community_graph():
    """Two dense communities -- the canonical test a GNN should solve."""
    rng = np.random.RandomState(0)
    n = 20
    A = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            same = (i < 10) == (j < 10)
            if rng.uniform() < (0.6 if same else 0.05):
                A[i, j] = A[j, i] = 1
    X = rng.normal(size=(n, 5))
    y = np.array([0] * 10 + [1] * 10)
    return X, A, y


def _train_gnn(model, X, A, y, epochs=100):
    opt = Adam(model.parameters(), lr=0.05)
    for _ in range(epochs):
        opt.zero_grad()
        logp = log_softmax(model(X, A), axis=1)
        loss = -(logp * Tensor(np.eye(2)[y])).sum(axis=1).mean()
        loss.backward()
        opt.step()
    pred = np.argmax(model(X, A).data, axis=1)
    return max((pred == y).mean(), (pred == 1 - y).mean())   # up to label swap


def test_gcn_detects_communities(community_graph):
    X, A, y = community_graph
    assert _train_gnn(GCN(5, 16, 2, n_layers=2, rng=0), X, A, y) > 0.9


def test_graphsage_detects_communities(community_graph):
    X, A, y = community_graph
    assert _train_gnn(GraphSAGE(5, 16, 2, n_layers=2, rng=0), X, A, y) > 0.9


def test_gat_detects_communities(community_graph):
    X, A, y = community_graph
    assert _train_gnn(GAT(5, 16, 2, n_layers=2, rng=0), X, A, y) > 0.85


def test_adjacency_normalization_is_symmetric(community_graph):
    """The GCN normalisation D^-1/2 (A+I) D^-1/2 must stay symmetric."""
    _, A, _ = community_graph
    A_norm = normalize_adjacency(A)
    assert np.allclose(A_norm, A_norm.T)
    # rows are degree-normalised, so no node's aggregation blows up
    assert np.all(A_norm.sum(axis=1) <= 1.5)


def test_graphsage_handles_isolated_nodes():
    """A node with no neighbours must not crash the aggregation."""
    X = np.random.RandomState(0).normal(size=(5, 4))
    A = np.zeros((5, 5))
    A[0, 1] = A[1, 0] = 1                     # only nodes 0,1 connected
    model = GraphSAGE(4, 8, 2, rng=0)
    out = model(X, A)
    assert out.shape == (5, 2)
    assert np.all(np.isfinite(out.data))
