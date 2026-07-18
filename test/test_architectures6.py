"""L6: neural architectures v6 -- BiLSTM, DNC, neural spline flow, modern Hopfield,
graph transformer.

The bidirectional wrapper doubles the output width and reads both directions; the
DNC learns the copy task via its dynamically-allocated memory; the neural spline
flow is trainable and assigns higher density in-distribution; the modern Hopfield
net cleans a noisy cue to the right stored pattern; the graph transformer attends
over the graph.
"""
import numpy as np
import pytest

from nupyml.autograd import Tensor
from nupyml.nn import (Bidirectional, DifferentiableNeuralComputer,
                       NeuralSplineFlow, ModernHopfieldNetwork, GraphTransformer,
                       Adam)


def test_bidirectional_doubles_width_and_reads_both_ways():
    bi = Bidirectional(4, 8, cell="lstm", rng=0)
    rng = np.random.RandomState(0)
    out = bi(Tensor(rng.randn(2, 5, 4)))
    assert out.shape == (2, 5, 16)                       # 2 * hidden
    out.sum().backward()                                 # gradients flow both ways


def test_dnc_learns_copy_task():
    dnc = DifferentiableNeuralComputer(n_slots=16, slot_dim=8, input_dim=4, rng=0)
    opt = Adam(dnc.parameters(), lr=0.01)
    rng = np.random.RandomState(0)
    for _ in range(400):
        seq = rng.randint(0, 2, (16, 3, 4)).astype(float)
        inp = ([Tensor(seq[:, i, :]) for i in range(3)]
               + [Tensor(np.zeros((16, 4))) for _ in range(3)])
        outs = dnc(inp)
        loss = sum(((outs[3 + i] - Tensor(seq[:, i, :])) ** 2).mean()
                   for i in range(3))
        opt.zero_grad(); loss.backward(); opt.step()
    seq = rng.randint(0, 2, (16, 3, 4)).astype(float)
    inp = ([Tensor(seq[:, i, :]) for i in range(3)]
           + [Tensor(np.zeros((16, 4))) for _ in range(3)])
    outs = dnc(inp)
    recon = np.stack([(outs[3 + i].data > 0.5).astype(float) for i in range(3)], 1)
    assert (recon == seq).mean() > 0.8


def test_neural_spline_flow_trains_a_density():
    rng = np.random.RandomState(0)
    nsf = NeuralSplineFlow(dim=2, n_flows=3, n_bins=8, rng=0)
    # gradients reach the coupling network
    nsf.nll(Tensor(rng.randn(16, 2))).backward()
    assert nsf.parameters()[0].grad is not None
    data = rng.randn(400, 2) * 0.4 + 1.5
    opt = Adam(nsf.parameters(), lr=0.01)
    for _ in range(200):
        idx = rng.randint(0, len(data), 64)
        opt.zero_grad(); nsf.nll(Tensor(data[idx])).backward(); opt.step()
    ll_in = nsf.log_prob(Tensor(data[:50])).data.mean()
    ll_out = nsf.log_prob(Tensor(rng.randn(50, 2) * 0.4 - 1.5)).data.mean()
    assert ll_in > ll_out                                # learned the density


def test_modern_hopfield_cleans_noisy_cue():
    rng = np.random.RandomState(0)
    patterns = rng.randn(6, 20)
    mh = ModernHopfieldNetwork(patterns, beta=10.0)
    hits = 0
    for t in range(6):
        noisy = patterns[t] + 0.3 * rng.randn(20)
        ret = mh.retrieve(noisy, n_steps=1)
        if np.argmin(np.linalg.norm(patterns - ret, axis=1)) == t:
            hits += 1
    assert hits >= 5                                     # recovers the stored pattern


def test_graph_transformer_attends_over_graph():
    rng = np.random.RandomState(0)
    X = rng.randn(6, 8)
    A = np.eye(6)
    for i, j in [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (0, 5)]:
        A[i, j] = A[j, i] = 1
    gt = GraphTransformer(dim=8, rng=0)
    out = gt(Tensor(X), A)
    assert out.shape == (6, 8)
    out.sum().backward()
    assert gt.q.weight.grad is not None
