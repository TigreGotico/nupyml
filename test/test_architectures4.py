"""J8: neural architectures v4 -- WaveNet, capsule net, spatial transformer,
echo state network, pointer network.

Held to their defining properties: WaveNet is strictly causal with gated
activations; capsule lengths are probabilities in [0,1); the spatial transformer's
identity theta is a no-op and a translation theta shifts the image; the echo state
network fits a temporal signal with only a linear readout; the pointer network
learns to output an ordering of its variable-length input.
"""
import numpy as np
import pytest

from nupyml.autograd import Tensor
from nupyml.autograd import functional as F
from nupyml.nn import (WaveNet, CapsuleLayer, SpatialTransformer,
                       EchoStateNetwork, PointerNetwork)


def test_wavenet_is_causal_and_backprops():
    wn = WaveNet(in_ch=1, residual_ch=8, skip_ch=8, out_ch=1, n_layers=3)
    rng = np.random.RandomState(0)
    x = rng.randn(2, 16, 1)
    out = wn(Tensor(x))
    assert out.shape == (2, 16, 1)
    x2 = x.copy(); x2[0, 15] = 9.0                    # change only the last step
    assert np.allclose(wn(Tensor(x)).data[0, :8], wn(Tensor(x2)).data[0, :8])
    wn(Tensor(x)).sum().backward()                   # gradients flow


def test_capsule_lengths_are_probabilities():
    cap = CapsuleLayer(n_in=6, d_in=4, n_out=3, d_out=4, routing_iters=3)
    u = Tensor(np.random.RandomState(0).randn(5, 6, 4))
    lengths = cap.lengths(u).data
    assert lengths.shape == (5, 3)
    assert lengths.max() < 1.0 and lengths.min() >= 0.0   # squashed to [0,1)
    cap.forward(u).sum().backward()                   # differentiable w.r.t. W


def test_capsule_routing_concentrates_on_agreement():
    # feed identical input capsules; routing should raise one output's length
    cap = CapsuleLayer(n_in=8, d_in=4, n_out=2, d_out=4, routing_iters=1, rng=0)
    x = Tensor(np.tile(np.random.RandomState(0).randn(1, 1, 4), (1, 8, 1)))
    len1 = cap.lengths(x).data
    cap3 = CapsuleLayer(n_in=8, d_in=4, n_out=2, d_out=4, routing_iters=3, rng=0)
    len3 = cap3.lengths(x).data
    # more routing iterations sharpen the winner (higher max length)
    assert len3.max() >= len1.max() - 1e-6


def test_spatial_transformer_identity_and_translation():
    st = SpatialTransformer(8, 8)
    rng = np.random.RandomState(0)
    img = rng.rand(1, 8, 8)
    identity = np.array([[[1., 0, 0], [0, 1, 0]]])
    out = st.transform(img, identity)
    assert np.allclose(out[0], img[0], atol=1e-6)     # identity is a no-op
    # a horizontal shift in normalised coords moves content left/right
    shift = np.array([[[1., 0, 0.5], [0, 1, 0]]])
    shifted = st.transform(img, shift)
    assert not np.allclose(shifted[0], img[0])


def test_echo_state_network_forecasts_sine():
    t = np.linspace(0, 8 * np.pi, 300)
    s = np.sin(t)
    X, y = s[:-1], s[1:]                               # one-step-ahead
    esn = EchoStateNetwork(n_reservoir=100, spectral_radius=0.9,
                           random_state=0).fit(X, y)
    pred = esn.predict(X)
    assert np.mean((pred[20:] - y[20:]) ** 2) < 1e-2  # readout captures dynamics


def test_pointer_network_learns_to_sort():
    rng = np.random.RandomState(0)
    pn = PointerNetwork(hidden=32, rng=0)
    params = pn.parameters()
    for _ in range(600):
        seq = rng.rand(5)
        target = np.argsort(seq)
        logits = pn.forward(seq, targets=target)      # teacher forcing
        onehot = np.eye(5)[target]
        logp = F.log_softmax(logits, axis=1)
        loss = -(logp * Tensor(onehot)).sum(axis=1).mean()
        for p in params:
            p.zero_grad()
        loss.backward()
        for p in params:
            p.data -= 0.1 * p.grad
    seqs = [rng.rand(5) for _ in range(50)]
    acc = np.mean([(pn.predict_order(s) == np.argsort(s)).mean() for s in seqs])
    assert acc > 0.9
