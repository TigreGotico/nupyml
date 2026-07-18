"""Neural architectures v4: audio generation, routing, spatial attention,
reservoirs, and pointing.

Five architectures that each break an assumption of the plain feed-forward net.
WaveNet stacks dilated causal convolutions with GATED activations to model raw
sequences. The capsule network replaces scalar neurons with VECTORS and routes by
agreement. The spatial transformer lets a net WARP its own input. The echo state
network leaves its recurrence RANDOM and trains only a linear readout. The pointer
network's output is a distribution over its INPUT positions.
"""
import numpy as np

from ..autograd import Tensor
from ..autograd import functional as F
from ..base import BaseEstimator
from ..utils import check_random_state
from .module import Module, Parameter
from .layers import Linear
from ._architectures3 import CausalConv1d


class WaveNet(Module):
    """Dilated causal convolutions with GATED activations (van den Oord, 2016).

    Built for raw audio: a stack of causal convolutions whose dilation DOUBLES each
    layer, so a modest stack sees thousands of past samples, and each layer uses a
    GATED activation ``tanh(conv_f(x)) * sigmoid(conv_g(x))`` -- the sigmoid gate
    decides how much of the tanh-filtered signal passes, an LSTM-style gate in
    convolutional form. Residual connections carry the input forward while SKIP
    connections from every layer are summed into the output, so gradients and
    features from all depths reach the end. Maps ``(batch, length, in_ch)`` to
    ``(batch, length, out_ch)``, strictly causal.
    """

    def __init__(self, in_ch, residual_ch=16, skip_ch=16, out_ch=1, n_layers=4,
                 kernel_size=2, rng=None):
        super().__init__()
        r = check_random_state(rng)
        self.start = Linear(in_ch, residual_ch, rng=r)
        self.filters, self.gates, self.res, self.skip = [], [], [], []
        for i in range(n_layers):
            d = 2 ** i
            self.filters.append(CausalConv1d(residual_ch, residual_ch,
                                             kernel_size, dilation=d, rng=r))
            self.gates.append(CausalConv1d(residual_ch, residual_ch,
                                           kernel_size, dilation=d, rng=r))
            self.res.append(Linear(residual_ch, residual_ch, rng=r))
            self.skip.append(Linear(residual_ch, skip_ch, rng=r))
        self.out1 = Linear(skip_ch, skip_ch, rng=r)
        self.out2 = Linear(skip_ch, out_ch, rng=r)

    def parameters(self):
        ps = list(self.start.parameters())
        for group in (self.filters, self.gates, self.res, self.skip):
            for m in group:
                ps += list(m.parameters())
        return ps + list(self.out1.parameters()) + list(self.out2.parameters())

    def forward(self, x):
        h = self.start(Tensor._wrap(x))
        skip_total = None
        for f, g, res, skip in zip(self.filters, self.gates, self.res, self.skip):
            gated = f(h).tanh() * g(h).sigmoid()        # gated activation unit
            skip_out = skip(gated)
            skip_total = skip_out if skip_total is None else skip_total + skip_out
            h = h + res(gated)                          # residual
        out = self.out2(self.out1(skip_total.relu()).relu())
        return out


def _squash(s, axis=-1, eps=1e-8):
    """Squash a capsule vector to length in [0, 1) keeping its direction."""
    sq = (s * s).sum(axis=axis, keepdims=True)
    scale = sq / (1.0 + sq)
    return s * (scale / (sq ** 0.5 + eps))


class CapsuleLayer(Module):
    """Vector neurons routed by AGREEMENT, not by pooling (Sabour et al., 2017).

    A scalar neuron loses all pose information -- max-pooling throws away WHERE a
    feature was. A capsule is a VECTOR whose length encodes the probability a
    feature is present and whose direction encodes its pose. Each lower capsule
    predicts each higher capsule via a learned transform, and DYNAMIC ROUTING
    iteratively raises the coupling to whichever higher capsule its prediction
    agrees with (measured by dot product) -- "routing by agreement" replaces
    pooling, so a part is assigned to the whole it actually votes for. Output
    lengths are class probabilities. Input ``(batch, n_in, d_in)``.
    """

    def __init__(self, n_in, d_in, n_out, d_out, routing_iters=3, rng=None):
        super().__init__()
        r = check_random_state(rng)
        self.n_in, self.n_out = n_in, n_out
        self.d_out = d_out
        self.routing_iters = routing_iters
        # transform matrix per (input capsule, output capsule)
        self.W = Parameter(r.randn(n_out, n_in, d_out, d_in) * 0.1)

    def parameters(self):
        return [self.W]

    def forward(self, x):
        x = Tensor._wrap(x)
        B = x.shape[0]
        # predictions u_hat[j][i] = W[j,i] @ u_i  -> list of (B, d_out) tensors
        u_hat = [[x[:, i, :] @ self.W[j, i].transpose() for i in range(self.n_in)]
                 for j in range(self.n_out)]
        b = np.zeros((B, self.n_out, self.n_in))       # routing logits (numpy)
        v = None
        for it in range(self.routing_iters):
            c = _softmax_np(b, axis=1)                  # couplings sum over outputs
            outs = []
            for j in range(self.n_out):
                s = None
                for i in range(self.n_in):
                    cij = Tensor(c[:, j, i][:, None])
                    term = u_hat[j][i] * cij
                    s = term if s is None else s + term
                outs.append(_squash(s).reshape(B, 1, self.d_out))
            v = Tensor.concatenate(outs, axis=1)        # (B, n_out, d_out)
            if it < self.routing_iters - 1:             # update agreement (detached)
                for j in range(self.n_out):
                    vj = v.data[:, j, :]
                    for i in range(self.n_in):
                        b[:, j, i] += (u_hat[j][i].data * vj).sum(axis=1)
        return v

    def lengths(self, x):
        v = self.forward(x)
        return ((v * v).sum(axis=-1) + 1e-9) ** 0.5     # capsule "probabilities"


def _softmax_np(x, axis):
    x = x - x.max(axis=axis, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=axis, keepdims=True)


class SpatialTransformer(Module):
    """Let the network WARP its own input before reading it (Jaderberg, 2015).

    CNNs are only locally translation-invariant; they struggle when the object is
    rotated, scaled, or shifted. A spatial transformer inserts a small module that
    LOOKS at the input, predicts an affine transform ``theta`` (a localisation
    net), builds the corresponding sampling GRID, and bilinearly SAMPLES the input
    onto it -- actively de-rotating or centring the object so the downstream net
    sees a canonical view. The whole warp is differentiable through the bilinear
    sampler. Here ``localize`` predicts theta and ``transform`` applies the affine
    grid sample to a batch of ``(H, W)`` images.
    """

    def __init__(self, in_h, in_w, rng=None):
        super().__init__()
        r = check_random_state(rng)
        self.in_h, self.in_w = in_h, in_w
        self.loc = Linear(in_h * in_w, 6, rng=r)
        # bias the localisation to the IDENTITY transform at init
        self.loc.weight.data[...] *= 0.01
        self.loc.bias.data[...] = np.array([1., 0., 0., 0., 1., 0.])

    def parameters(self):
        return list(self.loc.parameters())

    def localize(self, images):
        flat = Tensor(np.asarray(images).reshape(len(images), -1))
        return self.loc(flat).data.reshape(-1, 2, 3)

    def transform(self, images, theta=None):
        """Affine-warp each image via a sampling grid + bilinear interpolation."""
        images = np.asarray(images, dtype=float)
        N, H, W = images.shape
        if theta is None:
            theta = self.localize(images)
        # normalised output grid in [-1, 1]
        ys, xs = np.meshgrid(np.linspace(-1, 1, H), np.linspace(-1, 1, W),
                             indexing="ij")
        grid = np.stack([xs.ravel(), ys.ravel(), np.ones(H * W)], axis=0)  # (3, HW)
        out = np.empty((N, H, W))
        for n in range(N):
            src = theta[n] @ grid                       # (2, HW) source coords in [-1,1]
            sx = (src[0] + 1) * (W - 1) / 2             # to pixel coords
            sy = (src[1] + 1) * (H - 1) / 2
            out[n] = _bilinear_sample(images[n], sx, sy).reshape(H, W)
        return out


def _bilinear_sample(img, sx, sy):
    H, W = img.shape
    x0 = np.floor(sx).astype(int); y0 = np.floor(sy).astype(int)
    x1, y1 = x0 + 1, y0 + 1
    wx = sx - x0; wy = sy - y0
    def gv(yy, xx):
        inb = (yy >= 0) & (yy < H) & (xx >= 0) & (xx < W)
        v = np.zeros_like(sx)
        yy = np.clip(yy, 0, H - 1); xx = np.clip(xx, 0, W - 1)
        v[inb] = img[yy[inb], xx[inb]]
        return v
    return (gv(y0, x0) * (1 - wx) * (1 - wy) + gv(y0, x1) * wx * (1 - wy)
            + gv(y1, x0) * (1 - wx) * wy + gv(y1, x1) * wx * wy)


class EchoStateNetwork(BaseEstimator):
    """A random recurrent RESERVOIR with only the readout trained (Jaeger, 2001).

    Training recurrent nets is hard (backprop through time, vanishing gradients).
    Reservoir computing sidesteps it entirely: fix a large RANDOM recurrent network
    -- the "reservoir" -- whose rich nonlinear dynamics echo the input history, and
    train ONLY a linear readout from its state by ridge regression. No
    backpropagation, closed-form fit, yet it models complex temporal systems. The
    one thing that must be tuned is the reservoir's SPECTRAL RADIUS (< 1 for the
    "echo state property": past inputs must eventually fade). Fits ``X: (T, in)``
    to ``y: (T, out)`` and forecasts.
    """

    def __init__(self, n_reservoir=200, spectral_radius=0.95, sparsity=0.1,
                 leak=1.0, ridge=1e-6, washout=10, random_state=None):
        self.n_reservoir = n_reservoir
        self.spectral_radius = spectral_radius
        self.sparsity = sparsity
        self.leak = leak
        self.ridge = ridge
        self.washout = washout
        self.random_state = random_state

    def _init_reservoir(self, n_in, rng):
        self.W_in_ = rng.uniform(-1, 1, (self.n_reservoir, n_in))
        W = rng.uniform(-1, 1, (self.n_reservoir, self.n_reservoir))
        mask = rng.rand(*W.shape) > self.sparsity
        W[mask] = 0.0
        radius = np.max(np.abs(np.linalg.eigvals(W)))
        self.W_res_ = W * (self.spectral_radius / (radius + 1e-12))

    def _run(self, X):
        T = len(X)
        state = np.zeros(self.n_reservoir)
        states = np.empty((T, self.n_reservoir))
        for t in range(T):
            pre = self.W_in_ @ X[t] + self.W_res_ @ state
            state = (1 - self.leak) * state + self.leak * np.tanh(pre)
            states[t] = state
        return states

    def fit(self, X, y):
        X = np.atleast_2d(np.asarray(X, float))
        if X.shape[0] == 1:
            X = X.T
        y = np.asarray(y, float)
        Y = y.reshape(len(y), -1)
        rng = check_random_state(self.random_state)
        self._init_reservoir(X.shape[1], rng)
        states = self._run(X)
        S = np.hstack([states, np.ones((len(states), 1))])[self.washout:]
        Yt = Y[self.washout:]
        A = S.T @ S + self.ridge * np.eye(S.shape[1])   # ridge-regressed readout
        self.W_out_ = np.linalg.solve(A, S.T @ Yt)
        self._last_state = states[-1]
        return self

    def predict(self, X):
        X = np.atleast_2d(np.asarray(X, float))
        if X.shape[0] == 1:
            X = X.T
        states = self._run(X)
        S = np.hstack([states, np.ones((len(states), 1))])
        out = S @ self.W_out_
        return out.ravel() if out.shape[1] == 1 else out


class PointerNetwork(Module):
    """An output distribution over the INPUT positions (Vinyals et al., 2015).

    Ordinary seq2seq emits tokens from a fixed vocabulary, so it cannot output "the
    3rd input element" when the input length varies (sorting, routing, convex
    hull). A pointer network's decoder, at each step, uses attention to POINT: it
    scores every encoder position and the softmax over those scores IS the output
    -- a pointer into the input. This makes the output vocabulary the input itself,
    of whatever length. Trained here to output an ordering (e.g. argsort) of a
    variable-length numeric input.
    """

    def __init__(self, hidden=32, rng=None):
        super().__init__()
        r = check_random_state(rng)
        self.hidden = hidden
        self.enc = Linear(1, hidden, rng=r)
        self.dec = Linear(hidden, hidden, rng=r)
        self.W1 = Linear(hidden, hidden, rng=r)
        self.W2 = Linear(hidden, hidden, rng=r)
        self.v = Parameter(r.randn(hidden) * 0.1)

    def parameters(self):
        return (list(self.enc.parameters()) + list(self.dec.parameters())
                + list(self.W1.parameters()) + list(self.W2.parameters())
                + [self.v])

    def _pointer_logits(self, enc, dec_state):
        # score_i = v . tanh(W1 enc_i + W2 dec_state)
        scored = (self.W1(enc) + self.W2(dec_state)).tanh()
        return (scored @ self.v.reshape(-1, 1)).reshape(1, -1)

    def forward(self, seq, targets=None):
        """Per-step pointer logits. With ``targets`` (teacher forcing) the true
        previous element is fed to the decoder; otherwise it decodes greedily."""
        x = Tensor(np.asarray(seq, float).reshape(-1, 1))
        enc = self.enc(x).tanh()                        # (n, hidden)
        n = enc.shape[0]
        prev = Tensor(np.zeros((1, self.hidden)))       # start token
        used = np.zeros(n, dtype=bool)
        logits = []
        for t in range(n):
            dec_state = self.dec(prev).tanh()
            lg = self._pointer_logits(enc, dec_state)
            mask = np.where(used, -1e9, 0.0).reshape(1, -1)   # forbid repeats
            lg = lg + Tensor(mask)
            logits.append(lg)
            if targets is not None:
                chosen = int(targets[t])
            else:
                chosen = int(lg.data.argmax())
            used[chosen] = True
            prev = enc[chosen:chosen + 1]               # feed the pointed element
        return Tensor.concatenate(logits, axis=0)       # (n, n)

    def predict_order(self, seq):
        return self.forward(seq).data.argmax(axis=1)


__all__ = ["WaveNet", "CapsuleLayer", "SpatialTransformer", "EchoStateNetwork",
           "PointerNetwork"]
