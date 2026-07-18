"""Neural architectures v5: patch transformers, external memory, deep flows,
fast diffusion sampling, and relative attention.

Five architectures that each extend a modern idea. The Vision Transformer applies
a plain transformer to image PATCHES. The Neural Turing Machine gives a net an
addressable external MEMORY. Glow is a normalizing flow with invertible linear
mixing. DDIM samples a trained diffusion model DETERMINISTICALLY in a few steps.
Relative-position attention scores by the DISTANCE between tokens rather than their
absolute positions.
"""
import numpy as np

from ..autograd import Tensor
from ..autograd import functional as F
from ..utils import check_random_state
from .module import Module, Parameter
from .layers import Linear
from .attention import TransformerEncoderLayer


class VisionTransformer(Module):
    """A plain transformer over image PATCHES (Dosovitskiy et al., 2021).

    Convolutions bake in locality and translation equivariance. The Vision
    Transformer throws that away: cut the image into a grid of patches, linearly
    embed each patch as a "token", prepend a learnable CLASS token, add positional
    embeddings, and run an ordinary transformer encoder -- then classify from the
    class token's output. With enough data it matches or beats CNNs, showing the
    convolutional prior is helpful but not necessary. Input images are
    ``(batch, H, W)`` (single channel); ``patch_size`` must divide ``H`` and ``W``.
    """

    def __init__(self, img_size, patch_size, n_classes, embed_dim=32, depth=2,
                 n_heads=4, rng=None):
        super().__init__()
        r = check_random_state(rng)
        self.p = patch_size
        self.grid = img_size // patch_size
        self.n_patches = self.grid * self.grid
        self.patch_embed = Linear(patch_size * patch_size, embed_dim, rng=r)
        self.cls_token = Parameter(r.randn(1, embed_dim) * 0.02)
        self.pos_embed = Parameter(r.randn(self.n_patches + 1, embed_dim) * 0.02)
        self.layers = [TransformerEncoderLayer(embed_dim, n_heads, rng=r)
                       for _ in range(depth)]
        self.head = Linear(embed_dim, n_classes, rng=r)

    def parameters(self):
        ps = list(self.patch_embed.parameters()) + [self.cls_token, self.pos_embed]
        for l in self.layers:
            ps += list(l.parameters())
        return ps + list(self.head.parameters())

    def _patchify(self, images):
        images = np.asarray(images, float)
        B = len(images); p, g = self.p, self.grid
        out = np.empty((B, self.n_patches, p * p))
        for i in range(g):
            for j in range(g):
                patch = images[:, i * p:(i + 1) * p, j * p:(j + 1) * p]
                out[:, i * g + j] = patch.reshape(B, -1)
        return out

    def forward(self, images):
        B = len(images)
        tokens = self.patch_embed(Tensor(self._patchify(images)))   # (B, n, d)
        cls = self.cls_token.reshape(1, 1, -1)
        cls = Tensor(np.tile(cls.data, (B, 1, 1)))
        x = Tensor.concatenate([cls, tokens], axis=1) + self.pos_embed
        for layer in self.layers:
            x = layer(x)
        return self.head(x[:, 0, :])                                 # class token


class NeuralTuringMachine(Module):
    """A controller with an addressable external MEMORY (Graves et al., 2014).

    An RNN's memory is its fixed-size hidden state -- it cannot store and recall an
    arbitrary sequence. The NTM couples a neural controller to an external MEMORY
    matrix it reads and writes through differentiable attention: it emits a KEY,
    addresses memory by CONTENT (cosine similarity → softmax), reads a weighted
    combination, and writes with an erase/add rule. Because addressing is soft, the
    whole thing trains by backprop, and it learns algorithmic tasks (copy, sort)
    that defeat a plain RNN. This is a content-addressed NTM with a feed-forward
    controller, trained here on the copy task.
    """

    def __init__(self, n_slots=16, slot_dim=8, input_dim=8, controller_dim=32,
                 rng=None):
        super().__init__()
        r = check_random_state(rng)
        self.n_slots, self.slot_dim = n_slots, slot_dim
        self.controller = Linear(input_dim + slot_dim, controller_dim, rng=r)
        self.key = Linear(controller_dim, slot_dim, rng=r)
        self.erase = Linear(controller_dim, slot_dim, rng=r)
        self.add = Linear(controller_dim, slot_dim, rng=r)
        self.out = Linear(controller_dim + slot_dim, input_dim, rng=r)

    def parameters(self):
        ps = []
        for m in (self.controller, self.key, self.erase, self.add, self.out):
            ps += list(m.parameters())
        return ps

    def _address(self, memory, key):
        # content addressing: cosine similarity between the key and every slot,
        # PER EXAMPLE (memory is (B, n_slots, slot_dim), key is (B, slot_dim))
        k = key / ((key * key).sum(axis=1, keepdims=True) ** 0.5 + 1e-8)
        m = memory / ((memory * memory).sum(axis=2, keepdims=True) ** 0.5 + 1e-8)
        sims = (k.reshape(-1, 1, self.slot_dim) @ m.transpose(0, 2, 1)).reshape(
            -1, self.n_slots)                             # (B, n_slots)
        return F.softmax(sims, axis=1)

    def forward(self, sequence):
        """Run the controller over an input sequence, returning the output sequence.

        Each example keeps its OWN memory matrix, so a sequence can store its
        contents and read them back -- the whole point of the copy task."""
        B = sequence[0].shape[0]
        memory = Tensor(np.full((B, self.n_slots, self.slot_dim), 1e-3))
        read = Tensor(np.zeros((B, self.slot_dim)))
        outputs = []
        for x in sequence:
            x = Tensor._wrap(x)
            h = self.controller(Tensor.concatenate([x, read], axis=1)).tanh()
            w = self._address(memory, self.key(h))        # (B, n_slots)
            erase = self.erase(h).sigmoid(); add = self.add(h).tanh()
            wcol = w.reshape(B, self.n_slots, 1)          # per-slot write weight
            # erase then add, as per-example outer products (B, n_slots, slot_dim)
            memory = memory * (1.0 - wcol * erase.reshape(B, 1, self.slot_dim)) \
                + wcol * add.reshape(B, 1, self.slot_dim)
            read = (w.reshape(B, 1, self.n_slots) @ memory).reshape(B, self.slot_dim)
            outputs.append(self.out(Tensor.concatenate([h, read], axis=1)))
        return outputs


class Glow(Module):
    """A normalizing flow with invertible LINEAR mixing (Kingma & Dhariwal, 2018).

    RealNVP's coupling layers only ever transform half the dimensions at a time and
    permute by a fixed rule. Glow replaces the fixed permutation with a LEARNED
    invertible linear map (a "1x1 convolution"), so the dimensions get mixed
    optimally between couplings, and adds ``ActNorm`` (a data-dependent affine
    normalisation) for stable training. Each layer contributes an exact log-
    determinant, so the model has a tractable likelihood and both samples and
    scores densities. Implemented for vector data (the 1x1 conv is a dense
    invertible matrix).
    """

    def __init__(self, dim, n_flows=4, hidden=32, rng=None):
        super().__init__()
        r = check_random_state(rng)
        self.dim = dim
        self.flows = []
        for _ in range(n_flows):
            W = np.linalg.qr(r.randn(dim, dim))[0]        # start orthogonal (invertible)
            self.flows.append({
                "log_s": Parameter(np.zeros(dim)), "b": Parameter(np.zeros(dim)),
                "W": Parameter(W),
                "nn": [Linear(dim // 2, hidden, rng=r), Linear(hidden, dim - dim // 2, rng=r),
                       Linear(dim // 2, hidden, rng=r), Linear(hidden, dim - dim // 2, rng=r)],
            })

    def parameters(self):
        ps = []
        for f in self.flows:
            ps += [f["log_s"], f["b"], f["W"]]
            for m in f["nn"]:
                ps += list(m.parameters())
        return ps

    def forward(self, x):
        """Map data -> latent, accumulating the log-determinant of the Jacobian."""
        x = Tensor._wrap(x)
        logdet = Tensor(np.zeros(len(x.data)))
        d = self.dim // 2
        for f in self.flows:
            # actnorm
            x = x * f["log_s"].exp() + f["b"]
            logdet = logdet + f["log_s"].sum()
            # invertible linear (1x1 conv)
            x = x @ f["W"]
            sign, ld = np.linalg.slogdet(f["W"].data)
            logdet = logdet + ld
            # affine coupling
            xa, xb = x[:, :d], x[:, d:]
            scale = self.__nn(f["nn"][0], f["nn"][1], xa).tanh()
            shift = self.__nn(f["nn"][2], f["nn"][3], xa)
            xb = xb * scale.exp() + shift
            logdet = logdet + scale.sum(axis=1)
            x = Tensor.concatenate([xa, xb], axis=1)
        return x, logdet

    @staticmethod
    def __nn(l1, l2, x):
        return l2(l1(x).relu())

    def inverse(self, z):
        """Map latent -> data (exact inverse of ``forward``), for sampling."""
        z = np.asarray(Tensor._wrap(z).data, float)
        d = self.dim // 2
        for f in reversed(self.flows):
            za, zb = z[:, :d], z[:, d:]
            scale = np.tanh(self.__nn(f["nn"][0], f["nn"][1], Tensor(za)).data)
            shift = self.__nn(f["nn"][2], f["nn"][3], Tensor(za)).data
            zb = (zb - shift) * np.exp(-scale)            # invert coupling
            z = np.concatenate([za, zb], axis=1)
            z = z @ np.linalg.inv(f["W"].data)            # invert 1x1 conv
            z = (z - f["b"].data) * np.exp(-f["log_s"].data)   # invert actnorm
        return z

    def sample(self, n, rng=None):
        rng = check_random_state(rng)
        return self.inverse(rng.normal(size=(n, self.dim)))

    def log_prob(self, x):
        z, logdet = self.forward(x)
        # standard-normal base density
        logpz = -0.5 * ((z * z).sum(axis=1) + self.dim * np.log(2 * np.pi))
        return logpz + logdet

    def nll(self, x):
        return -self.log_prob(x).mean()


def ddim_sample(ddpm, n, n_steps=20, eta=0.0, rng=None):
    """Sample a trained diffusion model DETERMINISTICALLY, in few steps
    (Song et al., 2021).

    DDPM sampling is stochastic and needs all ``T`` denoising steps. DDIM reuses the
    SAME trained noise predictor but follows a non-Markovian, (with ``eta=0``)
    fully DETERMINISTIC trajectory: at each step it predicts ``x_0`` from the current
    noisy sample and jumps directly toward it along a chosen subsequence of
    timesteps. This lets you sample in, say, 20 steps instead of 1000, and -- being
    deterministic -- gives a meaningful latent-to-image map. ``eta`` interpolates
    back toward stochastic DDPM.
    """
    rng = check_random_state(rng)
    ab = ddpm.alpha_bars
    d = ddpm.model.net  # noise predictor MLP  (via ddpm.model)
    x = rng.normal(size=(n, ddpm.n_features))
    ts = np.linspace(ddpm.T - 1, 0, n_steps).astype(int)
    for i, t in enumerate(ts):
        eps = ddpm.model(Tensor(x), np.full(n, t)).data
        x0 = (x - np.sqrt(1 - ab[t]) * eps) / np.sqrt(ab[t])   # predict clean sample
        if i == len(ts) - 1:
            x = x0
            break
        t_prev = ts[i + 1]
        sigma = eta * np.sqrt((1 - ab[t_prev]) / (1 - ab[t]) *
                              (1 - ab[t] / ab[t_prev]))
        noise = rng.normal(size=x.shape) if eta > 0 else 0.0
        x = (np.sqrt(ab[t_prev]) * x0
             + np.sqrt(np.maximum(1 - ab[t_prev] - sigma ** 2, 0)) * eps
             + sigma * noise)
    return x


class RelativePositionAttention(Module):
    """Attention scored by the DISTANCE between tokens (Shaw 2018; Dai 2019).

    Absolute positional encodings tie a token's meaning to WHERE it sits, so a
    pattern learned at position 5 does not transfer to position 50. Relative
    attention instead adds a learned bias that depends only on the OFFSET ``i - j``
    between query and key, so the same relative pattern is recognised anywhere in
    the sequence (and it extrapolates to longer sequences). This is the mechanism
    behind Transformer-XL. Single-head here for clarity; input ``(batch, seq, dim)``.
    """

    def __init__(self, dim, max_len=64, rng=None):
        super().__init__()
        r = check_random_state(rng)
        self.dim = dim
        self.q = Linear(dim, dim, rng=r)
        self.k = Linear(dim, dim, rng=r)
        self.v = Linear(dim, dim, rng=r)
        self.max_len = max_len
        # a learned bias per relative offset in [-(L-1), L-1]
        self.rel_bias = Parameter(r.randn(2 * max_len - 1) * 0.02)

    def parameters(self):
        return (list(self.q.parameters()) + list(self.k.parameters())
                + list(self.v.parameters()) + [self.rel_bias])

    def _bias_matrix(self, n):
        idx = np.arange(n)
        offset = idx[:, None] - idx[None, :] + (self.max_len - 1)   # map to [0, 2L-2]
        return offset

    def forward(self, x):
        x = Tensor._wrap(x)
        B, n, d = x.shape
        Q, K, V = self.q(x), self.k(x), self.v(x)
        scores = (Q @ K.transpose(0, 2, 1)) * (1.0 / np.sqrt(d))    # (B, n, n)
        bias = self.rel_bias[self._bias_matrix(n).ravel()].reshape(1, n, n)
        attn = F.softmax(scores + bias, axis=2)                     # relative bias
        return attn @ V


__all__ = ["VisionTransformer", "NeuralTuringMachine", "Glow", "ddim_sample",
           "RelativePositionAttention"]
