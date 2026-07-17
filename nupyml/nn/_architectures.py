"""Higher-level neural architectures assembled from the nn building blocks.

The base package has autoencoders, VAE, GAN, RealNVP, DDPM, RBM, MADE, SimCLR,
attention, and a transformer ENCODER. These fill the remaining gaps: residual
blocks, a transformer DECODER + causal language model, a mixture density network,
and a Siamese network.
"""
import numpy as np

from ..autograd import Tensor
from ..autograd import functional as F
from .module import Module
from .layers import Linear, LayerNorm, Embedding, Dropout, ReLU, GELU
from .attention import MultiHeadAttention, causal_mask


class ResidualBlock(Module):
    """A pre-norm residual MLP block: ``x + FF(norm(x))``.

    THE POINT OF THE SKIP
    ---------------------
    Stacking many layers makes gradients vanish and the identity map hard to
    learn. A residual block computes a small CORRECTION and ADDS it to the input,
    so the default behaviour is to pass the input through unchanged and each block
    only has to learn the delta. That is what let networks go from tens of layers
    to hundreds -- gradients flow straight down the skip connections. Pre-norm
    (LayerNorm before the sublayer) is the arrangement that trains deep stacks
    stably.
    """

    def __init__(self, dim, hidden=None, rng=None):
        super().__init__()
        hidden = hidden or 4 * dim
        self.norm = LayerNorm(dim)
        self.fc1 = Linear(dim, hidden, rng=rng)
        self.act = GELU()
        self.fc2 = Linear(hidden, dim, rng=rng)

    def forward(self, x):
        h = self.fc2(self.act(self.fc1(self.norm(x))))
        return x + h                                  # the residual connection


class TransformerDecoderLayer(Module):
    """One decoder-only (GPT-style) block: causal self-attention + feed-forward.

    Unlike the encoder layer, the self-attention is MASKED so position ``t`` can
    only attend to positions ``<= t`` -- the model must predict each token from
    the past alone, never peeking ahead. Pre-norm residual around both the
    attention and the feed-forward sublayers.
    """

    def __init__(self, embed_dim, num_heads, ff_dim=None, rng=None):
        super().__init__()
        ff_dim = ff_dim or 4 * embed_dim
        self.norm1 = LayerNorm(embed_dim)
        self.attn = MultiHeadAttention(embed_dim, num_heads, rng=rng)
        self.norm2 = LayerNorm(embed_dim)
        self.fc1 = Linear(embed_dim, ff_dim, rng=rng)
        self.act = GELU()
        self.fc2 = Linear(ff_dim, embed_dim, rng=rng)

    def forward(self, x, mask=None):
        h = self.norm1(x)
        x = x + self.attn(h, h, h, mask=mask)         # masked self-attention
        h = self.norm2(x)
        return x + self.fc2(self.act(self.fc1(h)))    # feed-forward


class CausalLanguageModel(Module):
    """A decoder-only (GPT-style) language model over token ids.

    Token embedding + learned positional embedding, a stack of causal decoder
    layers, a final norm, and a linear projection to vocabulary logits. Trained to
    predict the NEXT token at every position (teacher forcing); the causal mask is
    what makes one forward pass supply a next-token target for every position at
    once. ``generate`` samples greedily, feeding predictions back in.
    """

    def __init__(self, vocab_size, embed_dim=64, num_heads=4, num_layers=2,
                 max_len=128, rng=None):
        super().__init__()
        self.tok = Embedding(vocab_size, embed_dim, rng=rng)
        self.pos = Embedding(max_len, embed_dim, rng=rng)
        self.layers = [TransformerDecoderLayer(embed_dim, num_heads, rng=rng)
                       for _ in range(num_layers)]
        self.norm = LayerNorm(embed_dim)
        self.head = Linear(embed_dim, vocab_size, rng=rng)
        self.vocab_size = vocab_size

    def parameters(self):
        ps = list(self.tok.parameters()) + list(self.pos.parameters())
        for l in self.layers:
            ps += list(l.parameters())
        return ps + list(self.norm.parameters()) + list(self.head.parameters())

    def forward(self, tokens):
        tokens = np.asarray(tokens)
        if tokens.ndim == 1:
            tokens = tokens[None, :]
        T = tokens.shape[1]
        pos_ids = np.tile(np.arange(T), (tokens.shape[0], 1))
        x = self.tok(tokens) + self.pos(pos_ids)
        mask = causal_mask(T)
        for layer in self.layers:
            x = layer(x, mask=mask)
        return self.head(self.norm(x))                # (B, T, vocab)

    def generate(self, prompt, n_new, max_len=128):
        seq = list(np.asarray(prompt).ravel())
        for _ in range(n_new):
            logits = self.forward(np.array(seq[-max_len:]))
            nxt = int(logits.data[0, -1].argmax())    # greedy next token
            seq.append(nxt)
        return np.array(seq)


class MixtureDensityNetwork(Module):
    """Predict a DISTRIBUTION over the target, not a point (Bishop, 1994).

    THE PROBLEM WITH POINT REGRESSION
    ---------------------------------
    Squared-error regression predicts the conditional MEAN -- which is disastrous
    when the target is MULTIMODAL (given x, y could be one of two valid values):
    the mean sits between the modes, at a value that never actually occurs (an
    inverse-kinematics arm pointing nowhere). An MDN outputs the parameters of a
    MIXTURE of Gaussians -- weights, means, and variances -- so it can say "y is
    near A OR near B", and it is trained by maximising the mixture likelihood.

    ``forward`` returns ``(pi, mu, sigma)``; ``nll`` is the training loss;
    ``predict`` returns the most-likely component's mean.
    """

    def __init__(self, in_dim, n_components=3, hidden=32, rng=None):
        super().__init__()
        self.k = n_components
        self.fc = Linear(in_dim, hidden, rng=rng)
        self.act = ReLU()
        self.pi = Linear(hidden, n_components, rng=rng)
        self.mu = Linear(hidden, n_components, rng=rng)
        self.logsigma = Linear(hidden, n_components, rng=rng)

    def parameters(self):
        return (list(self.fc.parameters()) + list(self.pi.parameters())
                + list(self.mu.parameters()) + list(self.logsigma.parameters()))

    def forward(self, x):
        h = self.act(self.fc(Tensor._wrap(x) if not isinstance(x, Tensor) else x))
        pi = F.softmax(self.pi(h), axis=-1)
        mu = self.mu(h)
        sigma = self.logsigma(h).exp() + 1e-3
        return pi, mu, sigma

    def nll(self, x, y):
        pi, mu, sigma = self.forward(x)
        y = np.asarray(y, float).reshape(-1, 1)
        # log N(y | mu_k, sigma_k) for every component
        z = (Tensor(y) - mu) / sigma
        log_comp = (-0.5 * z * z - sigma.log()
                    - 0.5 * np.log(2 * np.pi) + (pi + 1e-12).log())
        # stable log-sum-exp over components (subtract the row max as a constant)
        m = log_comp.data.max(axis=1, keepdims=True)
        lse = (log_comp - Tensor(m)).exp().sum(axis=1).log() + Tensor(m.ravel())
        return (-lse).mean()

    def predict(self, x):
        pi, mu, sigma = self.forward(x)
        best = pi.data.argmax(axis=1)                 # the dominant component's mean
        return mu.data[np.arange(len(best)), best]


class SiameseNetwork(Module):
    """Twin encoders with SHARED weights, for learning a similarity metric.

    A Siamese network runs two inputs through the SAME encoder and compares their
    embeddings; trained with a contrastive or triplet loss (see ``metric_losses``)
    it learns an embedding where "same" pairs are close and "different" pairs far.
    Weight sharing is the essence -- both inputs are mapped by one function, so the
    geometry is consistent -- and it is the standard approach for verification and
    few-shot tasks where the classes at test time were never seen in training.
    """

    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder

    def parameters(self):
        return list(self.encoder.parameters())

    def encode(self, x):
        return self.encoder(Tensor._wrap(x) if not isinstance(x, Tensor) else x)

    def forward(self, x1, x2):
        return self.encode(x1), self.encode(x2)       # shared-weight twin pass


__all__ = ["ResidualBlock", "TransformerDecoderLayer", "CausalLanguageModel",
           "MixtureDensityNetwork", "SiameseNetwork"]
