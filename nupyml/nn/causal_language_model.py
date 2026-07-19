"""A decoder-only (GPT-style) language model over token ids."""
import numpy as np
from .module import Module
from .layers import Linear, LayerNorm, Embedding, Dropout, ReLU, GELU
from .attention import MultiHeadAttention, causal_mask
from .transformer_decoder_layer import TransformerDecoderLayer


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


__all__ = ["CausalLanguageModel"]
