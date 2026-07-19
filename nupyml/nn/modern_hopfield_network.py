"""Associative memory that IS attention (Ramsauer et al., 2020)."""
import numpy as np
from .module import Module, Parameter


class ModernHopfieldNetwork(Module):
    """Associative memory that IS attention (Ramsauer et al., 2020).

    The classic Hopfield network stores patterns and retrieves the nearest one from
    a noisy cue, but it saturates at ~0.14N patterns. The MODERN Hopfield network
    uses an exponential energy, and its retrieval update turns out to be EXACTLY the
    transformer's attention: ``softmax(beta * cue . patterns) . patterns``. This
    stores exponentially many patterns and converges to the right one in a SINGLE
    step, which is why "attention is retrieval from an associative memory" is more
    than a slogan. Here it stores a set of patterns and cleans up a corrupted query.
    """

    def __init__(self, patterns, beta=8.0):
        super().__init__()
        self.patterns = np.asarray(patterns, float)      # (n_patterns, dim)
        self.beta = beta

    def retrieve(self, query, n_steps=1):
        q = np.asarray(query, float)
        single = q.ndim == 1
        q = np.atleast_2d(q)
        # scale by dimension so beta is comparable across problem sizes (as in the
        # paper), and compare on a common footing regardless of pattern norm
        scale = self.beta / np.sqrt(self.patterns.shape[1])
        pnorm2 = (self.patterns ** 2).sum(axis=1)        # ||p||^2 makes it a distance
        for _ in range(n_steps):
            scores = scale * (q @ self.patterns.T - 0.5 * pnorm2)
            scores -= scores.max(axis=1, keepdims=True)
            attn = np.exp(scores); attn /= attn.sum(axis=1, keepdims=True)
            q = attn @ self.patterns                     # weighted retrieval
        return q[0] if single else q


__all__ = ["ModernHopfieldNetwork"]
