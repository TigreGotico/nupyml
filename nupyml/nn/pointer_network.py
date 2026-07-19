"""An output distribution over the INPUT positions (Vinyals et al., 2015)."""
import numpy as np
from ..autograd import Tensor
from ..utils import check_random_state
from .module import Module, Parameter
from .layers import Linear


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


__all__ = ["PointerNetwork"]
