"""Read a sequence BOTH ways and combine (Schuster & Paliwal, 1997)."""
import numpy as np
from ..autograd import Tensor
from ..utils import check_random_state
from .module import Module, Parameter
from .recurrent import LSTM, GRU, RNN


class Bidirectional(Module):
    """Read a sequence BOTH ways and combine (Schuster & Paliwal, 1997).

    A plain LSTM only knows the past when it labels a token -- but for tagging,
    named-entity recognition or any task where later words disambiguate earlier
    ones, the FUTURE matters too. A bidirectional wrapper runs one recurrent cell
    left-to-right and an independent copy right-to-left, then concatenates their
    per-step outputs, so every position's representation summarises the whole
    sequence around it. This is the ``BiLSTM`` (or BiGRU) at the heart of most
    pre-transformer sequence models. Wraps any cell; output width is ``2 *
    hidden_size``.
    """

    def __init__(self, input_size, hidden_size, cell="lstm", rng=None):
        super().__init__()
        r = check_random_state(rng)
        cls = {"lstm": LSTM, "gru": GRU, "rnn": RNN}[cell]
        self.fwd = cls(input_size, hidden_size, rng=r)
        self.bwd = cls(input_size, hidden_size, rng=r)
        self.hidden_size = hidden_size

    def parameters(self):
        return list(self.fwd.parameters()) + list(self.bwd.parameters())

    @staticmethod
    def _out(y):
        return y[0] if isinstance(y, tuple) else y

    def forward(self, x):
        x = Tensor._wrap(x)
        seq = x.shape[1]
        rev = np.arange(seq - 1, -1, -1)
        out_f = self._out(self.fwd(x))                   # left-to-right
        out_b = self._out(self.bwd(x[:, rev, :]))        # right-to-left on reversed
        out_b = out_b[:, rev, :]                          # un-reverse to align steps
        return Tensor.concatenate([out_f, out_b], axis=2)


__all__ = ["Bidirectional"]
