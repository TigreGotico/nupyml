"""A controller with an addressable external MEMORY (Graves et al., 2014)."""
import numpy as np
from ..autograd import Tensor
from ..autograd import functional as F
from ..utils import check_random_state
from .module import Module, Parameter
from .layers import Linear


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


__all__ = ["NeuralTuringMachine"]
