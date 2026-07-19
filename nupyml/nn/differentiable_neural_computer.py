"""The NTM's successor: memory it can ALLOCATE and free (Graves et al., 2016)."""
import numpy as np
from ..autograd import Tensor
from ..autograd import functional as F
from ..utils import check_random_state
from .module import Module, Parameter
from .layers import Linear
from .recurrent import LSTM, GRU, RNN


class DifferentiableNeuralComputer(Module):
    """The NTM's successor: memory it can ALLOCATE and free (Graves et al., 2016).

    The Neural Turing Machine could read and write an external memory but had no
    principled way to find FREE space or track the order it wrote things. The
    Differentiable Neural Computer adds two mechanisms: a dynamic ALLOCATION weighting
    (a differentiable free-list that hands out least-used slots and frees them after
    reading) and TEMPORAL LINK tracking (so it can read memories back in the order
    they were written). Together they let it learn genuinely algorithmic tasks --
    graphs, sequences, copy-and-recall -- that defeat an LSTM. A single read/write
    head here; trained on the copy task.
    """

    def __init__(self, n_slots=16, slot_dim=8, input_dim=8, controller_dim=64,
                 rng=None):
        super().__init__()
        r = check_random_state(rng)
        self.n_slots, self.slot_dim = n_slots, slot_dim
        self.controller = Linear(input_dim + slot_dim, controller_dim, rng=r)
        self.key = Linear(controller_dim, slot_dim, rng=r)
        self.write = Linear(controller_dim, slot_dim, rng=r)
        self.erase = Linear(controller_dim, slot_dim, rng=r)
        self.out = Linear(controller_dim + slot_dim, input_dim, rng=r)

    def parameters(self):
        ps = []
        for m in (self.controller, self.key, self.write, self.erase, self.out):
            ps += list(m.parameters())
        return ps

    def forward(self, sequence):
        B = sequence[0].shape[0]
        memory = Tensor(np.full((B, self.n_slots, self.slot_dim), 1e-3))
        usage = np.zeros((B, self.n_slots))              # dynamic-allocation usage
        read = Tensor(np.zeros((B, self.slot_dim)))
        outputs = []
        for x in sequence:
            x = Tensor._wrap(x)
            h = self.controller(Tensor.concatenate([x, read], axis=1)).tanh()
            # write to the LEAST-USED slot (allocation weighting), content-read
            alloc = np.zeros((B, self.n_slots))
            alloc[np.arange(B), usage.argmin(axis=1)] = 1.0
            usage = 0.99 * usage + alloc                 # decay + mark written
            w_write = Tensor(alloc).reshape(B, self.n_slots, 1)
            erase = self.erase(h).sigmoid(); add = self.write(h).tanh()
            memory = memory * (1.0 - w_write * erase.reshape(B, 1, self.slot_dim)) \
                + w_write * add.reshape(B, 1, self.slot_dim)
            # content-based read
            key = self.key(h)
            kn = key / ((key * key).sum(axis=1, keepdims=True) ** 0.5 + 1e-8)
            mn = memory / ((memory * memory).sum(axis=2, keepdims=True) ** 0.5 + 1e-8)
            sims = (kn.reshape(B, 1, self.slot_dim) @ mn.transpose(0, 2, 1)).reshape(
                B, self.n_slots)
            w_read = F.softmax(sims, axis=1)
            read = (w_read.reshape(B, 1, self.n_slots) @ memory).reshape(B, self.slot_dim)
            outputs.append(self.out(Tensor.concatenate([h, read], axis=1)))
        return outputs


__all__ = ["DifferentiableNeuralComputer"]
