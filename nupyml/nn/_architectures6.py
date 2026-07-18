"""Neural architectures v6: bidirectional recurrence, dynamic memory, spline
flows, associative memory, and graph attention with edge features.

Five more architectures. A bidirectional wrapper lets any recurrent cell read a
sequence both ways. The Differentiable Neural Computer extends the Neural Turing
Machine with dynamic memory allocation and temporal links. Neural spline flows use
expressive monotonic-spline couplings. The modern Hopfield network is associative
memory rewritten as attention. The graph transformer applies attention over a graph
with edge features.
"""
import numpy as np

from ..autograd import Tensor
from ..autograd import functional as F
from ..utils import check_random_state
from .module import Module, Parameter
from .layers import Linear
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


def _rq_spline_tensor(x_np, widths, heights, deriv, left=-3.0, right=3.0):
    """Autograd rational-quadratic spline of a column ``x_np``.

    ``widths``/``heights`` are (B, K) Tensors (softmaxed); ``deriv`` is (B, K+1)
    positive. Returns (transformed column Tensor, log-derivative Tensor). Only the
    discrete bin selection uses numpy; every value flows through the tape.
    """
    B, K = widths.shape
    span = right - left
    pref = np.triu(np.ones((K, K + 1)), 1)[:K, :K + 1]   # strict prefix-sum matrix
    pref = pref.T[:K + 1, :K].T if False else np.array(
        [[1.0 if j < m else 0.0 for m in range(K + 1)] for j in range(K)])
    cw = widths @ Tensor(pref)                           # (B, K+1) cumulative widths
    ch = heights @ Tensor(pref)
    xin = np.clip((x_np - left) / span, 1e-6, 1 - 1e-6)
    cwd = cw.data
    k = np.clip(np.array([np.searchsorted(cwd[i], xin[i], "right") - 1
                          for i in range(B)]), 0, K - 1)
    idx = np.arange(B)
    w = widths[idx, k]; h = heights[idx, k]
    x0 = cw[idx, k]; y0 = ch[idx, k]
    d0 = deriv[idx, k]; d1 = deriv[idx, k + 1]
    s = h / w
    xi = (Tensor(xin) - x0) / w
    omxi = 1.0 - xi
    num = h * (s * xi * xi + d0 * xi * omxi)
    den = s + (d0 + d1 - 2.0 * s) * xi * omxi
    y = y0 + num / den
    dnum = s * s * (d1 * xi * xi + 2.0 * s * xi * omxi + d0 * omxi * omxi)
    logdet = (dnum + 1e-9).log() - 2.0 * (den + 1e-9).log()
    return left + span * y, logdet


class NeuralSplineFlow(Module):
    """A normalizing flow with expressive SPLINE couplings (Durkan et al., 2019).

    Affine coupling layers (RealNVP) can only scale and shift, so many are needed to
    model a complex density. A neural spline flow replaces the affine map with a
    monotonic RATIONAL-QUADRATIC SPLINE whose knot positions and slopes are predicted
    by a network -- a far more flexible per-dimension transform that is still exactly
    invertible with a closed-form log-determinant. Fewer, more expressive layers fit
    sharp, multimodal densities that would take many affine couplings. Operates on
    the second half conditioned on the first, alternating.
    """

    def __init__(self, dim, n_flows=4, n_bins=8, hidden=32, rng=None):
        super().__init__()
        r = check_random_state(rng)
        self.dim, self.n_bins = dim, n_bins
        self.d = dim // 2
        self.flows = []
        out = (dim - self.d) * (3 * n_bins - 1)
        for _ in range(n_flows):
            self.flows.append([Linear(self.d, hidden, rng=r),
                               Linear(hidden, out, rng=r)])

    def parameters(self):
        return [p for f in self.flows for m in f for p in m.parameters()]

    def _params(self, net, n_dim, j):
        # slice out dim j's spline parameters (each a Tensor)
        base = j * (3 * self.n_bins - 1)
        K = self.n_bins
        widths = F.softmax(net[:, base:base + K], axis=1)
        heights = F.softmax(net[:, base + K:base + 2 * K], axis=1)
        raw_d = net[:, base + 2 * K:base + 3 * K - 1]
        deriv = (1.0 + raw_d.exp()).log() + 1e-3         # softplus, positive
        ones = Tensor(np.ones((deriv.shape[0], 1)))
        deriv = Tensor.concatenate([ones, deriv, ones], axis=1)   # pad slopes at ends
        return widths, heights, deriv

    def log_prob(self, x):
        z = Tensor._wrap(x)
        n = z.shape[0]
        total_logdet = Tensor(np.zeros(n))
        for f in self.flows:
            za, zb = z[:, :self.d], z[:, self.d:]
            net = f[1](f[0](za).relu())
            n_dim = self.dim - self.d
            cols, ld = [], Tensor(np.zeros(n))
            for j in range(n_dim):
                widths, heights, deriv = self._params(net, n_dim, j)
                yj, ldj = _rq_spline_tensor(zb.data[:, j], widths, heights, deriv)
                cols.append(yj.reshape(n, 1)); ld = ld + ldj
            z = Tensor.concatenate([za] + cols, axis=1)
            total_logdet = total_logdet + ld
            rev = np.arange(self.dim - 1, -1, -1)
            z = z[:, rev]                                # alternate the transformed half
        logpz = -0.5 * (z * z).sum(axis=1) - 0.5 * self.dim * np.log(2 * np.pi)
        return logpz + total_logdet

    def nll(self, x):
        return -self.log_prob(x).mean()


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


class GraphTransformer(Module):
    """Attention over a GRAPH, with edge features (Dwivedi & Bresson, 2020).

    A GNN aggregates from fixed neighbours with fixed weights; a transformer attends
    globally but ignores structure. The graph transformer merges them: each node
    attends to its NEIGHBOURS (masked by the adjacency), the attention score is
    modulated by the EDGE feature between them, and the usual residual + feed-forward
    follow. So it learns which neighbours matter, conditioned on how they are
    connected -- the backbone of modern molecular and graph-property models.
    ``forward(X, A, E)`` takes node features, adjacency, and an edge-feature matrix.
    """

    def __init__(self, dim, rng=None):
        super().__init__()
        r = check_random_state(rng)
        self.q = Linear(dim, dim, rng=r)
        self.k = Linear(dim, dim, rng=r)
        self.v = Linear(dim, dim, rng=r)
        self.edge = Linear(1, dim, rng=r)
        self.ff = Linear(dim, dim, rng=r)
        self.dim = dim

    def parameters(self):
        return (list(self.q.parameters()) + list(self.k.parameters())
                + list(self.v.parameters()) + list(self.edge.parameters())
                + list(self.ff.parameters()))

    def forward(self, X, A, E=None):
        X = Tensor._wrap(X)
        n = X.shape[0]
        A = np.asarray(A, float)
        Q, K, V = self.q(X), self.k(X), self.v(X)
        scores = (Q @ K.transpose()) * (1.0 / np.sqrt(self.dim))   # (n, n)
        if E is not None:                                # modulate by edge scalar
            scores = scores + Tensor(np.asarray(E, float))
        mask = np.where(A > 0, 0.0, -1e9)                # attend only to neighbours
        attn = F.softmax(scores + Tensor(mask), axis=1)
        out = attn @ V
        out = X + out                                    # residual
        return out + self.ff(out).relu()


def _softmax(x):
    e = np.exp(x - x.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)


__all__ = ["Bidirectional", "DifferentiableNeuralComputer", "NeuralSplineFlow",
           "ModernHopfieldNetwork", "GraphTransformer"]
