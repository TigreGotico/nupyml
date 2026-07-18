"""Neural architectures v2: sets, graphs, efficient attention, continuous-depth,
state spaces, and learnable-activation networks.

These extend the architecture set (residual/transformer/CausalLM/MDN/Siamese)
with structures for new input types and scaling regimes.
"""
import numpy as np

from ..autograd import Tensor
from ..autograd import functional as F
from .module import Module
from .layers import Linear, LayerNorm, ReLU, GELU
from .attention import MultiHeadAttention


def _mlp(dims, rng):
    layers = []
    for a, b in zip(dims[:-1], dims[1:]):
        layers += [Linear(a, b, rng=rng), ReLU()]
    return layers[:-1]


class _Seq(Module):
    def __init__(self, layers):
        super().__init__()
        self.layers = layers

    def parameters(self):
        return [p for l in self.layers for p in l.parameters()]

    def forward(self, x):
        for l in self.layers:
            x = l(x)
        return x


class DeepSets(Module):
    """Permutation-INVARIANT network over a set (Zaheer et al., 2017).

    A function of a SET must not depend on the order of its elements. DeepSets
    guarantees that structurally: apply the same ``phi`` to every element, POOL
    the results with a symmetric operation (sum), then apply ``rho``::

        f({x_1..x_n}) = rho( sum_i phi(x_i) )

    Any permutation-invariant function has this form, so it is not just a heuristic
    but the general recipe. Input is ``(batch, set_size, features)``; output is
    ``(batch, out_dim)``.
    """

    def __init__(self, in_dim, hidden=64, out_dim=1, rng=None):
        super().__init__()
        self.phi = _Seq(_mlp([in_dim, hidden, hidden], rng))
        self.rho = _Seq(_mlp([hidden, hidden, out_dim], rng))

    def parameters(self):
        return self.phi.parameters() + self.rho.parameters()

    def forward(self, x):
        x = Tensor._wrap(x)
        h = self.phi(x)                               # (b, n, hidden)
        pooled = h.sum(axis=1)                         # symmetric pool -> (b, hidden)
        return self.rho(pooled)


class SetTransformer(Module):
    """Attention-based set model with pooling-by-attention (Lee et al., 2019).

    DeepSets pools with a plain sum, so it cannot model INTERACTIONS between set
    elements ("is there a pair that...?"). The Set Transformer replaces both the
    per-element map and the pooling with ATTENTION: a self-attention block lets
    elements attend to each other, and a learnable SEED vector attends to the set
    to pool it (pooling-by-multihead-attention, PMA). More expressive than
    DeepSets, still permutation-invariant. Input ``(batch, set_size, features)``.
    """

    def __init__(self, in_dim, embed_dim=32, num_heads=4, out_dim=1, rng=None):
        super().__init__()
        self.proj = Linear(in_dim, embed_dim, rng=rng)
        self.sab = MultiHeadAttention(embed_dim, num_heads, rng=rng)   # self-attn
        self.norm = LayerNorm(embed_dim)
        self.seed = Linear(1, embed_dim, rng=rng)     # produces the pooling query
        self.pma = MultiHeadAttention(embed_dim, num_heads, rng=rng)
        self.out = Linear(embed_dim, out_dim, rng=rng)

    def parameters(self):
        return (list(self.proj.parameters()) + list(self.sab.parameters())
                + list(self.norm.parameters()) + list(self.seed.parameters())
                + list(self.pma.parameters()) + list(self.out.parameters()))

    def forward(self, x):
        x = Tensor._wrap(x)
        b = x.shape[0]
        z = self.proj(x)
        z = self.norm(z + self.sab(z, z, z))          # elements attend to each other
        seed = self.seed(Tensor(np.ones((b, 1, 1))))  # one learnable seed per batch
        pooled = self.pma(seed, z, z)                 # seed attends to the set
        return self.out(pooled[:, 0])


class GIN(Module):
    """Graph Isomorphism Network -- the maximally-expressive message-passing GNN
    (Xu et al., 2019).

    A GNN layer updates each node from its neighbours. GIN uses the aggregation
    that makes it as discriminative as the Weisfeiler-Lehman graph-isomorphism
    test (the theoretical ceiling for message passing): SUM the neighbours (sum
    injectively distinguishes multisets, where mean/max lose count information)
    and pass through an MLP::

        h_v <- MLP( (1 + eps) * h_v + sum_{u in N(v)} h_u )

    ``forward(X, A)`` takes node features ``(n, d)`` and an adjacency ``(n, n)``;
    ``graph_embedding`` sum-pools the final node features for graph-level tasks.
    """

    def __init__(self, in_dim, hidden=32, n_layers=2, eps=0.0, rng=None):
        super().__init__()
        self.eps = eps
        self.mlps = []
        d = in_dim
        for _ in range(n_layers):
            self.mlps.append(_Seq(_mlp([d, hidden, hidden], rng)))
            d = hidden

    def parameters(self):
        return [p for m in self.mlps for p in m.parameters()]

    def forward(self, X, A):
        h = Tensor._wrap(X)
        A = Tensor(np.asarray(A, float))
        for mlp in self.mlps:
            h = mlp((1 + self.eps) * h + A @ h)       # sum-aggregate + transform
        return h

    def graph_embedding(self, X, A):
        return self.forward(X, A).sum(axis=0)         # readout for the whole graph


class LinearAttention(Module):
    """Attention in LINEAR time via a kernel feature map (Katharopoulos, 2020).

    Softmax attention costs ``O(n^2)`` because it forms the full n-by-n score
    matrix -- prohibitive for long sequences. Linear attention replaces
    ``softmax(QK^T)V`` with a feature map ``phi`` and reassociates::

        out = phi(Q) ( phi(K)^T V ) / ( phi(Q) phi(K)^T 1 )

    Computing ``phi(K)^T V`` first (a ``d-by-d`` matrix) makes the cost ``O(n d^2)``
    -- LINEAR in sequence length. The trade is a low-rank approximation to the
    attention matrix; here ``phi(x) = relu(x) + 1`` (a non-negative map).
    """

    def __init__(self, embed_dim, rng=None):
        super().__init__()
        self.q = Linear(embed_dim, embed_dim, rng=rng)
        self.k = Linear(embed_dim, embed_dim, rng=rng)
        self.v = Linear(embed_dim, embed_dim, rng=rng)

    def parameters(self):
        return list(self.q.parameters()) + list(self.k.parameters()) + list(self.v.parameters())

    def forward(self, x):
        x = Tensor._wrap(x)
        q = self.q(x).relu() + 1.0                     # phi(Q)  (b, n, d)
        k = self.k(x).relu() + 1.0                     # phi(K)
        v = self.v(x)
        kv = k.swapaxes(1, 2) @ v                       # phi(K)^T V  (b, d, d)
        num = q @ kv                                    # (b, n, d)
        # normaliser: phi(Q) . sum_n phi(K)
        z = q @ k.sum(axis=1).reshape(k.shape[0], k.shape[2], 1)   # (b, n, 1)
        return num / (z + 1e-6)


class NeuralODE(Module):
    """Continuous-depth network: the hidden state evolves by an ODE (Chen, 2018).

    A residual block computes ``h + f(h)``; stack many and you are doing Euler
    steps of an ODE ``dh/dt = f(h)``. Neural ODEs take that limit literally -- the
    "depth" becomes continuous TIME, and the output is the ODE solution at ``t=1``,
    produced by a numerical solver (RK4 here). This gives constant memory
    (backprop through the solver) and adaptive depth. ``forward`` integrates the
    input state from 0 to 1.
    """

    def __init__(self, dim, hidden=32, n_steps=10, rng=None):
        super().__init__()
        self.func = _Seq(_mlp([dim, hidden, dim], rng))
        self.n_steps = n_steps

    def parameters(self):
        return self.func.parameters()

    def forward(self, x):
        h = Tensor._wrap(x)
        dt = 1.0 / self.n_steps
        for _ in range(self.n_steps):                  # fixed-step RK4 integration
            k1 = self.func(h)
            k2 = self.func(h + 0.5 * dt * k1)
            k3 = self.func(h + 0.5 * dt * k2)
            k4 = self.func(h + dt * k3)
            h = h + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        return h


class StateSpaceModel(Module):
    """A trainable linear state-space (diagonal SSM) for long sequences
    (the S4/Mamba idea).

    An RNN mixes memory and nonlinearity and is slow and forgetful over long
    ranges. A linear STATE-SPACE model keeps a linear recurrence
    ``x_t = A x_{t-1} + B u_t``, ``y_t = C x_t`` with a DIAGONAL ``A`` -- so each
    hidden dimension is an independent leaky integrator with its own decay rate.
    Diagonal ``A`` makes it cheap and, crucially, lets you set very slow decays
    that carry information across thousands of steps -- the long-range memory RNNs
    lack. ``A`` is parameterised as ``-softplus`` (via sigmoid gate here) to stay
    stable (decays in (0, 1)). Input ``(batch, T, in_dim)`` -> ``(batch, T, out_dim)``.
    """

    def __init__(self, in_dim, state_dim=16, out_dim=1, rng=None):
        super().__init__()
        self.encode = Linear(in_dim, state_dim, rng=rng)
        self.decode = Linear(state_dim, out_dim, rng=rng)
        # log-decay -> A = sigmoid(a) in (0,1), a stable per-dim decay
        self.a = Linear(1, state_dim, rng=rng)
        self.state_dim = state_dim

    def parameters(self):
        return (list(self.encode.parameters()) + list(self.decode.parameters())
                + list(self.a.parameters()))

    def forward(self, u):
        u = Tensor._wrap(u)
        b, T, _ = u.shape
        Bu = self.encode(u)                            # (b, T, state) = B u_t
        decay = self.a(Tensor(np.ones((1, 1)))).sigmoid()   # (1, state), in (0,1)
        outs = []
        x = Tensor(np.zeros((b, self.state_dim)))
        for t in range(T):
            x = x * decay + Bu[:, t]                    # x_t = A x_{t-1} + B u_t
            outs.append(self.decode(x))
        return Tensor.stack(outs, axis=1)              # (b, T, out_dim)


class KAN(Module):
    """Kolmogorov-Arnold Network: learnable activations on the EDGES (Liu, 2024).

    An MLP puts fixed activations on the NODES and learns linear weights on the
    edges. A KAN inverts this: every edge carries a LEARNABLE univariate function,
    and nodes just sum. By the Kolmogorov-Arnold representation theorem, any
    multivariate function decomposes into sums of univariate ones -- KAN learns
    those univariate pieces directly. Here each edge function is a sum of RBF basis
    functions with learnable coefficients, plus a SiLU base path::

        f_ij(x) = w_base * silu(x) + sum_k c_ijk * rbf_k(x)

    so the network can shape each connection's response rather than picking one
    fixed nonlinearity. One KAN layer maps ``in_dim -> out_dim``.
    """

    def __init__(self, in_dim, out_dim, n_basis=8, grid_range=(-2, 2), rng=None):
        super().__init__()
        from ..utils import check_random_state
        r = check_random_state(rng)
        self.in_dim, self.out_dim, self.n_basis = in_dim, out_dim, n_basis
        self.centers = np.linspace(grid_range[0], grid_range[1], n_basis)
        self.width = (grid_range[1] - grid_range[0]) / n_basis
        # learnable spline coefficients per (out, in, basis) and a base weight
        from .module import Parameter
        self.coef = Parameter(r.randn(out_dim, in_dim, n_basis) * 0.1)
        self.w_base = Parameter(r.randn(out_dim, in_dim) * 0.1)

    def parameters(self):
        return [self.coef, self.w_base]

    def forward(self, x):
        x = Tensor._wrap(x)
        # RBF basis of each input feature: (b, in, n_basis)
        xe = x.reshape(x.shape[0], self.in_dim, 1)
        diff = xe - Tensor(self.centers.reshape(1, 1, -1))
        rbf = (-(diff * diff) * (1.0 / (2 * self.width ** 2))).exp()
        # spline term: coef (out,in,basis) x rbf (b,in,basis), summed -> (b, out)
        spline = (self.coef.reshape(1, self.out_dim, self.in_dim, self.n_basis)
                  * rbf.reshape(x.shape[0], 1, self.in_dim, self.n_basis))
        spline = spline.sum(axis=3).sum(axis=2)        # (b, out)
        base = (x.sigmoid() * x).reshape(x.shape[0], 1, self.in_dim) \
            * self.w_base.reshape(1, self.out_dim, self.in_dim)
        base = base.sum(axis=2)                         # (b, out)
        return spline + base


__all__ = ["DeepSets", "SetTransformer", "GIN", "LinearAttention", "NeuralODE",
           "StateSpaceModel", "KAN"]
