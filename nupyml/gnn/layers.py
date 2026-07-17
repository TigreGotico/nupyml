"""Graph convolution, GraphSAGE, and graph attention, on the autograd engine.

Graphs are given as a feature matrix ``X`` of shape (n_nodes, n_features) and an
adjacency matrix ``A`` of shape (n_nodes, n_nodes). Everything is a full-batch
message pass -- the whole graph at once -- which is the clearest form; sampling
(as GraphSAGE motivates) is the scaling refinement on top.
"""
import numpy as np

from ..autograd import Tensor
from ..autograd.functional import softmax
from ..nn import Adam, Linear, Module, ModuleList, ReLU
from ..utils import check_random_state


def normalize_adjacency(A):
    """The GCN normalisation: ``D^-1/2 (A + I) D^-1/2``.

    Two fixes rolled into one, and both matter. Adding the IDENTITY gives every
    node a self-loop, so a node keeps its own features instead of being overwritten
    by its neighbours' each layer. The symmetric ``D^-1/2 .. D^-1/2`` scaling
    stops high-degree nodes from dominating: without it, a node with a thousand
    neighbours would accumulate a thousand times the signal and blow up, while a
    leaf node would fade. Normalising by degree puts every node on a comparable
    footing -- it is the difference between a stable GCN and one whose activations
    diverge with graph density.
    """
    A = np.asarray(A, float)
    A_hat = A + np.eye(len(A))                    # self-loops
    deg = A_hat.sum(axis=1)
    d_inv_sqrt = np.diag(1.0 / np.sqrt(np.maximum(deg, 1e-12)))
    return d_inv_sqrt @ A_hat @ d_inv_sqrt


class GCNLayer(Module):
    """One graph-convolution layer: normalised neighbour-average, then a linear map.

    ``H' = A_norm @ H @ W`` -- aggregate each node's normalised neighbourhood, then
    transform. The aggregation is a fixed weighted MEAN (the normalised
    adjacency), which is what makes GCN simple and permutation-invariant, and also
    what limits it: every neighbour counts by degree alone, never by relevance
    (which is what GAT adds).
    """

    def __init__(self, in_features, out_features, rng=None):
        super().__init__()
        self.linear = Linear(in_features, out_features, rng=rng)

    def forward(self, X, A_norm):
        # aggregate over the normalised graph, then apply the shared linear map
        aggregated = Tensor(A_norm) @ Tensor._wrap(X)
        return self.linear(aggregated)


class GCN(Module):
    """A stack of graph-convolution layers for node classification.

    ``k`` layers let each node see its ``k``-hop neighbourhood. Note the classic
    limit: too many layers cause OVER-SMOOTHING -- after enough hops every node has
    aggregated most of the graph and all the node features converge to the same
    vector, so the model can no longer tell nodes apart. Two or three layers is
    usually the sweet spot, which is a striking contrast to the "deeper is better"
    rule elsewhere in deep learning, and worth understanding: on a graph, depth
    means REACH, and too much reach erases local distinctions.
    """

    def __init__(self, in_features, hidden, n_classes, n_layers=2, rng=None):
        super().__init__()
        rng = check_random_state(rng)
        dims = [in_features] + [hidden] * (n_layers - 1) + [n_classes]
        self.layers = ModuleList([GCNLayer(dims[i], dims[i + 1], rng=rng)
                                  for i in range(n_layers)])
        self._relu = ReLU()

    def forward(self, X, A):
        A_norm = normalize_adjacency(A)
        h = Tensor._wrap(X)
        for i, layer in enumerate(self.layers):
            h = layer(h, A_norm)
            if i < len(self.layers) - 1:
                h = self._relu(h)
        return h


class GraphSAGELayer(Module):
    """Aggregate SAMPLED neighbours, then concatenate with the node's own state.

    TWO IDEAS THAT MAKE IT SCALE
    ----------------------------
    * **Sampling.** Instead of aggregating ALL neighbours (impossible for a node
      with a million of them), sample a fixed number. The cost per node becomes
      constant, independent of degree, which is what lets GraphSAGE run on
      web-scale graphs.
    * **Concatenate, not merge.** The node's own vector is CONCATENATED with the
      aggregated neighbour vector before the linear map, rather than summed in.
      Keeping "self" and "neighbourhood" as separate halves lets the layer weigh
      them differently, and helps avoid the over-smoothing that pure averaging
      invites.

    "SAGE" = SAmple and aGgrEgate. It is also INDUCTIVE: because it learns an
    aggregation FUNCTION rather than a per-node embedding, it can embed nodes
    (even whole graphs) never seen during training -- which transductive GCN
    cannot.

    Hamilton, Ying & Leskovec (2017).
    """

    def __init__(self, in_features, out_features, n_sample=10, rng=None):
        super().__init__()
        self.n_sample = n_sample
        # 2*in_features: the concatenation of self and aggregated-neighbours
        self.linear = Linear(2 * in_features, out_features, rng=rng)
        self._rng = check_random_state(rng)

    def forward(self, X, A):
        X = Tensor._wrap(X)
        n = X.shape[0]
        A = np.asarray(A)
        # mean of a SAMPLED subset of each node's neighbours
        agg = np.zeros((n, X.shape[1]))
        for i in range(n):
            nbrs = np.where(A[i] > 0)[0]
            if len(nbrs) == 0:
                continue
            if len(nbrs) > self.n_sample:
                nbrs = self._rng.choice(nbrs, self.n_sample, replace=False)
            agg[i] = X.data[nbrs].mean(axis=0)
        # concatenate self with the neighbourhood, then transform
        combined = Tensor.concatenate([X, Tensor(agg)], axis=1)
        return self.linear(combined)


class GraphSAGE(Module):
    """A stack of GraphSAGE layers."""

    def __init__(self, in_features, hidden, n_classes, n_layers=2, n_sample=10,
                 rng=None):
        super().__init__()
        rng = check_random_state(rng)
        dims = [in_features] + [hidden] * (n_layers - 1) + [n_classes]
        self.layers = ModuleList([
            GraphSAGELayer(dims[i], dims[i + 1], n_sample, rng=rng)
            for i in range(n_layers)])
        self._relu = ReLU()

    def forward(self, X, A):
        h = Tensor._wrap(X)
        for i, layer in enumerate(self.layers):
            h = layer(h, A)
            if i < len(self.layers) - 1:
                h = self._relu(h)
        return h


class GATLayer(Module):
    """Graph attention: LEARN how much each neighbour matters.

    THE STEP BEYOND GCN
    -------------------
    GCN weights every neighbour by degree alone -- a fixed, structural average.
    GAT instead computes an ATTENTION score for each edge from the two endpoints'
    features, so a node can attend MORE to the neighbours that are relevant to it
    and less to incidental ones. The weights are learned and data-dependent, the
    same idea as transformer self-attention, restricted to the graph's edges
    rather than all pairs.

    The scores are softmax-normalised OVER EACH NODE'S NEIGHBOURS, so they form a
    proper distribution per node -- how it divides its attention. This lets GAT
    handle graphs where some edges matter far more than others, which the
    degree-only GCN cannot express, and it does so without needing the graph
    structure to be known in advance for normalisation.

    Velickovic et al. (2018).
    """

    def __init__(self, in_features, out_features, rng=None):
        super().__init__()
        self.W = Linear(in_features, out_features, bias=False, rng=rng)
        # the attention mechanism: a small vector scoring each (source, target) pair
        self.a_src = Linear(out_features, 1, bias=False, rng=rng)
        self.a_dst = Linear(out_features, 1, bias=False, rng=rng)

    def forward(self, X, A):
        X = Tensor._wrap(X)
        h = self.W(X)                            # (n, out)
        n = h.shape[0]
        # additive attention score for every edge: a_src(h_i) + a_dst(h_j)
        src = self.a_src(h).data.reshape(-1, 1)   # (n, 1)
        dst = self.a_dst(h).data.reshape(1, -1)   # (1, n)
        scores = src + dst                        # (n, n)
        scores = np.where(scores > 0, scores, 0.2 * scores)   # LeakyReLU

        A = np.asarray(A)
        mask = (A + np.eye(n)) > 0                # attend to neighbours and self
        scores = np.where(mask, scores, -1e9)     # non-edges get no attention
        # softmax over each node's neighbours -> a distribution of attention
        attn = np.exp(scores - scores.max(axis=1, keepdims=True))
        attn = attn / attn.sum(axis=1, keepdims=True)
        return Tensor(attn) @ h

    __constants__ = ()


class GAT(Module):
    """A stack of graph-attention layers."""

    def __init__(self, in_features, hidden, n_classes, n_layers=2, rng=None):
        super().__init__()
        rng = check_random_state(rng)
        dims = [in_features] + [hidden] * (n_layers - 1) + [n_classes]
        self.layers = ModuleList([GATLayer(dims[i], dims[i + 1], rng=rng)
                                  for i in range(n_layers)])
        self._relu = ReLU()

    def forward(self, X, A):
        h = Tensor._wrap(X)
        for i, layer in enumerate(self.layers):
            h = layer(h, A)
            if i < len(self.layers) - 1:
                h = self._relu(h)
        return h


__all__ = ["GCN", "GraphSAGE", "GAT", "normalize_adjacency"]
