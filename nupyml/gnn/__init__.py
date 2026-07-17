"""Graph neural networks: learning on data that is a GRAPH, not a grid or sequence.

WHAT MAKES GRAPHS DIFFERENT
---------------------------
A CNN assumes a grid (each pixel has the same neighbour layout); an RNN assumes a
sequence (each token has one predecessor). A graph has neither -- nodes have
varying numbers of neighbours in no fixed order. A social network, a molecule, a
citation graph: the connectivity IS the data, and a model must respect two things
a grid/sequence model can ignore:

* **Permutation invariance.** Relabelling the nodes must not change the answer --
  a molecule is the same molecule however you number its atoms. So neighbours must
  be combined by a SYMMETRIC aggregation (sum, mean, max), never anything that
  depends on order.
* **Variable neighbourhoods.** Each node has a different number of neighbours, so
  the operation must handle any degree.

THE ONE IDEA: MESSAGE PASSING
-----------------------------
Every GNN here is the same loop. Each node gathers ("aggregates") a message from
its neighbours, combines it with its own state, and updates::

    for each node:
        message = AGGREGATE( transform(h_u) for u in neighbours )
        h_node  = UPDATE( h_node, message )

Stack ``k`` such layers and information flows ``k`` hops -- a node ends up knowing
about its ``k``-hop neighbourhood. The whole variety of GNNs is just different
choices of AGGREGATE and transform:

* ``GCN`` -- aggregate by a NORMALISED mean of neighbours. The simplest, and a
  first-order approximation to spectral graph convolution.
* ``GraphSAGE`` -- SAMPLE a fixed number of neighbours and aggregate them, so it
  scales to huge graphs and generalises to unseen nodes.
* ``GAT`` -- learn ATTENTION weights over neighbours, so a node can weigh
  important neighbours more than incidental ones.

Kipf & Welling (2017); Hamilton et al. (2017); Velickovic et al. (2018).
"""
from .layers import GCN, GraphSAGE, GAT

__all__ = ["GCN", "GraphSAGE", "GAT"]
