"""Graph algorithms and network embeddings: learning from connectivity.

WHAT THE DATA IS
----------------
A graph -- nodes joined by edges -- where the STRUCTURE is the signal: who links
to whom, who is central, which nodes cluster together. This is not feature-vector
data, so it needs its own toolkit. Everything here takes an adjacency matrix
``A`` (``A[i, j] > 0`` means an edge) and answers a structural question.

THE FAMILIES
------------
* **Importance** (``algorithms.py``) -- which nodes MATTER? ``pagerank`` (a
  node is important if important nodes link to it -- the recursive definition
  that ranked the web), ``hits`` (hubs and authorities), and the classical
  ``centrality`` measures (degree, closeness, betweenness, eigenvector), each a
  different notion of "central".
* **Community** (``community.py``) -- which nodes CLUSTER? ``louvain`` (greedily
  maximise modularity) and ``label_propagation`` (let labels spread along edges
  until they stabilise) find groups more densely connected inside than out --
  clustering, but on a graph rather than in space.
* **Embedding** (``embeddings.py``) -- turn nodes into VECTORS so ordinary ML
  applies. ``DeepWalk`` and ``node2vec`` take random walks over the graph and
  treat them as "sentences", then run word2vec: nodes that co-occur on walks get
  similar vectors. Structure becomes geometry.
* **Kernels** (``kernels.py``) -- compare whole GRAPHS. The Weisfeiler-Lehman
  kernel counts shared subtree patterns, so two molecules or two networks can be
  fed to an SVM.

The through-line: each turns "who is connected to whom" into something a
downstream model or a human can use -- a ranking, a partition, a vector, a
similarity.
"""
from .algorithms import pagerank, hits, centrality
from .community import louvain, label_propagation, modularity
from .embeddings import DeepWalk, Node2Vec
from .kernels import weisfeiler_lehman_kernel
from ._advanced import (common_neighbors, jaccard_coefficient,
                        adamic_adar_index, resource_allocation_index,
                        preferential_attachment, katz_index,
                        degree_assortativity, girvan_newman, max_flow, min_cut,
                        shortest_path_kernel, random_walk_kernel, LINE)

__all__ = [
    "pagerank", "hits", "centrality", "louvain", "label_propagation",
    "modularity", "DeepWalk", "Node2Vec", "weisfeiler_lehman_kernel",
    "common_neighbors", "jaccard_coefficient", "adamic_adar_index",
    "resource_allocation_index", "preferential_attachment", "katz_index",
    "degree_assortativity", "girvan_newman", "max_flow", "min_cut",
    "shortest_path_kernel", "random_walk_kernel", "LINE",
]
