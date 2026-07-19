"""Frequent subgraph mining (gSpan-lite): recurring connected subgraphs across a graph set."""
import math
from itertools import permutations


class _Pattern:
    """A small labelled undirected graph: node labels plus an edge->label map."""

    def __init__(self, node_labels, edges):
        self.node_labels = tuple(node_labels)
        # edges: dict {(i, j): edge_label} with i < j
        self.edges = dict(edges)

    def certificate(self):
        """Canonical form: the lexicographically smallest relabelling of the nodes.

        Two patterns are the same subgraph iff their certificates match, so this is how
        isomorphic duplicates are collapsed. Feasible only because patterns stay tiny.
        """
        n = len(self.node_labels)
        best = None
        for perm in permutations(range(n)):
            nl = tuple(self.node_labels[perm.index(i)] for i in range(n))
            es = tuple(sorted((min(perm[i], perm[j]), max(perm[i], perm[j]), lab)
                              for (i, j), lab in self.edges.items()))
            cand = (nl, es)
            if best is None or cand < best:
                best = cand
        return best


def _subgraph_present(pattern, graph):
    """True if `pattern` occurs as a node/edge-labelled subgraph of `graph` (VF2-lite)."""
    gnl, gadj = graph
    pn = len(pattern.node_labels)
    padj = {i: {} for i in range(pn)}
    for (i, j), lab in pattern.edges.items():
        padj[i][j] = lab; padj[j][i] = lab
    order = sorted(range(pn), key=lambda i: -len(padj[i]))   # most-constrained first

    def backtrack(k, mapping, used):
        if k == len(order):
            return True
        p = order[k]
        for g in range(len(gnl)):
            if g in used or gnl[g] != pattern.node_labels[p]:
                continue
            ok = True
            for pn2, lab in padj[p].items():                 # honour already-placed edges
                if pn2 in mapping:
                    gn2 = mapping[pn2]
                    if gadj.get(g, {}).get(gn2) != lab:
                        ok = False; break
            if ok and backtrack(k + 1, {**mapping, p: g}, used | {g}):
                return True
        return False

    return backtrack(0, {}, set())


def _to_adj(graph):
    node_labels, edges = graph
    adj = {i: {} for i in range(len(node_labels))}
    for u, v, lab in edges:
        adj[u][v] = lab; adj[v][u] = lab
    return tuple(node_labels), adj


def frequent_subgraphs(graphs, min_support=0.5, max_edges=3):
    """Find connected subgraphs recurring across many graphs (Yan & Han's gSpan, 2002).

    Molecules, programs and social circles are graphs, and what matters is the recurring
    MOTIF -- a functional group, a code idiom, a friendship triangle. Frequent subgraph
    mining finds every connected subgraph that appears in at least a fraction of the input
    graphs. The combinatorics are brutal (subgraph isomorphism is the inner loop), so it
    grows patterns one EDGE at a time and, like gSpan, keeps a canonical form of each
    pattern to avoid rediscovering the same subgraph through different build orders.
    Support is anti-monotone, so infrequent patterns are never extended. Each graph is
    ``(node_labels, [(u, v, edge_label), ...])``; returns ``(pattern, support)`` pairs.
    """
    n = len(graphs)
    min_count = math.ceil(min_support * n)
    adj_graphs = [_to_adj(g) for g in graphs]

    def support(pattern):
        return sum(_subgraph_present(pattern, g) for g in adj_graphs)

    # seed: all distinct labelled single edges
    seeds = {}
    for node_labels, edges in graphs:
        for u, v, lab in edges:
            a, b = sorted((node_labels[u], node_labels[v]))
            p = _Pattern([a, b], {(0, 1): lab})
            seeds[p.certificate()] = p
    frequent = [p for p in seeds.values() if support(p) >= min_count]
    results = [(p, support(p)) for p in frequent]

    edges_k = 1
    while frequent and edges_k < max_edges:
        grown = {}
        for p in frequent:
            n_nodes = len(p.node_labels)
            # extension A: attach a new node to an existing one
            for i in range(n_nodes):
                for node_labels, edges in graphs:
                    for u, v, lab in edges:
                        for new_lab in (node_labels[v], node_labels[u]):
                            nl = list(p.node_labels) + [new_lab]
                            e = dict(p.edges); e[(i, n_nodes)] = lab
                            cand = _Pattern(nl, e)
                            grown.setdefault(cand.certificate(), cand)
            # extension B: add an edge between two existing, non-adjacent nodes
            for i in range(n_nodes):
                for j in range(i + 1, n_nodes):
                    if (i, j) in p.edges:
                        continue
                    labs = {lab for _, es in graphs for _, _, lab in es}
                    for lab in labs:
                        e = dict(p.edges); e[(i, j)] = lab
                        cand = _Pattern(list(p.node_labels), e)
                        grown.setdefault(cand.certificate(), cand)
        frequent = []
        seen = {r[0].certificate() for r in results}
        for cert, p in grown.items():
            if cert in seen:
                continue
            sc = support(p)
            if sc >= min_count:
                frequent.append(p); results.append((p, sc)); seen.add(cert)
        edges_k += 1
    return sorted(results, key=lambda r: (-r[1], len(r[0].edges)))


__all__ = ["frequent_subgraphs"]
