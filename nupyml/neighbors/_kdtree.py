"""A KD-tree for fast exact nearest-neighbour queries.

Brute-force k-NN scores every training point for every query -- ``O(n)`` per
query. A KD-tree recursively partitions space by axis-aligned splits so a query
can PRUNE whole branches it cannot possibly beat, giving ``O(log n)`` queries in
low dimensions. This is the classic acceleration behind fast neighbour methods.
"""
import numpy as np

from ..base import BaseEstimator, check_is_fitted
from ..utils import check_array


class KDTree(BaseEstimator):
    """K-dimensional tree for exact k-nearest-neighbour search.

    THE PRUNING THAT MAKES IT FAST
    ------------------------------
    Build: recursively split the points on the axis of greatest spread at the
    median, so each node owns an axis-aligned box. Query: descend to the leaf
    containing the query, then unwind -- at each parent, only cross to the sibling
    box if the distance from the query to the splitting plane is smaller than the
    current k-th best. That single check lets whole subtrees be skipped, which is
    the entire speed-up over brute force. Exact (not approximate); the win fades
    as dimension grows (the curse of dimensionality), so it shines in low-D.
    """

    def __init__(self, leaf_size=16):
        self.leaf_size = leaf_size

    def fit(self, X, y=None):
        self.data_ = check_array(X)
        self.tree_ = self._build(np.arange(len(self.data_)))
        return self

    def _build(self, idx):
        if len(idx) <= self.leaf_size:
            return {"leaf": True, "idx": idx}
        pts = self.data_[idx]
        axis = int(np.argmax(pts.max(0) - pts.min(0)))   # split the widest axis
        order = idx[np.argsort(self.data_[idx, axis])]
        mid = len(order) // 2
        return {"leaf": False, "axis": axis,
                "split": self.data_[order[mid], axis],
                "point": order[mid],
                "left": self._build(order[:mid]),
                "right": self._build(order[mid + 1:])}

    def query(self, X, k=1):
        """Return (distances, indices) of the ``k`` nearest training points."""
        check_is_fitted(self, "tree_")
        X = check_array(X)
        dists = np.empty((len(X), k))
        idxs = np.empty((len(X), k), dtype=int)
        for i, q in enumerate(X):
            heap = []                                 # list of (-dist, index), size <= k
            self._search(self.tree_, q, k, heap)
            heap.sort(key=lambda t: -t[0])
            dists[i] = [np.sqrt(-d) for d, _ in heap]
            idxs[i] = [j for _, j in heap]
        return dists, idxs

    def _search(self, node, q, k, heap):
        if node["leaf"]:
            for j in node["idx"]:
                self._consider(q, j, k, heap)
            return
        self._consider(q, node["point"], k, heap)
        diff = q[node["axis"]] - node["split"]
        near, far = ("left", "right") if diff < 0 else ("right", "left")
        self._search(node[near], q, k, heap)
        # only cross to the far side if it could hold a closer point
        if len(heap) < k or diff ** 2 < -heap[0][0]:
            self._search(node[far], q, k, heap)

    def _consider(self, q, j, k, heap):
        import heapq
        d2 = np.sum((q - self.data_[j]) ** 2)
        if len(heap) < k:
            heapq.heappush(heap, (-d2, int(j)))       # max-heap on squared distance
        elif d2 < -heap[0][0]:
            heapq.heapreplace(heap, (-d2, int(j)))


__all__ = ["KDTree"]
