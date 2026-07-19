"""Graph-cut segmentation: globally optimal binary labelling by minimum cut."""
import numpy as np

from .vision import _gray


def graph_cut_segmentation(image, fg_seeds, bg_seeds, sigma=0.1, lam=2.0):
    """Globally-optimal binary segmentation by MINIMUM CUT (Boykov & Jolly, 2001).

    Segmentation is a labelling problem: each pixel is foreground or background, and
    good labellings keep similar neighbours together. Boykov-Jolly casts it as a graph
    where every pixel connects to a SOURCE (foreground) and SINK (background) by how
    well it matches the seeds, and to its neighbours by how similar they are. The
    MINIMUM CUT separating source from sink -- computable exactly by max-flow -- is the
    globally optimal segmentation under that energy, cleanly cutting along low-
    similarity (edge) boundaries. ``fg_seeds``/``bg_seeds`` are lists of (row, col).
    """
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import maximum_flow
    img = _gray(image) if np.asarray(image).ndim == 3 else np.asarray(image, float)
    H, W = img.shape
    n = H * W
    S, T = n, n + 1                                       # source, sink node ids
    fg = np.array([img[y, x] for y, x in fg_seeds]).mean()
    bg = np.array([img[y, x] for y, x in bg_seeds]).mean()
    scale = 1000
    rows, cols, data = [], [], []

    def add(a, b, w):
        rows.append(a); cols.append(b); data.append(max(int(w * scale), 0))

    for y in range(H):
        for x in range(W):
            i = y * W + x
            # data terms: cheaper to keep the label whose seed value is closer
            add(S, i, np.exp(-((img[y, x] - fg) ** 2) / (2 * sigma ** 2)))
            add(i, T, np.exp(-((img[y, x] - bg) ** 2) / (2 * sigma ** 2)))
            for dy, dx in ((0, 1), (1, 0)):              # neighbour smoothness terms
                ny, nx = y + dy, x + dx
                if ny < H and nx < W:
                    j = ny * W + nx
                    w = lam * np.exp(-((img[y, x] - img[ny, nx]) ** 2) / (2 * sigma ** 2))
                    add(i, j, w); add(j, i, w)
    for y, x in fg_seeds:
        add(S, y * W + x, 1e6)
    for y, x in bg_seeds:
        add(y * W + x, T, 1e6)
    graph = csr_matrix((data, (rows, cols)), shape=(n + 2, n + 2))
    res = maximum_flow(graph, S, T)
    flow = res.flow
    # foreground = nodes still reachable from the source in the residual graph
    residual = graph - flow
    reachable = np.zeros(n + 2, bool); stack = [S]; reachable[S] = True
    residual = residual.tocsr()
    while stack:
        u = stack.pop()
        for idx in range(residual.indptr[u], residual.indptr[u + 1]):
            v = residual.indices[idx]
            if residual.data[idx] > 0 and not reachable[v]:
                reachable[v] = True; stack.append(v)
    return reachable[:n].reshape(H, W).astype(int)


__all__ = ["graph_cut_segmentation"]
