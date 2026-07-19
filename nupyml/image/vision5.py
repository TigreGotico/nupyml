"""Vision v5: edge-preserving denoising, graph-cut segmentation, and invariant
shape descriptors.

Six more vision algorithms. The bilateral filter, non-local means, total-variation
and anisotropic diffusion all remove noise WITHOUT blurring edges, by four different
principles (range weighting, patch self-similarity, sparse gradients, edge-stopping
diffusion). Graph-cut segmentation finds a globally optimal binary labelling by
minimum cut. Hu moments describe a shape invariantly to translation, rotation and scale.
"""
import numpy as np
import scipy.ndimage as ndi

from .vision import _gray


def bilateral_filter(image, spatial_sigma=2.0, range_sigma=0.1, radius=3):
    """Smooth noise but KEEP edges, by weighting on intensity too (Tomasi, 1998).

    A Gaussian blur averages a pixel with its neighbours by DISTANCE alone, so it
    smears edges. The bilateral filter adds a second weight on INTENSITY difference: a
    neighbour only contributes if it is both nearby AND similar in value. So within a
    smooth region it averages away noise, but across an edge the dissimilar side is
    down-weighted to nothing -- the edge survives. It is the classic edge-preserving
    smoother, and the intuition behind many later ones. ``spatial_sigma`` sets the
    neighbourhood, ``range_sigma`` how much intensity difference is tolerated.
    """
    img = _gray(image) if np.asarray(image).ndim == 3 else np.asarray(image, float)
    H, W = img.shape
    out = np.zeros_like(img)
    wsum = np.zeros_like(img)
    ax = np.arange(-radius, radius + 1)
    for dy in ax:
        for dx in ax:
            spatial = np.exp(-(dy ** 2 + dx ** 2) / (2 * spatial_sigma ** 2))
            shifted = np.roll(np.roll(img, dy, axis=0), dx, axis=1)
            rng_w = np.exp(-((img - shifted) ** 2) / (2 * range_sigma ** 2))
            w = spatial * rng_w                          # spatial x range weight
            out += w * shifted; wsum += w
    return out / wsum


def non_local_means(image, patch_radius=2, search_radius=5, h=0.1):
    """Denoise by averaging SIMILAR PATCHES from across the image (Buades, 2005).

    Most denoisers use only nearby pixels; non-local means uses SELF-SIMILARITY. To
    restore a pixel it compares the small PATCH around it to patches all over the image
    and averages their centres, weighted by patch similarity -- so the many repetitions
    of texture and structure in a natural image reinforce each other and cancel the
    (independent) noise. It set the standard for denoising quality before deep methods.
    ``h`` controls how quickly the weight falls off with patch distance.
    """
    img = _gray(image) if np.asarray(image).ndim == 3 else np.asarray(image, float)
    H, W = img.shape
    pad = patch_radius
    padded = np.pad(img, pad, mode="reflect")
    out = np.zeros_like(img); wsum = np.zeros_like(img)

    def patch(y, x):
        return padded[y:y + 2 * pad + 1, x:x + 2 * pad + 1]

    for dy in range(-search_radius, search_radius + 1):
        for dx in range(-search_radius, search_radius + 1):
            shifted = np.roll(np.roll(img, dy, axis=0), dx, axis=1)
            sp = np.pad(shifted, pad, mode="reflect")
            # squared patch distance via a uniform box filter of the pixel diff^2
            diff2 = (padded - sp) ** 2
            pd = ndi.uniform_filter(diff2, 2 * pad + 1)[pad:-pad, pad:-pad]
            w = np.exp(-pd / (h ** 2))
            out += w * shifted; wsum += w
    return out / wsum


def total_variation_denoise(image, weight=0.1, n_iter=100, step=0.2):
    """Denoise by making the gradient SPARSE (Rudin-Osher-Fatemi, 1992).

    Total-variation denoising minimises ``||u - f||^2 + weight * TV(u)`` where TV is the
    integral of the gradient magnitude. Penalising the L1 of the gradient (not the L2)
    is the key: it removes noise -- whose gradient is everywhere -- while ALLOWING the
    few large jumps of true edges, because an L1 penalty tolerates sparse large values.
    The result is piecewise-smooth with crisp edges, the "cartoon" of the image.
    Gradient-descent on the ROF energy here.
    """
    img = _gray(image) if np.asarray(image).ndim == 3 else np.asarray(image, float)
    u = img.copy()
    for _ in range(n_iter):
        gx = np.diff(u, axis=1, append=u[:, -1:])
        gy = np.diff(u, axis=0, append=u[-1:, :])
        gmag = np.sqrt(gx ** 2 + gy ** 2) + 1e-8
        # divergence of the normalised gradient (curvature) drives the TV term
        div = (np.diff(gx / gmag, axis=1, prepend=(gx / gmag)[:, :1])
               + np.diff(gy / gmag, axis=0, prepend=(gy / gmag)[:1, :]))
        u = u - step * ((u - img) - weight * div)
    return u


def anisotropic_diffusion(image, n_iter=20, kappa=0.1, gamma=0.2):
    """Diffusion that STOPS at edges (Perona & Malik, 1990).

    Ordinary (isotropic) diffusion is exactly Gaussian blur -- it smooths everywhere,
    edges included. Perona-Malik makes the diffusion CONDUCTANCE depend on the local
    gradient: heat flows freely within smooth regions but is BLOCKED where the gradient
    is large (an edge). So noise diffuses away inside regions while edges are preserved
    and even sharpened. ``kappa`` sets the gradient scale at which diffusion stops;
    ``gamma`` is the time step.
    """
    img = _gray(image) if np.asarray(image).ndim == 3 else np.asarray(image, float)
    u = img.copy()
    for _ in range(n_iter):
        dn = np.roll(u, -1, 0) - u; ds = np.roll(u, 1, 0) - u
        de = np.roll(u, -1, 1) - u; dw = np.roll(u, 1, 1) - u
        # edge-stopping conductance: exp(-(grad/kappa)^2), small at strong edges
        cn = np.exp(-(dn / kappa) ** 2); cs = np.exp(-(ds / kappa) ** 2)
        ce = np.exp(-(de / kappa) ** 2); cw = np.exp(-(dw / kappa) ** 2)
        u = u + gamma * (cn * dn + cs * ds + ce * de + cw * dw)
    return u


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


def hu_moments(image):
    """Seven shape descriptors invariant to translation, rotation, scale (Hu, 1962).

    To recognise a shape regardless of where it sits, how big it is, or which way it is
    turned, you need descriptors that do not change under those transforms. Hu built
    seven such invariants out of NORMALISED CENTRAL MOMENTS: centring removes
    translation, scale-normalising removes size, and specific polynomial combinations
    cancel rotation. They are a compact, classic signature for template matching and
    optical character recognition. Returns the 7 (log-scaled) Hu moments.
    """
    img = _gray(image) if np.asarray(image).ndim == 3 else np.asarray(image, float)
    H, W = img.shape
    y, x = np.mgrid[0:H, 0:W]
    m00 = img.sum()
    xc = (x * img).sum() / m00; yc = (y * img).sum() / m00   # centroid
    xn, yn = x - xc, y - yc

    def mu(p, q):
        return (xn ** p * yn ** q * img).sum()

    def eta(p, q):
        return mu(p, q) / m00 ** (1 + (p + q) / 2)           # scale-normalised

    n20, n02, n11 = eta(2, 0), eta(0, 2), eta(1, 1)
    n30, n03, n21, n12 = eta(3, 0), eta(0, 3), eta(2, 1), eta(1, 2)
    h = np.zeros(7)
    h[0] = n20 + n02
    h[1] = (n20 - n02) ** 2 + 4 * n11 ** 2
    h[2] = (n30 - 3 * n12) ** 2 + (3 * n21 - n03) ** 2
    h[3] = (n30 + n12) ** 2 + (n21 + n03) ** 2
    h[4] = ((n30 - 3 * n12) * (n30 + n12) * ((n30 + n12) ** 2 - 3 * (n21 + n03) ** 2)
            + (3 * n21 - n03) * (n21 + n03) * (3 * (n30 + n12) ** 2 - (n21 + n03) ** 2))
    h[5] = ((n20 - n02) * ((n30 + n12) ** 2 - (n21 + n03) ** 2)
            + 4 * n11 * (n30 + n12) * (n21 + n03))
    h[6] = ((3 * n21 - n03) * (n30 + n12) * ((n30 + n12) ** 2 - 3 * (n21 + n03) ** 2)
            - (n30 - 3 * n12) * (n21 + n03) * (3 * (n30 + n12) ** 2 - (n21 + n03) ** 2))
    return -np.sign(h) * np.log10(np.abs(h) + 1e-30)         # log-scaled for range


__all__ = ["bilateral_filter", "non_local_means", "total_variation_denoise",
           "anisotropic_diffusion", "graph_cut_segmentation", "hu_moments"]
