"""Optimal transport solvers and distances."""
import numpy as np
from scipy.spatial.distance import cdist


def sinkhorn(a, b, cost, reg=0.1, n_iter=200, tol=1e-9):
    """Entropic optimal transport by Sinkhorn iterations.

    THE ALGORITHM
    -------------
    We want a transport plan ``P`` (how much mass moves from each source point to
    each target point) that minimises ``<P, cost>`` while matching the marginals
    ``a`` and ``b``. Adding an entropy penalty ``reg * H(P)`` makes the optimal
    ``P`` take the form ``diag(u) K diag(v)`` where ``K = exp(-cost/reg)`` -- and
    then ``u`` and ``v`` are found by ALTERNATELY rescaling so the row-sums match
    ``a`` and the column-sums match ``b``. Each update is one matrix-vector
    product, and they converge geometrically.

    That is the whole trick: the intractable transport LP becomes a few dozen
    cheap normalisations. ``reg`` trades speed and smoothness against accuracy --
    small ``reg`` approaches the true (sharp) transport plan but converges slower
    and can underflow; large ``reg`` gives a blurry plan fast.

    Returns ``(plan, cost_value)``.
    """
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    K = np.exp(-cost / reg)                    # the Gibbs kernel
    u = np.ones(len(a))
    v = np.ones(len(b))
    for _ in range(n_iter):
        u_prev = u
        # the two rescalings: make rows sum to a, then columns sum to b
        u = a / (K @ v + 1e-300)
        v = b / (K.T @ u + 1e-300)
        if np.max(np.abs(u - u_prev)) < tol:
            break
    plan = u[:, None] * K * v[None, :]
    return plan, float(np.sum(plan * cost))


def wasserstein_distance(x, y, p=2, reg=None):
    """The optimal-transport distance between two point sets (uniform weights).

    THE 1-D SHORTCUT
    ----------------
    In one dimension optimal transport has a closed form and needs no solver at
    all: SORT both samples and match them in order (the i-th smallest of one to
    the i-th smallest of the other). That monotone matching is provably optimal,
    so the Wasserstein-p distance is just the p-norm of the sorted differences --
    ``O(n log n)``, exact.

    HIGHER DIMENSIONS
    -----------------
    No such shortcut exists, so this falls back to a cost matrix plus either the
    exact assignment (small n) or ``sinkhorn`` (when ``reg`` is given). The 1-D
    case is worth isolating because it is where OT is cheapest and most-used
    (comparing histograms, quantile functions, and the like).
    """
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    if x.ndim == 1 or x.shape[1] == 1:
        # exact: sort and match in order
        xs = np.sort(x.ravel())
        ys = np.sort(y.ravel())
        # interpolate to a common length if the samples differ in size
        if len(xs) != len(ys):
            q = np.linspace(0, 1, max(len(xs), len(ys)))
            xs = np.quantile(xs, q)
            ys = np.quantile(ys, q)
        return float(np.mean(np.abs(xs - ys) ** p) ** (1 / p))

    x = np.atleast_2d(x)
    y = np.atleast_2d(y)
    cost = cdist(x, y) ** p
    a = np.full(len(x), 1 / len(x))
    b = np.full(len(y), 1 / len(y))
    if reg is not None:
        _, c = sinkhorn(a, b, cost, reg=reg)
        return float(c ** (1 / p))
    # exact small-scale transport via a linear assignment on the cost
    from scipy.optimize import linear_sum_assignment
    if len(x) == len(y):
        r, cidx = linear_sum_assignment(cost)
        return float((cost[r, cidx].mean()) ** (1 / p))
    _, c = sinkhorn(a, b, cost, reg=0.01)
    return float(c ** (1 / p))


def barycenter(distributions, weights=None, reg=0.1, n_iter=100):
    """The Wasserstein "average" of several distributions on a shared support.

    THE IDEA
    --------
    Averaging distributions pointwise blends them into a blurry superposition --
    the mean of two separated bumps is two half-height bumps, which resembles
    neither. The Wasserstein barycenter instead averages them by TRANSPORT: it is
    the distribution whose total optimal-transport cost to all the inputs is
    minimal, and for two separated bumps it is a single bump in the MIDDLE -- the
    shape-aware average you actually wanted.

    Computed by the entropic (Sinkhorn) barycenter iteration: repeatedly transport
    the current estimate toward each input and combine in the log domain.
    ``distributions`` is a list of histograms over a common support.

    Agueh & Carlier (2011); Benamou et al. (2015).
    """
    dists = [np.asarray(d, float) for d in distributions]
    n = len(dists[0])
    weights = np.full(len(dists), 1 / len(dists)) if weights is None \
        else np.asarray(weights, float)
    # cost between support bins (assume they lie on a line 0..n-1)
    support = np.arange(n)
    cost = (support[:, None] - support[None, :]) ** 2
    K = np.exp(-cost / reg)

    bary = np.ones(n) / n
    for _ in range(n_iter):
        # geometric mean of the transported inputs, weighted -- the log-domain
        # barycenter update
        log_bary = np.zeros(n)
        for w, d in zip(weights, dists):
            u = d / (K @ (bary / (K @ d + 1e-300)) + 1e-300)
            log_bary += w * np.log(K.T @ u + 1e-300)
        bary = np.exp(log_bary)
        bary /= bary.sum()
    return bary


def ot_domain_adaptation(X_source, y_source, X_target, reg=0.1):
    """Transport labelled SOURCE points onto the TARGET distribution.

    THE IDEA
    --------
    You have labels for a source domain (say, images from one camera) but only
    unlabelled data from a target domain (another camera, different lighting).
    The distributions differ, so a classifier trained on the source degrades on
    the target. Optimal transport bridges them: compute the transport plan from
    source to target, then MOVE each source point to the barycentre of the target
    points it maps to (barycentric mapping). The transported source keeps its
    labels but now lives in the target distribution, so a classifier trained on it
    works on the target.

    Returns the transported source features (same labels apply).

    Courty, Flamary & Tuia (2014).
    """
    X_source = np.asarray(X_source, float)
    X_target = np.asarray(X_target, float)
    cost = cdist(X_source, X_target) ** 2
    a = np.full(len(X_source), 1 / len(X_source))
    b = np.full(len(X_target), 1 / len(X_target))
    plan, _ = sinkhorn(a, b, cost, reg=reg)
    # barycentric map: each source point goes to the weighted mean of the target
    # points it sends mass to
    row_sums = plan.sum(axis=1, keepdims=True)
    return (plan @ X_target) / (row_sums + 1e-300)


def _rbf_kernel(X, Y, gamma):
    return np.exp(-gamma * cdist(X, Y, "sqeuclidean"))


def mmd(X, Y, gamma=1.0):
    """Maximum Mean Discrepancy: a kernel test of whether two samples MATCH.

    THE IDEA
    --------
    Map both samples into a kernel feature space and compare their MEAN
    embeddings. If two samples come from the same distribution their mean
    embeddings coincide (MMD = 0); the more the distributions differ, the larger
    the gap. Expanded with the kernel trick::

        MMD^2 = mean k(x,x') + mean k(y,y') - 2 mean k(x,y)

    -- entirely in terms of kernel evaluations, no density estimation. It is far
    cheaper than optimal transport (no plan to solve) and is the standard
    two-sample test and the loss for generative "moment matching" networks. With a
    characteristic kernel (like the RBF used here), MMD = 0 iff the distributions
    are identical, so it detects ANY difference given enough data.

    Gretton et al. (2012).
    """
    X = np.atleast_2d(np.asarray(X, float))
    Y = np.atleast_2d(np.asarray(Y, float))
    if X.shape[1] != Y.shape[1]:
        X, Y = X.reshape(-1, 1), Y.reshape(-1, 1)
    kxx = _rbf_kernel(X, X, gamma).mean()
    kyy = _rbf_kernel(Y, Y, gamma).mean()
    kxy = _rbf_kernel(X, Y, gamma).mean()
    return float(max(kxx + kyy - 2 * kxy, 0.0))


def energy_distance(X, Y):
    """Energy distance: MMD's kernel-free cousin, using plain distances.

    ``2 E|X-Y| - E|X-X'| - E|Y-Y'|`` over the samples. It is MMD with the distance
    itself as the (negative-definite) kernel, so it needs no bandwidth to tune --
    the appeal over MMD. Zero iff the two distributions are identical, positive
    otherwise, and it is the statistic behind the energy two-sample test and
    distance correlation.

    Szekely & Rizzo (2013).
    """
    X = np.atleast_2d(np.asarray(X, float))
    Y = np.atleast_2d(np.asarray(Y, float))
    if X.shape[1] != Y.shape[1]:
        X, Y = X.reshape(-1, 1), Y.reshape(-1, 1)
    d_xy = cdist(X, Y).mean()
    d_xx = cdist(X, X).mean()
    d_yy = cdist(Y, Y).mean()
    return float(max(2 * d_xy - d_xx - d_yy, 0.0))


__all__ = ["sinkhorn", "wasserstein_distance", "barycenter",
           "ot_domain_adaptation", "mmd", "energy_distance"]
