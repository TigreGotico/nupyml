"""Optimal transport expansion: unbalanced, Gromov-Wasserstein, sliced, mapping.

The ``core`` module has balanced Sinkhorn/Wasserstein/barycenter/MMD. These are
the variants that handle unequal mass, distributions in DIFFERENT spaces,
high-dimensional scale, and the transport MAP itself.
"""
import numpy as np

from ..utils import check_array, check_random_state


def _cost_matrix(X, Y, p=2):
    diff = X[:, None, :] - Y[None, :, :]
    return (np.abs(diff) ** p).sum(axis=2)


def sinkhorn_unbalanced(a, b, cost, reg=0.1, reg_marginal=1.0, n_iter=200):
    """Unbalanced OT: relax the marginal constraints so mass can be created/destroyed.

    THE PROBLEM WITH BALANCED OT
    ----------------------------
    Ordinary OT insists every unit of mass in ``a`` is moved onto ``b`` exactly --
    so one outlier point with spurious mass drags the whole plan to explain it.
    Unbalanced OT replaces the hard marginal constraints with a soft KL penalty
    (weight ``reg_marginal``): the plan may leave some mass unmatched if matching
    it would cost more than the penalty. That makes it ROBUST to outliers and
    usable when the two distributions genuinely have different total mass (e.g.
    cell populations of different sizes).

    Solved by a scaling iteration like Sinkhorn, but each marginal update is raised
    to the power ``reg_marginal/(reg_marginal+reg)`` instead of dividing exactly --
    the soft-constraint analogue of the balanced rescaling.

    Chizat et al. (2018). Returns the transport plan.
    """
    a = np.asarray(a, float); b = np.asarray(b, float)
    K = np.exp(-cost / reg)
    u = np.ones_like(a); v = np.ones_like(b)
    fi = reg_marginal / (reg_marginal + reg)
    for _ in range(n_iter):
        u = (a / (K @ v + 1e-300)) ** fi
        v = (b / (K.T @ u + 1e-300)) ** fi
    return u[:, None] * K * v[None, :]


def sliced_wasserstein(X, Y, n_projections=50, p=2, random_state=None):
    """Average 1-D Wasserstein distance over many random PROJECTIONS.

    THE TRICK
    ---------
    The Wasserstein distance in 1-D is trivial -- sort both samples and match them
    in order (no linear program). Sliced Wasserstein exploits that: project both
    point clouds onto a random direction, compute the cheap 1-D distance, and
    average over many directions. The result is a true metric that approximates
    the full Wasserstein distance while scaling to high dimensions and large
    samples, because it never builds an ``n*m`` cost matrix. The standard
    OT distance when you have many points in many dimensions.
    """
    X = check_array(X); Y = check_array(Y)
    rng = check_random_state(random_state)
    d = X.shape[1]
    total = 0.0
    for _ in range(n_projections):
        theta = rng.randn(d); theta /= np.linalg.norm(theta) + 1e-12
        xp = np.sort(X @ theta); yp = np.sort(Y @ theta)
        # match equal quantiles by resampling to a common length
        m = max(len(xp), len(yp))
        qx = np.interp(np.linspace(0, 1, m), np.linspace(0, 1, len(xp)), xp)
        qy = np.interp(np.linspace(0, 1, m), np.linspace(0, 1, len(yp)), yp)
        total += np.mean(np.abs(qx - qy) ** p)
    return float((total / n_projections) ** (1.0 / p))


def gromov_wasserstein(D1, D2, a=None, b=None, reg=0.05, n_iter=100):
    """Align distributions living in DIFFERENT spaces, via intra-domain distances.

    THE IDEA
    --------
    Ordinary OT needs a cost between a point in X and a point in Y -- impossible if
    X and Y live in different spaces (a 2-D shape and a 3-D shape, two graphs, two
    embeddings with no shared axes). Gromov-Wasserstein compares only the
    WITHIN-domain distance matrices ``D1`` and ``D2``: it seeks a coupling that
    matches pairs (i,j) in domain 1 to pairs (k,l) in domain 2 so that
    ``D1[i,j] ≈ D2[k,l]`` -- preserving the RELATIONAL structure rather than
    absolute positions. That is what lets it match two graphs or two shapes with
    no common coordinate system.

    Solved here by entropic projected-gradient (a Sinkhorn inner loop on the
    GW gradient ``-D1 T D2``). Returns the coupling.

    Mémoli (2011); Peyré et al. (2016).
    """
    from .core import sinkhorn
    n, m = D1.shape[0], D2.shape[0]
    a = np.full(n, 1.0 / n) if a is None else np.asarray(a, float)
    b = np.full(m, 1.0 / m) if b is None else np.asarray(b, float)
    T = np.outer(a, b)
    for _ in range(n_iter):
        # gradient of the GW objective at the current coupling
        grad = -D1 @ T @ D2.T
        grad = grad - grad.min()                      # keep the cost non-negative
        T = sinkhorn(a, b, grad, reg=reg, n_iter=50)[0]
    return T


def ot_mapping(Xs, Xt, reg=0.1, p=2):
    """Barycentric transport MAP: where does each source point go in the target?

    OT gives a coupling (a soft matching); to actually MOVE a source point you need
    a map. The barycentric projection sends each source point to the weighted
    average of the target points it is coupled to::

        map(x_i) = sum_j T[i,j] * x_t[j] / sum_j T[i,j]

    This is the workhorse of OT domain adaptation -- transport the source features
    onto the target distribution, then train there. Returns the mapped source
    points (same shape as ``Xs``).
    """
    from .core import sinkhorn
    Xs = check_array(Xs); Xt = check_array(Xt)
    a = np.full(len(Xs), 1.0 / len(Xs))
    b = np.full(len(Xt), 1.0 / len(Xt))
    T = sinkhorn(a, b, _cost_matrix(Xs, Xt, p), reg=reg)[0]
    row = T.sum(axis=1, keepdims=True) + 1e-300
    return (T @ Xt) / row


__all__ = ["sinkhorn_unbalanced", "sliced_wasserstein", "gromov_wasserstein",
           "ot_mapping"]
