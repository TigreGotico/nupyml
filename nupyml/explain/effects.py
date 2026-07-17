"""Accumulated Local Effects and the H-statistic -- effects done right when
features are correlated.

Partial dependence (in ``attribution``) averages the model over the MARGINAL of
the other features, which invents impossible feature combinations when features
are correlated -- and then reports a biased effect. These two tools fix or
quantify that.
"""
import numpy as np

from ..utils import check_array


def _predict(model, X):
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, -1]
    return model.predict(X)


def accumulated_local_effects(model, X, feature, n_bins=20):
    """ALE: the unbiased replacement for a partial-dependence plot.

    THE IDEA
    --------
    Instead of asking "what does the model predict if I move feature j across its
    whole range for everyone" (partial dependence -- which uses unrealistic
    points), ALE asks a LOCAL question inside each small interval of feature j::

        for points whose x_j is in [z_{k-1}, z_k], how much does the prediction
        change when x_j moves from z_{k-1} to z_k, holding their other features
        as they actually are?

    Averaging those local DIFFERENCES per interval and ACCUMULATING them across
    intervals gives the effect curve. Because it only ever perturbs x_j within a
    bin and uses real neighbouring points, it never evaluates the model off the
    data manifold -- so, unlike PDP, it stays unbiased under feature correlation.

    Apley & Zhu (2020). Returns (bin_centres, ale_values), the ALE centred to
    mean zero.
    """
    X = check_array(X)
    x = X[:, feature]
    edges = np.percentile(x, np.linspace(0, 100, n_bins + 1))
    edges = np.unique(edges)
    n_bins = len(edges) - 1
    ale = np.zeros(n_bins)
    for k in range(n_bins):
        lo, hi = edges[k], edges[k + 1]
        mask = (x >= lo) & (x <= hi) if k == n_bins - 1 else (x >= lo) & (x < hi)
        if not mask.any():
            continue
        X_lo = X[mask].copy(); X_lo[:, feature] = lo
        X_hi = X[mask].copy(); X_hi[:, feature] = hi
        # mean local change across the interval
        ale[k] = np.mean(_predict(model, X_hi) - _predict(model, X_lo))
    ale = np.cumsum(ale)
    ale -= ale.mean()                                  # centre to zero
    centres = (edges[:-1] + edges[1:]) / 2
    return centres, ale


def _partial_dependence_1d(model, X, feature, grid):
    out = np.empty(len(grid))
    for i, v in enumerate(grid):
        Xc = X.copy(); Xc[:, feature] = v
        out[i] = _predict(model, Xc).mean()
    return out


def h_statistic(model, X, feature_a, feature_b, grid_resolution=10):
    """Friedman's H-statistic: how much of the (a, b) effect is INTERACTION.

    If features a and b act independently, their joint partial dependence equals
    the sum of their individual ones. The H-statistic measures the fraction of
    the joint effect's variance that is NOT explained by that sum::

        H^2 = sum (PD_ab - PD_a - PD_b)^2  /  sum PD_ab^2

    all on centred partial-dependence values. H near 0 means the features
    combine additively; H near 1 means the effect of one depends heavily on the
    other. It is the standard way to DISCOVER interactions to add to a GA2M.

    Friedman & Popescu (2008). Returns H in [0, 1].
    """
    X = check_array(X)
    ga = np.percentile(X[:, feature_a], np.linspace(0, 100, grid_resolution))
    gb = np.percentile(X[:, feature_b], np.linspace(0, 100, grid_resolution))

    def centre(v):
        return v - v.mean()

    pd_a = centre(_partial_dependence_1d(model, X, feature_a, ga))
    pd_b = centre(_partial_dependence_1d(model, X, feature_b, gb))
    # joint partial dependence on the grid
    pd_ab = np.empty((len(ga), len(gb)))
    for i, va in enumerate(ga):
        for j, vb in enumerate(gb):
            Xc = X.copy(); Xc[:, feature_a] = va; Xc[:, feature_b] = vb
            pd_ab[i, j] = _predict(model, Xc).mean()
    pd_ab = pd_ab - pd_ab.mean()
    interaction = pd_ab - pd_a[:, None] - pd_b[None, :]
    denom = np.sum(pd_ab ** 2)
    return float(np.sqrt(np.sum(interaction ** 2) / denom)) if denom > 0 else 0.0


__all__ = ["accumulated_local_effects", "h_statistic"]
