"""Convergent cross mapping: nonlinear causality from reconstructed attractors."""
import numpy as np


def convergent_cross_mapping(x, y, embed_dim=3, tau=1, lib_sizes=None):
    """Detect nonlinear CAUSALITY from reconstructed attractors (Sugihara, 2012).

    Granger causality assumes linearity and separability; many real systems (ecology,
    physiology) are neither. Convergent cross mapping tests causality through Takens'
    theorem: if X drives Y, then the history of Y contains a shadow of X, so X should be
    RECOVERABLE from Y's reconstructed attractor -- and, crucially, the recovery should
    IMPROVE (converge) as more data fills in the attractor. It reconstructs each series
    by time-delay embedding, cross-predicts one from the other's neighbours, and reports
    the skill vs library size. Returns cross-map skill for each library size.
    """
    x = np.asarray(x, float); y = np.asarray(y, float)
    if lib_sizes is None:
        lib_sizes = [20, len(x) // 2, len(x) - embed_dim * tau - 1]

    def embed(s):
        n = len(s) - (embed_dim - 1) * tau
        return np.array([[s[i + j * tau] for j in range(embed_dim)]
                         for i in range(n)])

    My = embed(y)                                         # Y's shadow manifold
    target = x[(embed_dim - 1) * tau:]                    # aligned X values
    from scipy.spatial.distance import cdist
    skills = []
    for L in lib_sizes:
        L = min(L, len(My))
        lib = My[:L]; lib_t = target[:L]
        D = cdist(My, lib)
        pred = np.empty(len(My))
        for i in range(len(My)):
            nn = np.argsort(D[i])[:embed_dim + 1]
            d = D[i, nn]
            w = np.exp(-d / (d[0] + 1e-9)); w /= w.sum()  # cross-map from neighbours
            pred[i] = w @ lib_t[nn]
        skills.append(np.corrcoef(pred, target)[0, 1])
    return np.array(lib_sizes), np.array(skills)


__all__ = ["convergent_cross_mapping"]
