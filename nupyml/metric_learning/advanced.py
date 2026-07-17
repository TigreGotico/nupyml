"""More supervised metric learners: ITML, LFDA, RCA.

The base module has NCA and LMNN. These add three more takes on "learn a
Mahalanobis distance that respects the labels", each with a different objective.
"""
import numpy as np

from ..base import BaseEstimator, TransformerMixin, check_is_fitted
from ..utils import check_X_y, check_array, check_random_state


class ITML(BaseEstimator, TransformerMixin):
    """Information-Theoretic Metric Learning: stay close to a prior metric.

    THE IDEA
    --------
    Learn a Mahalanobis matrix ``M`` that satisfies pairwise CONSTRAINTS --
    similar pairs should be within a small distance, dissimilar pairs beyond a
    large one -- while staying as CLOSE AS POSSIBLE to a prior matrix (the
    identity, i.e. plain Euclidean). "Close" is measured by the LogDet divergence
    between the two matrices, which is why it is "information-theoretic": it is the
    KL divergence between the two Gaussians those matrices define.

    WHY THE PRIOR MATTERS
    ---------------------
    Anchoring to a prior is regularisation: with few constraints, ``M`` stays near
    Euclidean and does not overfit the handful of pairs it saw; each constraint
    nudges it just enough to satisfy that pair. The LogDet divergence also keeps
    ``M`` positive-definite automatically (a real distance) with no explicit
    projection -- an elegant side effect of the divergence's form. Solved by
    cyclic Bregman projections, satisfying one constraint at a time.

    Davis, Kulis, Jain, Sra & Dhillon (2007).
    """

    def __init__(self, gamma=1.0, n_constraints=200, max_iter=1000,
                 random_state=None):
        self.gamma = gamma              # slack: how hard to enforce constraints
        self.n_constraints = n_constraints
        self.max_iter = max_iter
        self.random_state = random_state

    def fit(self, X, y):
        X, y = check_X_y(X, y)
        rng = check_random_state(self.random_state)
        n, d = X.shape
        M = np.eye(d)                   # the prior: Euclidean

        # bounds: similar pairs within the 5th percentile of distances, dissimilar
        # beyond the 95th -- data-driven targets
        from scipy.spatial.distance import pdist
        dists = pdist(X)
        lo, hi = np.percentile(dists, 5) ** 2, np.percentile(dists, 95) ** 2

        # sample same-label (similar) and different-label (dissimilar) pairs
        constraints = []
        for _ in range(self.n_constraints):
            i, j = rng.randint(n), rng.randint(n)
            if i == j:
                continue
            target = lo if y[i] == y[j] else hi
            sign = 1 if y[i] == y[j] else -1     # similar: shrink; else: grow
            constraints.append((i, j, target, sign))

        lam = np.zeros(len(constraints))
        for it in range(self.max_iter):
            c = it % len(constraints)
            i, j, target, sign = constraints[c]
            diff = X[i] - X[j]
            dist = diff @ M @ diff
            if dist < 1e-10:
                continue
            # Bregman projection onto this one constraint, LogDet-style
            alpha = min(lam[c], self.gamma / 2 *
                        (1.0 / dist - sign / target))
            beta = sign * alpha / (1 - sign * alpha * dist)
            lam[c] -= alpha
            Mv = M @ diff
            M = M + beta * np.outer(Mv, Mv)
        self.M_ = M
        # a transform L with L'L = M, via the matrix square root
        vals, vecs = np.linalg.eigh(M)
        self.components_ = (vecs * np.sqrt(np.maximum(vals, 0))) @ vecs.T
        self.n_features_out_ = d
        return self

    def transform(self, X):
        check_is_fitted(self, "components_")
        return check_array(X) @ self.components_.T


class LFDA(BaseEstimator, TransformerMixin):
    """Local Fisher Discriminant Analysis: LDA that respects MULTIMODAL classes.

    THE PROBLEM WITH PLAIN LDA
    --------------------------
    Fisher's LDA maximises between-class scatter over within-class scatter,
    assuming each class is a single Gaussian blob. When a class is MULTIMODAL --
    two separate clusters that happen to share a label -- LDA tries to collapse
    them into one and mangles the projection. LFDA fixes this by making the
    scatter matrices LOCAL: pairs are weighted by their affinity (nearby points
    count, distant ones do not), so a class made of two clusters keeps its two
    clusters instead of being forced together.

    It is the metric-learning bridge between LDA (global, supervised) and
    dimensionality reduction that preserves local structure -- solving the same
    generalised eigenproblem as LDA but with locally-weighted scatter.

    Sugiyama (2007).
    """

    def __init__(self, n_components=None, n_neighbors=7):
        self.n_components = n_components
        self.n_neighbors = n_neighbors

    def fit(self, X, y):
        from scipy.spatial.distance import cdist
        X, y = check_X_y(X, y)
        n, d = X.shape
        k = min(self.n_neighbors, n - 1)
        # local scaling per point: distance to its k-th neighbour (Zelnik-Manor)
        D = cdist(X, X)
        sigma = np.sort(D, axis=1)[:, k]
        affinity = np.exp(-D ** 2 / (sigma[:, None] * sigma[None, :] + 1e-12))

        Sw = np.zeros((d, d))       # local within-class scatter
        Sb = np.zeros((d, d))       # local between-class scatter
        for i in range(n):
            for j in range(n):
                diff = (X[i] - X[j])[:, None]
                outer = diff @ diff.T
                if y[i] == y[j]:
                    nc = np.sum(y == y[i])
                    Sw += affinity[i, j] / nc * outer
                    Sb += affinity[i, j] * (1.0 / n - 1.0 / nc) * outer
                else:
                    Sb += affinity[i, j] / n * outer

        # generalised eigenproblem Sb w = lambda Sw w, as in LDA
        Sw += 1e-6 * np.eye(d)
        vals, vecs = np.linalg.eig(np.linalg.solve(Sw, Sb))
        vals, vecs = np.real(vals), np.real(vecs)
        order = np.argsort(vals)[::-1]
        n_comp = self.n_components or d
        self.components_ = vecs[:, order[:n_comp]].T
        self.n_features_out_ = n_comp
        return self

    def transform(self, X):
        check_is_fitted(self, "components_")
        return check_array(X) @ self.components_.T


class RCA(BaseEstimator, TransformerMixin):
    """Relevant Component Analysis: whiten away WITHIN-chunk variation.

    THE IDEA (AND ITS WEAK SUPERVISION)
    -----------------------------------
    RCA needs only "chunklets" -- small groups known to share a label, without
    knowing WHICH label (weaker supervision than full labels). Its insight:
    variation WITHIN a chunklet is irrelevant (those points are the same class),
    so a good metric should SHRINK it. RCA computes the average within-chunklet
    covariance and whitens by its inverse -- stretching the directions where
    same-class points barely vary, so that after the transform those directions
    stop dominating the distance.

    It is essentially "whiten out the noise the labels tell you to ignore", the
    simplest and fastest metric learner here, and it works from side-information
    (these belong together) rather than explicit labels.

    Bar-Hillel, Hertz, Shental & Weinshall (2005).
    """

    def __init__(self, n_components=None):
        self.n_components = n_components

    def fit(self, X, y):
        X, y = check_X_y(X, y)
        d = X.shape[1]
        # each label group is a chunklet; average their within-covariances
        C = np.zeros((d, d))
        total = 0
        for c in np.unique(y):
            Xc = X[y == c]
            if len(Xc) < 2:
                continue
            centered = Xc - Xc.mean(axis=0)
            C += centered.T @ centered
            total += len(Xc)
        C = C / max(total, 1) + 1e-6 * np.eye(d)
        # whiten by C^-1/2: directions of large within-chunklet variance shrink
        vals, vecs = np.linalg.eigh(C)
        W = (vecs / np.sqrt(vals)) @ vecs.T
        n_comp = self.n_components or d
        if n_comp < d:
            W = W[:n_comp]
        self.components_ = W
        self.n_features_out_ = W.shape[0]
        return self

    def transform(self, X):
        check_is_fitted(self, "components_")
        return check_array(X) @ self.components_.T


__all__ = ["ITML", "LFDA", "RCA"]
