"""Resamplers for imbalanced classification.

Each exposes ``fit_resample(X, y) -> (X_resampled, y_resampled)``, the
imbalanced-learn convention: it returns a NEW balanced dataset rather than
transforming in place, because resampling changes the number of rows.
"""
import numpy as np

from ..base import BaseEstimator
from ..utils import check_X_y, check_random_state


def _minority_majority(y):
    classes, counts = np.unique(y, return_counts=True)
    return classes[np.argmin(counts)], classes[np.argmax(counts)], dict(
        zip(classes, counts))


class RandomOverSampler(BaseEstimator):
    """Duplicate minority points until the classes balance.

    The simplest fix, and its flaw is exact: it adds no new information, only
    copies. A learner can then memorise the duplicated points -- an outlier
    minority sample repeated ten times becomes ten reasons to overfit to it. Fast
    and sometimes fine; ``SMOTE`` exists because interpolating beats copying.
    """

    def __init__(self, random_state=None):
        self.random_state = random_state

    def fit_resample(self, X, y):
        X, y = check_X_y(X, y)
        rng = check_random_state(self.random_state)
        classes, counts = np.unique(y, return_counts=True)
        n_max = counts.max()
        parts_X, parts_y = [X], [y]
        for c in classes:
            idx = np.where(y == c)[0]
            deficit = n_max - len(idx)
            if deficit > 0:
                extra = rng.choice(idx, size=deficit, replace=True)
                parts_X.append(X[extra])
                parts_y.append(y[extra])
        return np.vstack(parts_X), np.concatenate(parts_y)


class RandomUnderSampler(BaseEstimator):
    """Drop majority points until the classes balance.

    The mirror image: it discards real data, which is wasteful when the majority
    is informative -- and it can throw away exactly the majority points near the
    boundary that define it. Cheap, and useful when the majority is huge and
    redundant, but ``TomekLinks``/``NearMiss`` remove points by RELEVANCE rather
    than at random.
    """

    def __init__(self, random_state=None):
        self.random_state = random_state

    def fit_resample(self, X, y):
        X, y = check_X_y(X, y)
        rng = check_random_state(self.random_state)
        classes, counts = np.unique(y, return_counts=True)
        n_min = counts.min()
        keep = []
        for c in classes:
            idx = np.where(y == c)[0]
            keep.append(rng.choice(idx, size=n_min, replace=False))
        keep = np.concatenate(keep)
        return X[keep], y[keep]


class SMOTE(BaseEstimator):
    """Synthetic Minority Over-sampling: interpolate, do not copy.

    THE IDEA
    --------
    For a minority point, pick one of its ``k`` minority neighbours and place a
    new synthetic point at a RANDOM spot on the line between them::

        synthetic = point + random_fraction * (neighbour - point)

    So instead of duplicating existing points, SMOTE fills in the CONVEX region
    the minority class occupies. The classifier sees a denser, smoother minority
    region and draws a fairer boundary, without the overfitting that exact copies
    invite.

    WHERE IT GOES WRONG
    -------------------
    Interpolation assumes the space BETWEEN two minority points is also minority
    -- true for a convex cluster, false when the class is in scattered pieces. A
    synthetic point on a line spanning a majority region lands inside the wrong
    class, manufacturing noise exactly on the boundary. SMOTE is strongest when
    the minority is roughly convex; ``BorderlineSMOTE`` and ``ADASYN`` are refinements
    that steer where the synthesis happens.

    Chawla, Bowyer, Hall & Kegelmeyer (2002).
    """

    def __init__(self, k_neighbors=5, random_state=None):
        self.k_neighbors = k_neighbors
        self.random_state = random_state

    def _neighbors(self, X_min):
        from scipy.spatial.distance import cdist
        d = cdist(X_min, X_min)
        np.fill_diagonal(d, np.inf)                # a point is not its own neighbour
        k = min(self.k_neighbors, len(X_min) - 1)
        return np.argsort(d, axis=1)[:, :k]

    def fit_resample(self, X, y):
        X, y = check_X_y(X, y)
        rng = check_random_state(self.random_state)
        minority, majority, counts = _minority_majority(y)
        X_min = X[y == minority]
        deficit = counts[majority] - counts[minority]
        if deficit <= 0 or len(X_min) < 2:
            return X, y

        nn = self._neighbors(X_min)
        synth = np.empty((deficit, X.shape[1]))
        for s in range(deficit):
            i = rng.randint(len(X_min))
            j = nn[i, rng.randint(nn.shape[1])]     # a random minority neighbour
            gap = rng.uniform()                     # a random point on the segment
            synth[s] = X_min[i] + gap * (X_min[j] - X_min[i])

        X_new = np.vstack([X, synth])
        y_new = np.concatenate([y, np.full(deficit, minority)])
        return X_new, y_new


class BorderlineSMOTE(SMOTE):
    """SMOTE that synthesises ONLY near the decision boundary.

    Interior minority points are already safely classified -- adding synthetic
    copies among them helps nothing. The hard cases are the minority points on the
    BORDER, surrounded by majority neighbours, where the boundary is actually
    contested. Borderline-SMOTE finds those "in danger" points (more than half
    their neighbours are majority, but not ALL, which would make them pure noise)
    and interpolates only from them. Effort goes where the classifier is
    struggling, not where it has already won.

    Han, Wang & Mao (2005).
    """

    def fit_resample(self, X, y):
        from scipy.spatial.distance import cdist
        X, y = check_X_y(X, y)
        rng = check_random_state(self.random_state)
        minority, majority, counts = _minority_majority(y)
        X_min = X[y == minority]
        deficit = counts[majority] - counts[minority]
        if deficit <= 0 or len(X_min) < 2:
            return X, y

        # which minority points are "in danger": majority-dominated neighbourhoods
        k = min(self.k_neighbors, len(X) - 1)
        d_all = cdist(X_min, X)
        danger = []
        for i in range(len(X_min)):
            order = np.argsort(d_all[i])[1:k + 1]
            maj_frac = np.mean(y[order] == majority)
            # in danger, but not hopeless noise (all neighbours majority)
            if 0.5 <= maj_frac < 1.0:
                danger.append(i)
        if not danger:
            return super().fit_resample(X, y)       # fall back to plain SMOTE

        nn = self._neighbors(X_min)
        synth = np.empty((deficit, X.shape[1]))
        for s in range(deficit):
            i = danger[rng.randint(len(danger))]    # synthesise from danger points
            j = nn[i, rng.randint(nn.shape[1])]
            synth[s] = X_min[i] + rng.uniform() * (X_min[j] - X_min[i])
        return (np.vstack([X, synth]),
                np.concatenate([y, np.full(deficit, minority)]))


class ADASYN(BaseEstimator):
    """Adaptive synthesis: MORE synthetic points where the minority is hard.

    THE REFINEMENT OVER SMOTE
    -------------------------
    SMOTE spreads synthetic points evenly across the minority. ADASYN spreads them
    by DIFFICULTY: for each minority point it counts how many of its neighbours are
    majority, and generates synthetic points in proportion. Points deep in
    majority territory -- the ones the classifier gets wrong -- get many synthetic
    companions; points safely inside the minority get few.

    The effect is a boundary that shifts adaptively toward the hard region, since
    that is where the new density is added. It is SMOTE with an attention
    mechanism: the same interpolation, aimed by where the learning is failing.

    He, Bai, Garcia & Li (2008).
    """

    def __init__(self, k_neighbors=5, random_state=None):
        self.k_neighbors = k_neighbors
        self.random_state = random_state

    def fit_resample(self, X, y):
        from scipy.spatial.distance import cdist
        X, y = check_X_y(X, y)
        rng = check_random_state(self.random_state)
        minority, majority, counts = _minority_majority(y)
        min_idx = np.where(y == minority)[0]
        X_min = X[min_idx]
        deficit = counts[majority] - counts[minority]
        if deficit <= 0 or len(X_min) < 2:
            return X, y

        k = min(self.k_neighbors, len(X) - 1)
        d_all = cdist(X_min, X)
        # hardness r_i = fraction of majority neighbours; normalised to a
        # distribution that says how many synthetics each point deserves
        r = np.empty(len(X_min))
        for i in range(len(X_min)):
            order = np.argsort(d_all[i])[1:k + 1]
            r[i] = np.mean(y[order] == majority)
        if r.sum() == 0:
            r = np.ones(len(X_min))
        g = np.round(r / r.sum() * deficit).astype(int)   # synthetics per point

        d_min = cdist(X_min, X_min)
        np.fill_diagonal(d_min, np.inf)
        nn = np.argsort(d_min, axis=1)[:, :min(k, len(X_min) - 1)]

        synth = []
        for i in range(len(X_min)):
            for _ in range(g[i]):
                j = nn[i, rng.randint(nn.shape[1])]
                synth.append(X_min[i] + rng.uniform() * (X_min[j] - X_min[i]))
        if not synth:
            return X, y
        synth = np.array(synth)
        return (np.vstack([X, synth]),
                np.concatenate([y, np.full(len(synth), minority)]))


class TomekLinks(BaseEstimator):
    """Clean the boundary by removing majority points in Tomek links.

    A TOMEK LINK is a pair of points from opposite classes that are each other's
    nearest neighbour -- two points from different classes with nothing between
    them. Such a pair straddles the boundary and blurs it. Removing the MAJORITY
    member of each link sharpens the class separation without touching the rare
    minority.

    Unlike random undersampling, this removes points by RELEVANCE: only those
    sitting right on the contested boundary, never the informative interior. It is
    usually a cleanup pass AFTER oversampling (SMOTE + Tomek is a standard combo),
    tidying the synthetic points that landed across the line.

    Tomek (1976).
    """

    def fit_resample(self, X, y):
        from scipy.spatial.distance import cdist
        X, y = check_X_y(X, y)
        _, majority, _ = _minority_majority(y)
        d = cdist(X, X)
        np.fill_diagonal(d, np.inf)
        nn = np.argmin(d, axis=1)                   # each point's nearest neighbour

        remove = set()
        for i in range(len(X)):
            j = nn[i]
            # a Tomek link: mutual nearest neighbours of opposite classes
            if nn[j] == i and y[i] != y[j]:
                # drop the majority member, keeping the scarce minority
                remove.add(i if y[i] == majority else j)
        keep = np.array([i for i in range(len(X)) if i not in remove])
        return X[keep], y[keep]


class NearMiss(BaseEstimator):
    """Undersample the majority by keeping points CLOSEST to the minority.

    Rather than dropping majority points at random, NearMiss keeps the majority
    points nearest the minority class -- the ones that actually define the
    boundary. Version 1 keeps majority points with the smallest average distance
    to their nearest minority neighbours. The intent is a majority sample that
    still describes the frontier, so the boundary is not lost with the volume; the
    risk is that it hugs the minority so closely it becomes noise-sensitive, which
    is why it is used with care.

    Mani & Zhang (2003).
    """

    def __init__(self, n_neighbors=3):
        self.n_neighbors = n_neighbors

    def fit_resample(self, X, y):
        from scipy.spatial.distance import cdist
        X, y = check_X_y(X, y)
        minority, majority, counts = _minority_majority(y)
        maj_idx = np.where(y == majority)[0]
        min_X = X[y == minority]

        # average distance from each majority point to its nearest minority points
        d = cdist(X[maj_idx], min_X)
        k = min(self.n_neighbors, min_X.shape[0])
        avg = np.sort(d, axis=1)[:, :k].mean(axis=1)
        # keep the majority points CLOSEST to the minority -- the boundary ones
        keep_maj = maj_idx[np.argsort(avg)[:counts[minority]]]

        keep = np.concatenate([np.where(y == minority)[0], keep_maj])
        return X[keep], y[keep]


__all__ = ["RandomOverSampler", "RandomUnderSampler", "SMOTE", "ADASYN",
           "TomekLinks", "NearMiss", "BorderlineSMOTE"]
