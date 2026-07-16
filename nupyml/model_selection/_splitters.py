"""Additional cross-validation splitters."""
import numpy as np

from ..utils import check_random_state, column_or_1d


class GroupKFold:
    """K-fold with non-overlapping groups, balanced by group size."""

    def __init__(self, n_splits=5):
        self.n_splits = n_splits

    def split(self, X, y=None, groups=None):
        if groups is None:
            raise ValueError("GroupKFold requires groups")
        groups = column_or_1d(np.asarray(groups))
        uniq, counts = np.unique(groups, return_counts=True)
        if len(uniq) < self.n_splits:
            raise ValueError("Cannot have more splits than groups")
        # assign largest groups first to the lightest fold
        order = np.argsort(-counts)
        fold_sizes = np.zeros(self.n_splits)
        group_fold = {}
        for gi in order:
            f = int(np.argmin(fold_sizes))
            group_fold[uniq[gi]] = f
            fold_sizes[f] += counts[gi]
        folds = np.array([group_fold[g] for g in groups])
        for k in range(self.n_splits):
            yield np.where(folds != k)[0], np.where(folds == k)[0]

    def get_n_splits(self, X=None, y=None, groups=None):
        return self.n_splits


class StratifiedGroupKFold(GroupKFold):
    """Greedy stratified assignment of whole groups to folds."""

    def split(self, X, y=None, groups=None):
        if groups is None or y is None:
            raise ValueError("StratifiedGroupKFold requires y and groups")
        y = column_or_1d(y)
        groups = column_or_1d(np.asarray(groups))
        classes, y_idx = np.unique(y, return_inverse=True)
        uniq = np.unique(groups)
        # per-group class histograms
        hist = {g: np.bincount(y_idx[groups == g], minlength=len(classes))
                for g in uniq}
        order = sorted(uniq, key=lambda g: -hist[g].sum())
        fold_hist = np.zeros((self.n_splits, len(classes)))
        group_fold = {}
        total = np.bincount(y_idx, minlength=len(classes)).astype(float)
        for g in order:
            # fold whose class distribution stays closest to global after adding g
            best_f, best_dev = 0, np.inf
            for f in range(self.n_splits):
                trial = fold_hist[f] + hist[g]
                dev = np.abs(trial / max(trial.sum(), 1) - total / total.sum()).sum() \
                    + trial.sum() / total.sum()
                if dev < best_dev:
                    best_f, best_dev = f, dev
            group_fold[g] = best_f
            fold_hist[best_f] += hist[g]
        folds = np.array([group_fold[g] for g in groups])
        for k in range(self.n_splits):
            yield np.where(folds != k)[0], np.where(folds == k)[0]


class TimeSeriesSplit:
    def __init__(self, n_splits=5, max_train_size=None, test_size=None, gap=0):
        self.n_splits = n_splits
        self.max_train_size = max_train_size
        self.test_size = test_size
        self.gap = gap

    def split(self, X, y=None, groups=None):
        n = len(X)
        test_size = self.test_size or n // (self.n_splits + 1)
        starts = [n - (self.n_splits - i) * test_size for i in range(self.n_splits)]
        for start in starts:
            if start - self.gap <= 0:
                raise ValueError("Too many splits for data size")
            train_end = start - self.gap
            train_start = max(0, train_end - self.max_train_size) \
                if self.max_train_size else 0
            yield (np.arange(train_start, train_end),
                   np.arange(start, min(start + test_size, n)))

    def get_n_splits(self, X=None, y=None, groups=None):
        return self.n_splits


class ShuffleSplit:
    def __init__(self, n_splits=10, test_size=0.1, train_size=None,
                 random_state=None):
        self.n_splits = n_splits
        self.test_size = test_size
        self.train_size = train_size
        self.random_state = random_state

    def _sizes(self, n):
        n_test = int(np.ceil(n * self.test_size)) \
            if isinstance(self.test_size, float) else int(self.test_size)
        if self.train_size is None:
            n_train = n - n_test
        else:
            n_train = int(np.floor(n * self.train_size)) \
                if isinstance(self.train_size, float) else int(self.train_size)
        return n_train, n_test

    def split(self, X, y=None, groups=None):
        n = len(X)
        n_train, n_test = self._sizes(n)
        rng = check_random_state(self.random_state)
        for _ in range(self.n_splits):
            perm = rng.permutation(n)
            yield perm[n_test:n_test + n_train], perm[:n_test]

    def get_n_splits(self, X=None, y=None, groups=None):
        return self.n_splits


class StratifiedShuffleSplit(ShuffleSplit):
    def split(self, X, y=None, groups=None):
        y = column_or_1d(y)
        n = len(y)
        n_train, n_test = self._sizes(n)
        rng = check_random_state(self.random_state)
        classes = np.unique(y)
        for _ in range(self.n_splits):
            train_idx, test_idx = [], []
            for c in classes:
                c_idx = rng.permutation(np.where(y == c)[0])
                k_test = int(round(len(c_idx) * n_test / n))
                k_train = int(round(len(c_idx) * n_train / n))
                test_idx.append(c_idx[:k_test])
                train_idx.append(c_idx[k_test:k_test + k_train])
            yield np.concatenate(train_idx), np.concatenate(test_idx)


class RepeatedKFold:
    def __init__(self, n_splits=5, n_repeats=10, random_state=None):
        self.n_splits = n_splits
        self.n_repeats = n_repeats
        self.random_state = random_state

    def split(self, X, y=None, groups=None):
        from . import KFold
        rng = check_random_state(self.random_state)
        for _ in range(self.n_repeats):
            kf = KFold(n_splits=self.n_splits, shuffle=True,
                       random_state=rng.randint(0, 2 ** 31 - 1))
            yield from kf.split(X, y)

    def get_n_splits(self, X=None, y=None, groups=None):
        return self.n_splits * self.n_repeats


class RepeatedStratifiedKFold(RepeatedKFold):
    def split(self, X, y=None, groups=None):
        from . import StratifiedKFold
        rng = check_random_state(self.random_state)
        for _ in range(self.n_repeats):
            kf = StratifiedKFold(n_splits=self.n_splits, shuffle=True,
                                 random_state=rng.randint(0, 2 ** 31 - 1))
            yield from kf.split(X, y)


class LeavePOut:
    def __init__(self, p=2):
        self.p = p

    def split(self, X, y=None, groups=None):
        import itertools
        n = len(X)
        all_idx = np.arange(n)
        for test in itertools.combinations(range(n), self.p):
            test = np.array(test)
            yield np.setdiff1d(all_idx, test), test

    def get_n_splits(self, X, y=None, groups=None):
        from math import comb
        return comb(len(X), self.p)


class PredefinedSplit:
    def __init__(self, test_fold):
        self.test_fold = np.asarray(test_fold)

    def split(self, X=None, y=None, groups=None):
        for k in np.unique(self.test_fold[self.test_fold != -1]):
            yield (np.where(self.test_fold != k)[0],
                   np.where(self.test_fold == k)[0])

    def get_n_splits(self, X=None, y=None, groups=None):
        return len(np.unique(self.test_fold[self.test_fold != -1]))


__all__ = ["GroupKFold", "StratifiedGroupKFold", "TimeSeriesSplit",
           "ShuffleSplit", "StratifiedShuffleSplit", "RepeatedKFold",
           "RepeatedStratifiedKFold", "LeavePOut", "PredefinedSplit"]
