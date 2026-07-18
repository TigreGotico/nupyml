"""Predict each cell by its row's observed mean -- the no-factorization floor."""
import numpy as np


def solve(X_train, y_train, X_test):
    row_sum, row_cnt = {}, {}
    for (r, _), v in zip(X_train, y_train):
        row_sum[r] = row_sum.get(r, 0.0) + v
        row_cnt[r] = row_cnt.get(r, 0) + 1
    gm = float(np.mean(y_train))
    return np.array([row_sum.get(r, gm) / row_cnt[r] if r in row_cnt else gm
                     for r, _ in X_test])
