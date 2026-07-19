"""precision_recall_curve"""
import numpy as np
from ..utils import column_or_1d


def precision_recall_curve(y_true, y_score):
    y_true = column_or_1d(y_true)
    y_score = column_or_1d(y_score).astype(np.float64)
    classes = np.unique(y_true)
    y_bin = (y_true == classes[-1]).astype(np.float64)
    order = np.argsort(-y_score, kind="stable")
    y_bin = y_bin[order]
    y_score = y_score[order]
    distinct = np.where(np.diff(y_score))[0]
    threshold_idx = np.r_[distinct, len(y_score) - 1]
    tps = np.cumsum(y_bin)[threshold_idx]
    fps = 1 + threshold_idx - tps
    precision = tps / (tps + fps)
    recall = tps / tps[-1] if tps[-1] > 0 else np.ones_like(tps)
    # reverse so recall is decreasing, append the (1, 0) endpoint
    precision = np.r_[precision[::-1], 1.0]
    recall = np.r_[recall[::-1], 0.0]
    thresholds = y_score[threshold_idx][::-1]
    return precision, recall, thresholds


__all__ = ["precision_recall_curve"]
