"""Additional classification, regression, ranking and clustering metrics."""
import numpy as np

from ..utils import column_or_1d


# ---------------------------------------------------------------------------
# classification
# ---------------------------------------------------------------------------

def balanced_accuracy_score(y_true, y_pred):
    from . import confusion_matrix
    cm = confusion_matrix(y_true, y_pred)
    per_class = np.diag(cm) / np.maximum(cm.sum(axis=1), 1)
    return float(per_class.mean())


def matthews_corrcoef(y_true, y_pred):
    """Correlation between predictions and truth, in [-1, +1].

    The most honest single number for imbalanced binary problems, because it
    uses all four cells of the confusion matrix. Accuracy and F1 can both look
    excellent while a whole class is ignored; MCC cannot -- it is only high when
    the model does well on both classes. 0 is chance, negative is
    anti-correlated.
    """
    from . import confusion_matrix
    C = confusion_matrix(y_true, y_pred).astype(np.float64)
    t = C.sum(axis=1)   # true counts
    p = C.sum(axis=0)   # predicted counts
    n = C.sum()
    cov_ytp = np.trace(C) * n - t @ p
    cov_yy = n ** 2 - t @ t
    cov_pp = n ** 2 - p @ p
    denom = np.sqrt(cov_yy * cov_pp)
    return float(cov_ytp / denom) if denom > 0 else 0.0


def cohen_kappa_score(y1, y2):
    from . import confusion_matrix
    C = confusion_matrix(y1, y2).astype(np.float64)
    n = C.sum()
    po = np.trace(C) / n
    pe = (C.sum(axis=0) @ C.sum(axis=1)) / n ** 2
    return float((po - pe) / (1 - pe)) if pe != 1 else 1.0


def brier_score_loss(y_true, y_proba, pos_label=None):
    """Mean squared error of predicted probabilities. LOWER is better.

    Unlike log loss, it is bounded and forgiving of confident mistakes -- a
    prediction of 0.0 for a true positive costs 1.0, not infinity. That makes it
    the more stable choice for comparing calibration, and it decomposes neatly
    into calibration and refinement terms.
    """
    y_true = column_or_1d(y_true)
    y_proba = column_or_1d(y_proba).astype(np.float64)
    classes = np.unique(y_true)
    if pos_label is None:
        pos_label = classes[-1]
    t = (y_true == pos_label).astype(np.float64)
    return float(np.mean((y_proba - t) ** 2))


def hinge_loss(y_true, pred_decision):
    y_true = column_or_1d(y_true)
    pred_decision = column_or_1d(pred_decision).astype(np.float64)
    classes = np.unique(y_true)
    t = np.where(y_true == classes[-1], 1.0, -1.0)
    return float(np.mean(np.maximum(0.0, 1.0 - t * pred_decision)))


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


def average_precision_score(y_true, y_score):
    """Area under the precision-recall curve. The right AUC under imbalance.

    ROC-AUC uses the false-positive rate, whose denominator is the number of
    negatives. When negatives vastly outnumber positives, even a large number of
    false alarms barely moves that rate, so ROC-AUC stays high while the model
    is unusable in practice. Precision's denominator is the number of things
    FLAGGED, which reacts immediately -- so this metric tells the truth on the
    rare-positive problems where it matters.

    Computed as a step-wise sum rather than trapezoid: interpolating a PR curve
    is over-optimistic, because the curve is not linear between operating
    points.
    """
    precision, recall, _ = precision_recall_curve(y_true, y_score)
    # step-wise integral: sum (R_n - R_{n+1}) * P_n  (recall is decreasing here)
    return float(-np.sum(np.diff(recall) * precision[:-1]))


def calibration_curve(y_true, y_proba, n_bins=5, strategy="uniform"):
    """Are the predicted probabilities honest?

    Bin the predictions, and in each bin compare the mean predicted probability
    with the observed frequency. A perfectly calibrated model lies on the
    diagonal: of the cases it called 70% likely, 70% happen.

    Deviations have a shape worth recognising. An S-curve below the diagonal at
    the top means overconfidence -- the classic naive Bayes or boosted-tree
    signature. ``strategy="quantile"`` puts equal COUNTS in each bin rather than
    equal widths, which avoids near-empty bins when predictions cluster at the
    extremes.
    """
    y_true = column_or_1d(y_true).astype(np.float64)
    y_proba = column_or_1d(y_proba).astype(np.float64)
    if strategy == "uniform":
        bins = np.linspace(0.0, 1.0, n_bins + 1)
    elif strategy == "quantile":
        bins = np.unique(np.percentile(y_proba, np.linspace(0, 100, n_bins + 1)))
    else:
        raise ValueError(f"Unknown strategy: {strategy!r}")
    ids = np.clip(np.searchsorted(bins[1:-1], y_proba, side="right"), 0,
                  len(bins) - 2)
    prob_true, prob_pred = [], []
    for b in range(len(bins) - 1):
        mask = ids == b
        if mask.any():
            prob_true.append(y_true[mask].mean())
            prob_pred.append(y_proba[mask].mean())
    return np.array(prob_true), np.array(prob_pred)


def top_k_accuracy_score(y_true, y_score, k=2, labels=None):
    y_true = column_or_1d(y_true)
    y_score = np.asarray(y_score, dtype=np.float64)
    if labels is None:
        labels = np.unique(y_true)
    labels = np.asarray(labels)
    idx = np.searchsorted(labels, y_true)
    topk = np.argsort(-y_score, axis=1)[:, :k]
    return float(np.mean([i in row for i, row in zip(idx, topk)]))


# ---------------------------------------------------------------------------
# regression
# ---------------------------------------------------------------------------

def median_absolute_error(y_true, y_pred):
    return float(np.median(np.abs(np.asarray(y_true, dtype=np.float64)
                                  - np.asarray(y_pred, dtype=np.float64))))


def explained_variance_score(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    var_res = np.var(y_true - y_pred)
    var_y = np.var(y_true)
    if var_y == 0:
        return 0.0 if var_res > 0 else 1.0
    return float(1 - var_res / var_y)


def mean_absolute_percentage_error(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    denom = np.maximum(np.abs(y_true), np.finfo(np.float64).eps)
    return float(np.mean(np.abs((y_true - y_pred) / denom)))


def max_error(y_true, y_pred):
    return float(np.max(np.abs(np.asarray(y_true, dtype=np.float64)
                               - np.asarray(y_pred, dtype=np.float64))))


def mean_poisson_deviance(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.maximum(np.asarray(y_pred, dtype=np.float64), 1e-12)
    with np.errstate(divide="ignore", invalid="ignore"):
        term = np.where(y_true > 0, y_true * np.log(y_true / y_pred), 0.0)
    return float(2 * np.mean(term - y_true + y_pred))


def mean_gamma_deviance(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.maximum(np.asarray(y_pred, dtype=np.float64), 1e-12)
    return float(2 * np.mean(np.log(y_pred / y_true) + y_true / y_pred - 1))


def mean_pinball_loss(y_true, y_pred, alpha=0.5):
    diff = np.asarray(y_true, dtype=np.float64) - np.asarray(y_pred, dtype=np.float64)
    return float(np.mean(np.maximum(alpha * diff, (alpha - 1) * diff)))


# ---------------------------------------------------------------------------
# ranking
# ---------------------------------------------------------------------------

def dcg_score(y_true, y_score, k=None):
    y_true = np.atleast_2d(np.asarray(y_true, dtype=np.float64))
    y_score = np.atleast_2d(np.asarray(y_score, dtype=np.float64))
    order = np.argsort(-y_score, axis=1)
    gains = np.take_along_axis(y_true, order, axis=1)
    if k is not None:
        gains = gains[:, :k]
    discounts = 1.0 / np.log2(np.arange(gains.shape[1]) + 2)
    return float((gains @ discounts).mean())


def ndcg_score(y_true, y_score, k=None):
    y_true = np.atleast_2d(np.asarray(y_true, dtype=np.float64))
    ideal = dcg_score(y_true, y_true, k=k)
    if ideal == 0:
        return 0.0
    return dcg_score(y_true, y_score, k=k) / ideal


# ---------------------------------------------------------------------------
# clustering
# ---------------------------------------------------------------------------

def _contingency(labels_true, labels_pred):
    labels_true = column_or_1d(labels_true)
    labels_pred = column_or_1d(labels_pred)
    _, ti = np.unique(labels_true, return_inverse=True)
    _, pi = np.unique(labels_pred, return_inverse=True)
    C = np.zeros((ti.max() + 1, pi.max() + 1))
    np.add.at(C, (ti, pi), 1)
    return C


def _entropy(counts):
    p = counts[counts > 0] / counts.sum()
    return float(-(p * np.log(p)).sum())


def mutual_info_score(labels_true, labels_pred):
    C = _contingency(labels_true, labels_pred)
    n = C.sum()
    outer = np.outer(C.sum(axis=1), C.sum(axis=0))
    nz = C > 0
    return float((C[nz] / n * (np.log(C[nz] * n) - np.log(outer[nz]))).sum())


def normalized_mutual_info_score(labels_true, labels_pred):
    """Shared information between two labellings, scaled to [0, 1].

    Mutual information asks how much knowing one labelling tells you about the
    other. It is invariant to permutations of the label names -- which is what
    you want when comparing clusterings, where "cluster 0" is arbitrary.

    Raw MI grows with the number of clusters, so normalising by the entropies
    makes it comparable. It does NOT correct for chance, though; for that see
    ``adjusted_rand_score``.
    """
    mi = mutual_info_score(labels_true, labels_pred)
    C = _contingency(labels_true, labels_pred)
    h1, h2 = _entropy(C.sum(axis=1)), _entropy(C.sum(axis=0))
    denom = (h1 + h2) / 2
    return float(mi / denom) if denom > 0 else 1.0


def homogeneity_score(labels_true, labels_pred):
    C = _contingency(labels_true, labels_pred)
    h_c = _entropy(C.sum(axis=1))
    if h_c == 0:
        return 1.0
    mi = mutual_info_score(labels_true, labels_pred)
    return float(mi / h_c)


def completeness_score(labels_true, labels_pred):
    return homogeneity_score(labels_pred, labels_true)


def v_measure_score(labels_true, labels_pred, beta=1.0):
    h = homogeneity_score(labels_true, labels_pred)
    c = completeness_score(labels_true, labels_pred)
    if h + c == 0:
        return 0.0
    return float((1 + beta) * h * c / (beta * h + c))


def fowlkes_mallows_score(labels_true, labels_pred):
    C = _contingency(labels_true, labels_pred)
    n = C.sum()
    tk = (C ** 2).sum() - n
    pk = (C.sum(axis=0) ** 2).sum() - n
    qk = (C.sum(axis=1) ** 2).sum() - n
    if pk == 0 or qk == 0:
        return 0.0
    return float(tk / np.sqrt(pk * qk))


def calinski_harabasz_score(X, labels):
    X = np.asarray(X, dtype=np.float64)
    labels = column_or_1d(labels)
    uniq = np.unique(labels)
    k, n = len(uniq), len(X)
    overall = X.mean(axis=0)
    between, within = 0.0, 0.0
    for c in uniq:
        Xc = X[labels == c]
        mean_c = Xc.mean(axis=0)
        between += len(Xc) * ((mean_c - overall) ** 2).sum()
        within += ((Xc - mean_c) ** 2).sum()
    if within == 0:
        return float("inf")
    return float(between / within * (n - k) / (k - 1))


def davies_bouldin_score(X, labels):
    from scipy.spatial.distance import cdist
    X = np.asarray(X, dtype=np.float64)
    labels = column_or_1d(labels)
    uniq = np.unique(labels)
    centroids = np.array([X[labels == c].mean(axis=0) for c in uniq])
    spreads = np.array([
        np.mean(np.linalg.norm(X[labels == c] - centroids[i], axis=1))
        for i, c in enumerate(uniq)])
    dists = cdist(centroids, centroids)
    np.fill_diagonal(dists, np.inf)
    ratios = (spreads[:, None] + spreads[None, :]) / dists
    return float(np.max(ratios, axis=1).mean())


__all__ = [
    "balanced_accuracy_score", "matthews_corrcoef", "cohen_kappa_score",
    "brier_score_loss", "hinge_loss", "precision_recall_curve",
    "average_precision_score", "calibration_curve", "top_k_accuracy_score",
    "median_absolute_error", "explained_variance_score",
    "mean_absolute_percentage_error", "max_error", "mean_poisson_deviance",
    "mean_gamma_deviance", "mean_pinball_loss", "dcg_score", "ndcg_score",
    "mutual_info_score", "normalized_mutual_info_score", "homogeneity_score",
    "completeness_score", "v_measure_score", "fowlkes_mallows_score",
    "calinski_harabasz_score", "davies_bouldin_score",
]
