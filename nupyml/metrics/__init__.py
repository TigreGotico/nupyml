"""Model evaluation metrics."""
import numpy as np

from ..utils import column_or_1d, check_consistent_length


# --------------------------------------------------------------------------
# classification
# --------------------------------------------------------------------------

def accuracy_score(y_true, y_pred, sample_weight=None):
    y_true, y_pred = column_or_1d(y_true), column_or_1d(y_pred)
    check_consistent_length(y_true, y_pred)
    if sample_weight is None:
        return float(np.mean(y_true == y_pred))
    w = np.asarray(sample_weight, dtype=np.float64)
    return float(np.average(y_true == y_pred, weights=w))


def confusion_matrix(y_true, y_pred, labels=None, sample_weight=None):
    y_true, y_pred = column_or_1d(y_true), column_or_1d(y_pred)
    if labels is None:
        labels = np.unique(np.concatenate([y_true, y_pred]))
    labels = np.asarray(labels)
    n = len(labels)
    idx = {l: i for i, l in enumerate(labels)}
    w = np.ones(len(y_true)) if sample_weight is None \
        else np.asarray(sample_weight, dtype=np.float64)
    cm = np.zeros((n, n), dtype=np.int64 if sample_weight is None else np.float64)
    for t, p, wi in zip(y_true, y_pred, w):
        cm[idx[t], idx[p]] += wi
    return cm


def _prf_counts(y_true, y_pred, labels, sample_weight=None):
    cm = confusion_matrix(y_true, y_pred, labels=labels,
                          sample_weight=sample_weight)
    tp = np.diag(cm).astype(float)
    fp = cm.sum(axis=0) - tp
    fn = cm.sum(axis=1) - tp
    support = cm.sum(axis=1)
    return tp, fp, fn, support


def _average_prf(num, den, average, support):
    with np.errstate(divide="ignore", invalid="ignore"):
        per_class = np.where(den > 0, num / den, 0.0)
    if average is None:
        return per_class
    if average == "macro":
        return float(per_class.mean())
    if average == "micro":
        return float(num.sum() / den.sum()) if den.sum() > 0 else 0.0
    if average == "weighted":
        total = support.sum()
        return float((per_class * support).sum() / total) if total > 0 else 0.0
    raise ValueError(f"Unknown average: {average!r}")


def precision_score(y_true, y_pred, average="binary", labels=None,
                    sample_weight=None):
    return _prf(y_true, y_pred, average, labels, "precision", sample_weight)


def recall_score(y_true, y_pred, average="binary", labels=None,
                 sample_weight=None):
    return _prf(y_true, y_pred, average, labels, "recall", sample_weight)


def f1_score(y_true, y_pred, average="binary", labels=None, sample_weight=None):
    return _prf(y_true, y_pred, average, labels, "f1", sample_weight)


def _prf(y_true, y_pred, average, labels, which, sample_weight=None):
    y_true, y_pred = column_or_1d(y_true), column_or_1d(y_pred)
    if labels is None:
        labels = np.unique(np.concatenate([y_true, y_pred]))
    tp, fp, fn, support = _prf_counts(y_true, y_pred, labels, sample_weight)
    if average == "binary":
        if len(labels) != 2:
            raise ValueError("average='binary' requires binary targets")
        i = 1  # positive class = greater label
        p = tp[i] / (tp[i] + fp[i]) if tp[i] + fp[i] > 0 else 0.0
        r = tp[i] / (tp[i] + fn[i]) if tp[i] + fn[i] > 0 else 0.0
        if which == "precision":
            return float(p)
        if which == "recall":
            return float(r)
        return float(2 * p * r / (p + r)) if p + r > 0 else 0.0
    if which == "precision":
        return _average_prf(tp, tp + fp, average, support)
    if which == "recall":
        return _average_prf(tp, tp + fn, average, support)
    # f1 = 2tp / (2tp + fp + fn), averaged
    return _average_prf(2 * tp, 2 * tp + fp + fn, average, support)


def log_loss(y_true, y_proba, labels=None, eps=1e-15, sample_weight=None):
    y_true = column_or_1d(y_true)
    y_proba = np.asarray(y_proba, dtype=np.float64)
    if labels is None:
        labels = np.unique(y_true)
    labels = np.asarray(labels)
    if y_proba.ndim == 1:
        y_proba = np.column_stack([1 - y_proba, y_proba])
    y_proba = np.clip(y_proba, eps, 1 - eps)
    y_proba = y_proba / y_proba.sum(axis=1, keepdims=True)
    idx = np.searchsorted(labels, y_true)
    picked = np.log(y_proba[np.arange(len(y_true)), idx])
    return float(-np.average(picked, weights=sample_weight))


def roc_curve(y_true, y_score):
    y_true = column_or_1d(y_true)
    y_score = column_or_1d(y_score).astype(np.float64)
    classes = np.unique(y_true)
    if len(classes) != 2:
        raise ValueError("roc_curve requires binary targets")
    y_bin = (y_true == classes[1]).astype(np.float64)
    order = np.argsort(-y_score, kind="stable")
    y_bin, y_score = y_bin[order], y_score[order]
    distinct = np.where(np.diff(y_score))[0]
    threshold_idx = np.r_[distinct, len(y_score) - 1]
    tps = np.cumsum(y_bin)[threshold_idx]
    fps = 1 + threshold_idx - tps
    tps = np.r_[0.0, tps]
    fps = np.r_[0.0, fps]
    thresholds = np.r_[np.inf, y_score[threshold_idx]]
    P = y_bin.sum()
    N = len(y_bin) - P
    tpr = tps / P if P > 0 else np.zeros_like(tps)
    fpr = fps / N if N > 0 else np.zeros_like(fps)
    return fpr, tpr, thresholds


def roc_auc_score(y_true, y_score):
    fpr, tpr, _ = roc_curve(y_true, y_score)
    trapezoid = getattr(np, "trapezoid", None) or np.trapz
    return float(trapezoid(tpr, fpr))


def classification_report(y_true, y_pred, labels=None, digits=2):
    y_true, y_pred = column_or_1d(y_true), column_or_1d(y_pred)
    if labels is None:
        labels = np.unique(np.concatenate([y_true, y_pred]))
    tp, fp, fn, support = _prf_counts(y_true, y_pred, labels)
    p = _average_prf(tp, tp + fp, None, support)
    r = _average_prf(tp, tp + fn, None, support)
    f = _average_prf(2 * tp, 2 * tp + fp + fn, None, support)
    width = max(len(str(l)) for l in labels)
    width = max(width, len("weighted avg"))
    head = f"{'':>{width}}  {'precision':>9} {'recall':>9} {'f1-score':>9} {'support':>9}\n\n"
    rows = []
    for i, l in enumerate(labels):
        rows.append(f"{l!s:>{width}}  {p[i]:>9.{digits}f} {r[i]:>9.{digits}f} "
                    f"{f[i]:>9.{digits}f} {support[i]:>9d}")
    total = support.sum()
    rows.append("")
    rows.append(f"{'accuracy':>{width}}  {'':>9} {'':>9} "
                f"{accuracy_score(y_true, y_pred):>9.{digits}f} {total:>9d}")
    for name, avg in (("macro avg", "macro"), ("weighted avg", "weighted")):
        pp = _average_prf(tp, tp + fp, avg, support)
        rr = _average_prf(tp, tp + fn, avg, support)
        ff = _average_prf(2 * tp, 2 * tp + fp + fn, avg, support)
        rows.append(f"{name:>{width}}  {pp:>9.{digits}f} {rr:>9.{digits}f} "
                    f"{ff:>9.{digits}f} {total:>9d}")
    return head + "\n".join(rows) + "\n"


# --------------------------------------------------------------------------
# regression
# --------------------------------------------------------------------------

def mean_squared_error(y_true, y_pred, squared=True, sample_weight=None):
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    mse = float(np.average((y_true - y_pred) ** 2, weights=sample_weight))
    return mse if squared else float(np.sqrt(mse))


def root_mean_squared_error(y_true, y_pred):
    return mean_squared_error(y_true, y_pred, squared=False)


def mean_absolute_error(y_true, y_pred, sample_weight=None):
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    return float(np.average(np.abs(y_true - y_pred), weights=sample_weight))


def r2_score(y_true, y_pred, sample_weight=None):
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    w = np.ones(len(y_true)) if sample_weight is None \
        else np.asarray(sample_weight, dtype=np.float64)
    ss_res = np.sum(w * (y_true - y_pred) ** 2)
    ss_tot = np.sum(w * (y_true - np.average(y_true, weights=w)) ** 2)
    if ss_tot == 0:
        return 0.0 if ss_res > 0 else 1.0
    return float(1.0 - ss_res / ss_tot)


# --------------------------------------------------------------------------
# clustering
# --------------------------------------------------------------------------

def silhouette_score(X, labels):
    from scipy.spatial.distance import squareform, pdist
    X = np.asarray(X, dtype=np.float64)
    labels = column_or_1d(labels)
    uniq = np.unique(labels)
    if len(uniq) < 2 or len(uniq) >= len(X):
        raise ValueError("Number of labels must be 2 <= n_labels <= n_samples - 1")
    D = squareform(pdist(X))
    n = len(X)
    sil = np.zeros(n)
    masks = {l: labels == l for l in uniq}
    counts = {l: m.sum() for l, m in masks.items()}
    for i in range(n):
        own = labels[i]
        if counts[own] == 1:
            sil[i] = 0.0
            continue
        a = D[i, masks[own]].sum() / (counts[own] - 1)
        b = min(D[i, masks[l]].mean() for l in uniq if l != own)
        sil[i] = (b - a) / max(a, b)
    return float(sil.mean())


def adjusted_rand_score(labels_true, labels_pred):
    from scipy.special import comb
    labels_true = column_or_1d(labels_true)
    labels_pred = column_or_1d(labels_pred)
    classes, class_idx = np.unique(labels_true, return_inverse=True)
    clusters, cluster_idx = np.unique(labels_pred, return_inverse=True)
    table = np.zeros((len(classes), len(clusters)), dtype=np.int64)
    np.add.at(table, (class_idx, cluster_idx), 1)
    sum_comb_c = comb(table.sum(axis=1), 2).sum()
    sum_comb_k = comb(table.sum(axis=0), 2).sum()
    sum_comb = comb(table, 2).sum()
    n = comb(len(labels_true), 2)
    expected = sum_comb_c * sum_comb_k / n if n > 0 else 0.0
    max_index = (sum_comb_c + sum_comb_k) / 2
    if max_index == expected:
        return 1.0
    return float((sum_comb - expected) / (max_index - expected))


from ._extra import (  # noqa: E402
    balanced_accuracy_score, matthews_corrcoef, cohen_kappa_score,
    brier_score_loss, hinge_loss, precision_recall_curve,
    average_precision_score, calibration_curve, top_k_accuracy_score,
    median_absolute_error, explained_variance_score,
    mean_absolute_percentage_error, max_error, mean_poisson_deviance,
    mean_gamma_deviance, mean_pinball_loss, dcg_score, ndcg_score,
    mutual_info_score, normalized_mutual_info_score, homogeneity_score,
    completeness_score, v_measure_score, fowlkes_mallows_score,
    calinski_harabasz_score, davies_bouldin_score,
)

__all__ = [
    "accuracy_score", "confusion_matrix", "precision_score", "recall_score",
    "f1_score", "log_loss", "roc_curve", "roc_auc_score", "classification_report",
    "mean_squared_error", "root_mean_squared_error", "mean_absolute_error",
    "r2_score", "silhouette_score", "adjusted_rand_score",
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
