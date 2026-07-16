"""String-named scorers usable in cross_val_score / *SearchCV."""
import numpy as np


def _proba_scorer(metric, needs="proba", greater_is_better=True, **kw):
    sign = 1.0 if greater_is_better else -1.0

    def scorer(est, X, y):
        if needs == "proba":
            y_in = est.predict_proba(X)
            if y_in.shape[1] == 2:
                y_in = y_in[:, 1]
        elif needs == "decision":
            y_in = (est.decision_function(X) if hasattr(est, "decision_function")
                    else est.predict_proba(X)[:, 1])
        else:
            y_in = est.predict(X)
        return sign * metric(y, y_in, **kw)
    return scorer


def _make_scorers():
    from .. import metrics as m
    S = {}
    S["accuracy"] = _proba_scorer(m.accuracy_score, needs="predict")
    S["balanced_accuracy"] = _proba_scorer(m.balanced_accuracy_score, needs="predict")
    for avg in ("macro", "micro", "weighted"):
        S[f"f1_{avg}"] = _proba_scorer(m.f1_score, needs="predict", average=avg)
        S[f"precision_{avg}"] = _proba_scorer(m.precision_score, needs="predict", average=avg)
        S[f"recall_{avg}"] = _proba_scorer(m.recall_score, needs="predict", average=avg)
    S["f1"] = _proba_scorer(m.f1_score, needs="predict")
    S["precision"] = _proba_scorer(m.precision_score, needs="predict")
    S["recall"] = _proba_scorer(m.recall_score, needs="predict")
    S["roc_auc"] = _proba_scorer(m.roc_auc_score, needs="decision")
    S["average_precision"] = _proba_scorer(m.average_precision_score, needs="decision")
    def neg_log_loss(est, X, y):
        return -m.log_loss(y, est.predict_proba(X), labels=est.classes_)
    S["neg_log_loss"] = neg_log_loss
    S["neg_brier_score"] = _proba_scorer(m.brier_score_loss, needs="proba",
                                         greater_is_better=False)
    S["matthews_corrcoef"] = _proba_scorer(m.matthews_corrcoef, needs="predict")
    S["r2"] = _proba_scorer(m.r2_score, needs="predict")
    S["neg_mean_squared_error"] = _proba_scorer(
        m.mean_squared_error, needs="predict", greater_is_better=False)
    S["neg_root_mean_squared_error"] = _proba_scorer(
        m.root_mean_squared_error, needs="predict", greater_is_better=False)
    S["neg_mean_absolute_error"] = _proba_scorer(
        m.mean_absolute_error, needs="predict", greater_is_better=False)
    S["neg_median_absolute_error"] = _proba_scorer(
        m.median_absolute_error, needs="predict", greater_is_better=False)
    S["explained_variance"] = _proba_scorer(m.explained_variance_score,
                                            needs="predict")
    return S


_SCORERS = None


def get_scorer_names():
    global _SCORERS
    if _SCORERS is None:
        _SCORERS = _make_scorers()
    return sorted(_SCORERS)


def get_scorer(name):
    global _SCORERS
    if _SCORERS is None:
        _SCORERS = _make_scorers()
    if name not in _SCORERS:
        raise ValueError(f"Unknown scoring {name!r}; options: {sorted(_SCORERS)}")
    return _SCORERS[name]


def check_scoring(scoring):
    """None -> None (estimator.score); str -> scorer; callable -> itself."""
    if scoring is None or callable(scoring):
        return scoring
    return get_scorer(scoring)


__all__ = ["get_scorer", "get_scorer_names", "check_scoring"]
