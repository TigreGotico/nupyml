"""v_measure_score"""
from .completeness_score import completeness_score
from .homogeneity_score import homogeneity_score


def v_measure_score(labels_true, labels_pred, beta=1.0):
    h = homogeneity_score(labels_true, labels_pred)
    c = completeness_score(labels_true, labels_pred)
    if h + c == 0:
        return 0.0
    return float((1 + beta) * h * c / (beta * h + c))


__all__ = ["v_measure_score"]
