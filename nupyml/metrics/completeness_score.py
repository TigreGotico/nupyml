"""completeness_score"""
from .homogeneity_score import homogeneity_score


def completeness_score(labels_true, labels_pred):
    return homogeneity_score(labels_pred, labels_true)


__all__ = ["completeness_score"]
