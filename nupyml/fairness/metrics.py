"""Group-fairness metrics: is the model treating the groups differently?

Every metric here compares the model's behaviour ACROSS the values of a sensitive
attribute (sex, race, age band, ...). They answer different fairness questions,
and they GENUINELY CONFLICT -- you cannot in general satisfy all of them at once
(the impossibility results), so which one matters is a modelling and policy
decision, not a default. Each returns a difference (0 = parity) or a ratio
(1 = parity).
"""
import numpy as np


def _groups(sensitive):
    sensitive = np.asarray(sensitive)
    return np.unique(sensitive), sensitive


def _rate(mask):
    return mask.mean() if mask.size else 0.0


def selection_rate(y_pred, sensitive):
    """Positive-prediction rate within each group (the raw material of parity)."""
    y_pred = np.asarray(y_pred)
    groups, s = _groups(sensitive)
    return {g: _rate(y_pred[s == g] == 1) for g in groups}


def demographic_parity_difference(y_pred, sensitive):
    """max - min positive-prediction rate across groups.

    DEMOGRAPHIC PARITY asks that the model select each group at the same rate,
    REGARDLESS of the true label. Appropriate when the label itself may be biased
    (e.g. historical arrests) -- but it ignores genuine base-rate differences, so
    it can force equal outcomes where the ground truth differs.
    """
    rates = list(selection_rate(y_pred, sensitive).values())
    return max(rates) - min(rates)


def disparate_impact_ratio(y_pred, sensitive):
    """min/max selection-rate ratio -- the "80% rule" statistic.

    The legal four-fifths guideline: a ratio below 0.8 is treated as evidence of
    adverse impact. Same idea as demographic parity, expressed as a ratio.
    """
    rates = list(selection_rate(y_pred, sensitive).values())
    hi = max(rates)
    return (min(rates) / hi) if hi > 0 else 1.0


def _tpr_fpr(y_true, y_pred, mask):
    yt, yp = y_true[mask], y_pred[mask]
    pos, neg = (yt == 1), (yt == 0)
    tpr = _rate(yp[pos] == 1) if pos.any() else 0.0
    fpr = _rate(yp[neg] == 1) if neg.any() else 0.0
    return tpr, fpr


def equal_opportunity_difference(y_true, y_pred, sensitive):
    """max - min TRUE-POSITIVE rate across groups.

    EQUAL OPPORTUNITY asks that qualified individuals (true label 1) be accepted
    at the same rate in every group -- equal recall. It conditions on the truth,
    so unlike demographic parity it permits base-rate differences.
    """
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    groups, s = _groups(sensitive)
    tprs = [_tpr_fpr(y_true, y_pred, s == g)[0] for g in groups]
    return max(tprs) - min(tprs)


def equalized_odds_difference(y_true, y_pred, sensitive):
    """The larger of the TPR gap and the FPR gap across groups.

    EQUALIZED ODDS is the stricter cousin: equal true-positive AND equal
    false-positive rates. It demands the errors themselves be distributed evenly,
    which is why it is the hardest of the three to satisfy alongside calibration.
    """
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    groups, s = _groups(sensitive)
    tprs, fprs = [], []
    for g in groups:
        tpr, fpr = _tpr_fpr(y_true, y_pred, s == g)
        tprs.append(tpr)
        fprs.append(fpr)
    return max(max(tprs) - min(tprs), max(fprs) - min(fprs))


def predictive_parity_difference(y_true, y_pred, sensitive):
    """max - min PRECISION across groups.

    PREDICTIVE PARITY asks that a positive prediction mean the same thing in every
    group -- equal precision (positive predictive value). This is the criterion at
    the centre of the COMPAS debate, and it provably clashes with equalized odds
    whenever base rates differ.
    """
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    groups, s = _groups(sensitive)
    precs = []
    for g in groups:
        yp = y_pred[s == g] == 1
        yt = y_true[s == g]
        precs.append(_rate(yt[yp] == 1) if yp.any() else 0.0)
    return max(precs) - min(precs)


__all__ = ["selection_rate", "demographic_parity_difference",
           "disparate_impact_ratio", "equal_opportunity_difference",
           "equalized_odds_difference", "predictive_parity_difference"]
