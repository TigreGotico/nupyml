"""The DiD estimator: subtract the control group's TREND from the treated's."""
import numpy as np


def difference_in_differences(y_control_pre, y_control_post,
                              y_treated_pre, y_treated_post):
    """The DiD estimator: subtract the control group's TREND from the treated's.

    With before/after data on a treated and a control group, the treated group's
    raw change confounds the treatment with whatever else happened over time. DiD
    removes the common trend by taking the DIFFERENCE OF DIFFERENCES::

        effect = (treated_post - treated_pre) - (control_post - control_pre)

    Its one assumption is PARALLEL TRENDS: absent treatment, the two groups would
    have moved together. Then the control's change estimates what the treated
    would have done anyway, and the excess is the causal effect. Accepts group
    means (scalars) or arrays (averaged).
    """
    def m(v):
        return float(np.mean(v))
    return (m(y_treated_post) - m(y_treated_pre)) - (m(y_control_post) - m(y_control_pre))


__all__ = ["difference_in_differences"]
