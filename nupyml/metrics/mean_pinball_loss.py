"""mean_pinball_loss"""
import numpy as np


def mean_pinball_loss(y_true, y_pred, alpha=0.5):
    diff = np.asarray(y_true, dtype=np.float64) - np.asarray(y_pred, dtype=np.float64)
    return float(np.mean(np.maximum(alpha * diff, (alpha - 1) * diff)))


# ---------------------------------------------------------------------------
# ranking
# ---------------------------------------------------------------------------


__all__ = ["mean_pinball_loss"]
