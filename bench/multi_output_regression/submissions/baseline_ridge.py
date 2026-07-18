"""Ridge regression, one model per output column."""
import numpy as np

from nupyml.linear_model import Ridge


def solve(X_train, y_train, X_test):
    preds = [Ridge(alpha=1.0).fit(X_train, y_train[:, k]).predict(X_test)
             for k in range(y_train.shape[1])]
    return np.column_stack(preds)
