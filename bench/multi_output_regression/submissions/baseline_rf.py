"""A random forest per output (fit one regressor to each column)."""
import numpy as np

from nupyml.ensemble import RandomForestRegressor


def solve(X_train, y_train, X_test):
    preds = []
    for k in range(y_train.shape[1]):
        m = RandomForestRegressor(n_estimators=100, random_state=0).fit(X_train, y_train[:, k])
        preds.append(m.predict(X_test))
    return np.column_stack(preds)
