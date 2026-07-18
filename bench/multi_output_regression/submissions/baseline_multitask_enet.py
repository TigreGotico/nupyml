"""Multi-task elastic net -- shares feature support across the outputs."""
from nupyml.linear_model import MultiTaskElasticNet


def solve(X_train, y_train, X_test):
    return MultiTaskElasticNet(alpha=0.05, l1_ratio=0.5).fit(X_train, y_train).predict(X_test)
