"""Baseline: sparse variational GP with a handful of inducing points."""
from nupyml.inference import SparseVariationalGP


def solve(X_train, y_train, X_test):
    gp = SparseVariationalGP(n_inducing=25, length_scale=1.0, noise=0.1,
                             random_state=0).fit(X_train, y_train)
    return gp.predict(X_test)
