"""Label spreading: propagate labels through a similarity graph of all points."""
from nupyml.semi_supervised import LabelSpreading


def solve(X_train, y_train, X_test):
    ls = LabelSpreading(kernel="knn", n_neighbors=10).fit(X_train, y_train)
    return ls.predict(X_test)
