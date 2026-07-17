"""RankNet: a linear scorer trained on the pairwise ranking cross-entropy."""
from nupyml.ranking import RankNet


def solve(X_train, y_train, groups_train, X_test, groups_test):
    model = RankNet(n_epochs=200, random_state=0)
    model.fit(X_train, y_train, list(groups_train))
    return model.predict(X_test)
