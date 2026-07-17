"""LambdaMART: gradient-boosted trees driven by NDCG-scaled pairwise lambdas."""
from nupyml.ranking import LambdaMART


def solve(X_train, y_train, groups_train, X_test, groups_test):
    model = LambdaMART(n_estimators=60, max_depth=3, random_state=0)
    model.fit(X_train, y_train, list(groups_train))
    return model.predict(X_test)
