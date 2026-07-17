"""Random survival forest -- non-linear risk from log-rank-split trees."""
from nupyml.survival import RandomSurvivalForest


def solve(X_train, durations_train, events_train, X_test):
    model = RandomSurvivalForest(n_estimators=60, random_state=0)
    model.fit(X_train, durations_train, events_train.astype(int))
    return model.predict_risk(X_test)
