"""Weibull AFT -- risk as the negative predicted median survival time."""
from nupyml.survival import WeibullAFT


def solve(X_train, durations_train, events_train, X_test):
    model = WeibullAFT().fit(X_train, durations_train, events_train.astype(int))
    return -model.predict_median(X_test)   # shorter median => higher risk
