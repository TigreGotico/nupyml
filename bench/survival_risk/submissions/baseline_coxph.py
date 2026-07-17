"""Cox proportional hazards -- the standard semi-parametric risk model."""
from nupyml.survival import CoxPH


def solve(X_train, durations_train, events_train, X_test):
    model = CoxPH().fit(X_train, durations_train, events_train.astype(int))
    return model.predict_risk(X_test)
