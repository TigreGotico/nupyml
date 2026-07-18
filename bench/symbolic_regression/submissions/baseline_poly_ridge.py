"""Polynomial features + ridge -- approximate the formula in a fixed basis."""
import numpy as np

from nupyml.preprocessing import PolynomialFeatures
from nupyml.linear_model import Ridge
from nupyml.pipeline import make_pipeline


def solve(X_train, y_train, X_test):
    model = make_pipeline(PolynomialFeatures(degree=3), Ridge(alpha=0.1))
    model.fit(X_train, y_train)
    return model.predict(X_test)
