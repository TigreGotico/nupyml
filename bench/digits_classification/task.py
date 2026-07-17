"""Digits classification: recognise 8x8 handwritten digits (10 classes)."""
from nupyml.datasets import load_digits
from nupyml.metrics import accuracy_score
from nupyml.model_selection import train_test_split

KIND = "supervised"
GOAL = "Classify 8x8 handwritten digits into the ten classes 0-9."
METRIC = "accuracy"
HIGHER_IS_BETTER = True
MIN_SCORE = 0.90            # QA floor every baseline must clear


def load():
    X, y = load_digits(return_X_y=True)
    X = X / 16.0            # pixels are 0-16; scale for the distance/linear models
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=0, stratify=y)
    return X_train, y_train, X_test, y_test


def metric(y_true, y_pred):
    return accuracy_score(y_true, y_pred)
