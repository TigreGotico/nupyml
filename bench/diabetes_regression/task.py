"""Diabetes: predict disease progression one year on from ten baseline features."""
from nupyml.datasets import load_diabetes
from nupyml.metrics import r2_score
from nupyml.model_selection import train_test_split

KIND = "supervised"
GOAL = "Predict quantitative diabetes progression from ten baseline measurements."
METRIC = "r2"
HIGHER_IS_BETTER = True
MIN_SCORE = 0.15           # a QA floor: diabetes is genuinely hard (~0.45 is strong)


def load():
    X, y = load_diabetes(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=0)
    return X_train, y_train, X_test, y_test


def metric(y_true, y_pred):
    return r2_score(y_true, y_pred)
