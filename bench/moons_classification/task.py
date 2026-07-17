"""Two-moons: a nonlinearly-separable binary classification task.

The two classes interleave as crescents, so no straight line separates them --
a linear model tops out near 0.88 while kernel/tree/MLP methods reach ~1.0. That
gap is the point of the task.
"""
from nupyml.datasets import make_moons
from nupyml.metrics import accuracy_score
from nupyml.model_selection import train_test_split

KIND = "supervised"
GOAL = "Separate two interleaving half-moons (nonlinear binary classification)."
METRIC = "accuracy"
HIGHER_IS_BETTER = True
MIN_SCORE = 0.90


def load():
    X, y = make_moons(n_samples=1000, noise=0.2, random_state=0)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=0, stratify=y)
    return X_train, y_train, X_test, y_test


def metric(y_true, y_pred):
    return accuracy_score(y_true, y_pred)
