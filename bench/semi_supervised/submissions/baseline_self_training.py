"""Self-training: label the confident unlabeled points and retrain, iterating."""
import numpy as np

from nupyml.semi_supervised import SelfTrainingClassifier
from nupyml.linear_model import LogisticRegression


def solve(X_train, y_train, X_test):
    clf = SelfTrainingClassifier(LogisticRegression(max_iter=500), threshold=0.8)
    clf.fit(X_train, y_train)              # -1 marks the unlabeled points
    return clf.predict(X_test)
