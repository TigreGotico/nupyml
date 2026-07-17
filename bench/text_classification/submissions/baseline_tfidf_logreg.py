"""tf-idf features + logistic regression -- the text-classification workhorse."""
import numpy as np

from nupyml.feature_extraction import TfidfVectorizer
from nupyml.linear_model import LogisticRegression


def solve(X_train, y_train, X_test):
    vec = TfidfVectorizer()
    Xtr = vec.fit_transform([str(d) for d in X_train])
    Xte = vec.transform([str(d) for d in X_test])
    clf = LogisticRegression(max_iter=500).fit(Xtr.toarray(), y_train)
    return clf.predict(Xte.toarray())
