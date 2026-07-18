"""Raw flattened pixels + RBF SVC -- the position-sensitive floor."""
from nupyml.pipeline import make_pipeline
from nupyml.preprocessing import StandardScaler
from nupyml.svm import SVC


def solve(X_train, y_train, X_test):
    Xtr = X_train.reshape(len(X_train), -1)
    Xte = X_test.reshape(len(X_test), -1)
    m = make_pipeline(StandardScaler(), SVC(C=10, gamma="scale")).fit(Xtr, y_train)
    return m.predict(Xte)
