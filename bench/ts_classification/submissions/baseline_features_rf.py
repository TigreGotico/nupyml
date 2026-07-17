"""Hand-crafted summary features + a random forest.

Turn each series into a short vector of phase-invariant statistics -- moments,
zero-crossings, autocorrelation, and the dominant FFT magnitudes -- then let a
tabular classifier do the rest. Cheap, and surprisingly hard to beat.
"""
import numpy as np

from nupyml.ensemble import RandomForestClassifier


def _features(series):
    s = np.asarray(series, dtype=float)
    fft = np.abs(np.fft.rfft(s - s.mean()))
    ac1 = np.corrcoef(s[:-1], s[1:])[0, 1] if len(s) > 2 else 0.0
    return np.array([
        s.mean(), s.std(), s.min(), s.max(),
        np.mean(np.abs(np.diff(s))),                 # mean absolute change
        ((s[:-1] * s[1:]) < 0).sum(),                # zero crossings
        ac1,                                          # lag-1 autocorrelation
        fft.argmax(),                                 # dominant frequency bin
        fft.max(),                                    # its magnitude
    ])


def solve(X_train, y_train, X_test):
    Ftr = np.array([_features(x) for x in X_train])
    Fte = np.array([_features(x) for x in X_test])
    clf = RandomForestClassifier(n_estimators=100, random_state=0).fit(Ftr, y_train)
    return clf.predict(Fte)
