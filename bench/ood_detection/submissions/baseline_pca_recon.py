"""PCA reconstruction error -- OOD points lie off the principal subspace."""
from nupyml.anomaly import PCAReconstructionDetector


def solve(X):
    return PCAReconstructionDetector(n_components=3).fit(X).decision_function(X)
