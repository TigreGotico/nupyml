"""Time-series CLASSIFICATION -- labelling a whole series by its shape.

The ``timeseries`` package forecasts; this one classifies. Each estimator takes a
2-D array whose rows are series and returns a label per row, following the usual
``fit``/``predict`` contract, so they drop into pipelines and searches.

The four modern families, plus supporting tools:

* ``rocket`` -- **ROCKET**: thousands of RANDOM convolutional kernels summarised by
  max + proportion-of-positive-values, fed to a linear classifier. Fast and
  state-of-the-art-for-cheap, with no kernel learning.
* ``shapelets`` -- distance to discriminative SUBSEQUENCES; interpretable, and
  invariant to where the shape occurs.
* ``dictionary`` -- **BOSS**: histogram of symbolic (SFA) words; robust to noise
  and warping via its low-pass + discretise design.
* ``distance`` -- **DTW k-NN**: elastic-distance nearest neighbour, the classic
  strong baseline every method must beat.
* ``features`` -- a compact tsfresh-style feature bank feeding any tabular model.
* ``matrix_profile`` -- motif and discord discovery from the all-subsequences
  nearest-neighbour distance profile (one FFT-based computation, two answers).

Only numpy and scipy are used; DTW reuses ``nupyml.sequence``.
"""
from .rocket import Rocket, RocketClassifier
from .shapelets import ShapeletTransform, ShapeletTransformClassifier
from .dictionary import BOSS
from .distance import KNeighborsTimeSeriesClassifier
from .features import TSFeatureExtractor, FEATURE_NAMES
from .matrix_profile import matrix_profile, motif, discord

__all__ = ["Rocket", "RocketClassifier", "ShapeletTransform",
           "ShapeletTransformClassifier", "BOSS",
           "KNeighborsTimeSeriesClassifier", "TSFeatureExtractor",
           "FEATURE_NAMES", "matrix_profile", "motif", "discord"]
