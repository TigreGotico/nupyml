"""Represent an image as a HISTOGRAM of quantised local features (Sivic, 2003)."""
import numpy as np
from ..base import BaseEstimator
from .vision import harris_corners, corner_peaks, _gray


class BagOfVisualWords(BaseEstimator):
    """Represent an image as a HISTOGRAM of quantised local features (Sivic, 2003).

    Borrowed from text: treat local image descriptors as "visual words". Cluster a
    corpus of descriptors into a VOCABULARY (k-means centres), then describe any
    image by the histogram of which visual words its descriptors fall into -- a
    fixed-length vector regardless of image size or keypoint count, ready for a
    plain classifier. Order and position are discarded, exactly like bag-of-words
    for documents.
    """

    def __init__(self, vocab_size=32, patch=8, stride=8, random_state=None):
        self.vocab_size = vocab_size
        self.patch = patch
        self.stride = stride
        self.random_state = random_state

    def _patches(self, image):
        img = _gray(image) if np.asarray(image).ndim == 3 else np.asarray(image, float)
        p, s = self.patch, self.stride
        out = []
        for y in range(0, img.shape[0] - p + 1, s):
            for x in range(0, img.shape[1] - p + 1, s):
                patch = img[y:y + p, x:x + p].ravel()
                out.append(patch - patch.mean())         # contrast-normalise
        return np.array(out)

    def fit(self, images):
        from ..cluster import KMeans
        feats = np.vstack([self._patches(im) for im in images])
        self.kmeans_ = KMeans(n_clusters=self.vocab_size,
                              random_state=self.random_state).fit(feats)
        return self

    def transform(self, images):
        out = []
        for im in images:
            words = self.kmeans_.predict(self._patches(im))
            hist = np.bincount(words, minlength=self.vocab_size).astype(float)
            out.append(hist / (hist.sum() + 1e-9))       # normalised histogram
        return np.array(out)


__all__ = ["BagOfVisualWords"]
