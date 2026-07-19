"""Segment by finding MODES in colour+position space (Comaniciu & Meer, 2002)."""
import numpy as np
from ..base import BaseEstimator
from .vision import _gray


class MeanShiftSegmentation(BaseEstimator):
    """Segment by finding MODES in colour+position space (Comaniciu & Meer, 2002).

    K-means needs the number of segments; mean-shift discovers it. Each pixel becomes
    a point in a joint COLOUR-and-POSITION space, and mean-shift climbs the density
    of those points -- each pixel iteratively moves to the weighted mean of its
    neighbours within a bandwidth -- until it reaches a MODE. Pixels that converge to
    the same mode form a segment, so the number of regions falls out of the data. The
    ``spatial_radius`` and ``color_radius`` bandwidths trade region size against
    detail. Returns an integer label map.
    """

    def __init__(self, spatial_radius=2.0, color_radius=0.3, max_iter=20):
        self.spatial_radius = spatial_radius
        self.color_radius = color_radius
        self.max_iter = max_iter

    def fit_predict(self, image):
        img = _gray(image) if np.asarray(image).ndim == 3 else np.asarray(image, float)
        H, W = img.shape
        rows, cols = np.mgrid[0:H, 0:W]
        feats = np.column_stack([rows.ravel() / self.spatial_radius,
                                 cols.ravel() / self.spatial_radius,
                                 img.ravel() / self.color_radius])
        pts = feats.copy()
        for _ in range(self.max_iter):
            new = np.empty_like(pts)
            for i in range(len(pts)):
                d = np.linalg.norm(feats - pts[i], axis=1)
                near = d < 1.0
                new[i] = feats[near].mean(axis=0)        # mean shift toward the mode
            if np.abs(new - pts).max() < 1e-3:
                pts = new; break
            pts = new
        # group pixels whose modes coincide
        rounded = np.round(pts, 1)
        _, labels = np.unique(rounded, axis=0, return_inverse=True)
        return labels.reshape(H, W)


__all__ = ["MeanShiftSegmentation"]
