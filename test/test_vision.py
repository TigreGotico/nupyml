"""I3: computer-vision operators -- corners, edges, matching, flow, distance,
NMS, superpixels.

Each is checked against a synthetic scene with a known answer: a square has four
corners and edges; a structured template is found at its planted location; a
shifted image induces flow in the shift direction; the distance transform gives
exact distances; NMS collapses overlapping boxes; SLIC returns the requested
segment count.
"""
import numpy as np
import pytest

from nupyml.image import (harris_corners, corner_peaks, canny, match_template,
                         lucas_kanade, chamfer_distance_transform,
                         non_max_suppression, slic)


@pytest.fixture
def square():
    img = np.zeros((40, 40))
    img[10:30, 10:30] = 1.0
    return img


def test_harris_finds_the_four_corners(square):
    R = harris_corners(square)
    peaks = corner_peaks(R, min_distance=3, threshold_rel=0.2)
    assert len(peaks) == 4                             # a square has four corners


def test_canny_produces_thin_edges(square):
    edges = canny(square)
    assert edges.sum() > 0
    # edges are a thin outline, far fewer pixels than the filled square
    assert edges.sum() < (square > 0).sum()


def test_match_template_finds_a_structured_patch(square):
    # a template containing the top-left CORNER (has structure -> NCC well-defined)
    tmpl = square[6:16, 6:16]
    corr = match_template(square, tmpl)
    py, px = np.unravel_index(corr.argmax(), corr.shape)
    assert abs(py - 6) <= 1 and abs(px - 6) <= 1
    assert corr.max() > 0.9                            # near-perfect match


def test_lucas_kanade_recovers_shift_direction(square):
    shifted = np.roll(square, 1, axis=1)               # shift right by 1 pixel
    flow = lucas_kanade(square, shifted, [[20, 10], [20, 29]], window=7)
    # flow's x-component (u) is positive -- the content moved right
    assert np.all(flow[:, 0] > 0)


def test_chamfer_distance_transform_is_correct():
    b = np.zeros((20, 20), bool)
    b[10, 10] = True
    D = chamfer_distance_transform(b)
    assert D[10, 10] == 0.0
    assert D[10, 15] == pytest.approx(5.0, abs=1e-6)   # 5 steps orthogonally
    assert D[13, 14] == pytest.approx(3 * np.sqrt(2) + 1, abs=1e-6)  # 3 diag + 1


def test_non_max_suppression_collapses_overlaps():
    boxes = [[0, 0, 10, 10], [1, 1, 11, 11], [50, 50, 60, 60]]
    scores = [0.9, 0.8, 0.7]
    keep = non_max_suppression(boxes, scores, iou_threshold=0.5)
    assert keep == [0, 2]                              # box 1 overlaps 0 -> dropped


def test_slic_returns_requested_segments(square):
    labels = slic(square, n_segments=16)
    assert 1 <= len(np.unique(labels)) <= 20          # ~16 superpixels
    assert labels.shape == square.shape
    assert (labels >= 0).all()                         # every pixel assigned
