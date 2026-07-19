"""Oriented FAST + rotated BRIEF: fast BINARY descriptors (Rublee, 2011)."""
import numpy as np
from ..utils import check_random_state
from .vision import harris_corners, corner_peaks, _gray


def orb_descriptors(image, keypoints=None, n_keypoints=50, patch=15, n_bits=256,
                    random_state=0):
    """Oriented FAST + rotated BRIEF: fast BINARY descriptors (Rublee, 2011).

    ORB describes a keypoint by a string of BITS, each the outcome of comparing two
    pixel intensities in its neighbourhood ("is point a brighter than point b?").
    Crucially the comparison pattern is ROTATED to the patch's dominant orientation
    (from the intensity centroid), so the descriptor is rotation-invariant, and
    matching is a Hamming distance -- an XOR and a bit-count, orders of magnitude
    faster than the float descriptors it replaced. Returns (keypoints, bit
    descriptors).
    """
    img = _gray(image) if np.asarray(image).ndim == 3 else np.asarray(image, float)
    rng = check_random_state(random_state)
    if keypoints is None:
        resp = harris_corners(img)
        keypoints = corner_peaks(resp, min_distance=5)[:n_keypoints]
    half = patch // 2
    pairs = rng.randint(-half, half + 1, size=(n_bits, 4))   # (ay,ax,by,bx)
    desc, kept = [], []
    H, W = img.shape
    for (y, x) in keypoints:
        if not (half <= y < H - half and half <= x < W - half):
            continue
        win = img[y - half:y + half + 1, x - half:x + half + 1]
        # orientation from the intensity centroid
        ys, xs = np.mgrid[-half:half + 1, -half:half + 1]
        m00 = win.sum() + 1e-9
        theta = np.arctan2((ys * win).sum() / m00, (xs * win).sum() / m00)
        ct, st = np.cos(theta), np.sin(theta)
        bits = []
        for ay, ax, by, bx in pairs:
            ra = int(round(ay * ct - ax * st)); ca = int(round(ay * st + ax * ct))
            rb = int(round(by * ct - bx * st)); cb = int(round(by * st + bx * ct))
            pa = img[np.clip(y + ra, 0, H - 1), np.clip(x + ca, 0, W - 1)]
            pb = img[np.clip(y + rb, 0, H - 1), np.clip(x + cb, 0, W - 1)]
            bits.append(1 if pa < pb else 0)
        desc.append(bits); kept.append((y, x))
    return np.array(kept), np.array(desc, dtype=np.uint8)


__all__ = ["orb_descriptors"]
