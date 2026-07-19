"""Scale-invariant keypoints and descriptors (Lowe, 2004)."""
import numpy as np
import scipy.ndimage as ndi
from .vision import _gray


def _sift_descriptor(img, y, x, scale, size=16):
    H, W = img.shape
    half = size // 2
    y0, x0 = int(np.clip(y, half, H - half - 1)), int(np.clip(x, half, W - half - 1))
    win = img[y0 - half:y0 + half, x0 - half:x0 + half]
    gy, gx = np.gradient(win)
    mag = np.sqrt(gx ** 2 + gy ** 2)
    ang = (np.arctan2(gy, gx) + np.pi)                  # [0, 2pi)
    dominant = ang.ravel()[mag.argmax()]               # orientation normalisation
    ang = (ang - dominant) % (2 * np.pi)
    desc = []
    cell = size // 4
    for i in range(4):
        for j in range(4):
            m = mag[i * cell:(i + 1) * cell, j * cell:(j + 1) * cell].ravel()
            a = ang[i * cell:(i + 1) * cell, j * cell:(j + 1) * cell].ravel()
            hist, _ = np.histogram(a, bins=8, range=(0, 2 * np.pi), weights=m)
            desc.extend(hist)
    desc = np.array(desc)
    return desc / (np.linalg.norm(desc) + 1e-8)


# ------------------------------------------------------------------ optical flow


def sift(image, n_octaves=3, n_scales=3, sigma=1.6, contrast_threshold=0.03,
         n_keypoints=100):
    """Scale-invariant keypoints and descriptors (Lowe, 2004).

    A corner detector fires at one scale only, so a zoomed object is missed. SIFT
    searches a Gaussian SCALE-SPACE: it finds extrema of the difference-of-Gaussians
    across both position AND scale (so a feature is detected at whatever size it
    appears), assigns each a dominant ORIENTATION from local gradients, and encodes
    the surrounding gradients as a 128-D histogram descriptor -- normalised so it is
    invariant to scale, rotation, and illumination. Matching those descriptors is
    how images are stitched and objects recognised across viewpoints. Returns
    ``(keypoints, descriptors)``.
    """
    img = _gray(image) if np.asarray(image).ndim == 3 else np.asarray(image, float)
    keypoints = []
    for octave in range(n_octaves):
        scaled = img[::2 ** octave, ::2 ** octave]
        if min(scaled.shape) < 8:
            break
        gaussians = [ndi.gaussian_filter(scaled, sigma * 1.6 ** s)
                     for s in range(n_scales + 2)]
        dogs = [gaussians[s + 1] - gaussians[s] for s in range(n_scales + 1)]
        for s in range(1, n_scales):
            prev, cur, nxt = dogs[s - 1], dogs[s], dogs[s + 1]
            H, W = cur.shape
            for y in range(1, H - 1):
                for x in range(1, W - 1):
                    v = cur[y, x]
                    if abs(v) < contrast_threshold:
                        continue
                    patch = np.stack([prev[y - 1:y + 2, x - 1:x + 2],
                                      cur[y - 1:y + 2, x - 1:x + 2],
                                      nxt[y - 1:y + 2, x - 1:x + 2]])
                    if v >= patch.max() or v <= patch.min():   # 26-neighbour extremum
                        keypoints.append((y * 2 ** octave, x * 2 ** octave,
                                          sigma * 1.6 ** s * 2 ** octave))
    keypoints = keypoints[:n_keypoints]
    desc = np.array([_sift_descriptor(img, y, x, sc) for (y, x, sc) in keypoints]) \
        if keypoints else np.empty((0, 128))
    return np.array(keypoints), desc


__all__ = ["sift"]
