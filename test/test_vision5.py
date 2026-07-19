"""M4: vision v5 -- bilateral, non-local means, TV, anisotropic diffusion,
graph-cut, Hu moments.

The four denoisers each reduce noise while keeping the edge; graph-cut segments a
two-region image from seeds; Hu moments are invariant to translation, scale and
rotation of an asymmetric shape.
"""
import numpy as np
import scipy.ndimage as ndi
import pytest

from nupyml.image import (bilateral_filter, non_local_means,
                          total_variation_denoise, anisotropic_diffusion,
                          graph_cut_segmentation, hu_moments)


def _noisy_edge(rng):
    clean = np.zeros((40, 40)); clean[:, 20:] = 1.0
    return clean, clean + rng.randn(40, 40) * 0.2


@pytest.mark.parametrize("name", ["bilateral", "nlm", "tv", "aniso"])
def test_denoisers_reduce_noise_and_keep_edge(name):
    rng = np.random.RandomState(0)
    clean, noisy = _noisy_edge(rng)
    d = {"bilateral": lambda: bilateral_filter(noisy, 2.0, 0.3, 3),
         "nlm": lambda: non_local_means(noisy, 2, 5, 0.3),
         "tv": lambda: total_variation_denoise(noisy, 0.15, 80),
         "aniso": lambda: anisotropic_diffusion(noisy, 25, 0.15, 0.2)}[name]()
    assert np.mean((d - clean) ** 2) < np.mean((noisy - clean) ** 2)   # denoises
    assert abs(d[:, 19].mean() - d[:, 21].mean()) > 0.7                # edge preserved


def test_graph_cut_segments_two_regions():
    rng = np.random.RandomState(0)
    img = np.zeros((20, 20)); img[:, 10:] = 1.0
    img += rng.randn(20, 20) * 0.05
    seg = graph_cut_segmentation(img, [(5, 15), (15, 15)], [(5, 5), (15, 5)],
                                 sigma=0.2, lam=2.0)
    truth = (np.arange(20)[None, :] >= 10).astype(int)
    truth = np.repeat(truth, 20, axis=0)
    assert np.mean(seg == truth) > 0.95


def test_hu_moments_are_invariant():
    shape = np.zeros((80, 80))
    shape[20:60, 20:35] = 1.0; shape[45:60, 20:55] = 1.0      # asymmetric L
    h = hu_moments(shape)
    # translation
    tr = np.zeros((80, 80)); tr[25:65, 30:45] = 1.0; tr[50:65, 30:65] = 1.0
    assert np.allclose(h[:3], hu_moments(tr)[:3], atol=0.05)
    # scale
    sc = ndi.zoom(shape, 0.6, order=0)
    assert np.allclose(h[:3], hu_moments(sc)[:3], atol=0.3)
    # rotation
    rot = ndi.rotate(shape, 45, reshape=False, order=1)
    assert np.allclose(h[:3], hu_moments(rot)[:3], atol=0.5)
