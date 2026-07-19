"""Image denoising: recover clean images from noisy ones, scored by PSNR.

Each sample is a piecewise-constant image (random bright rectangles) corrupted by
additive Gaussian noise. A submission sees noisy images and must return denoised ones;
scored by mean PSNR against the hidden clean images. Edge-preserving denoisers should
beat a plain blur. Framed as supervised (clean images are the held-out target).
"""
import numpy as np

KIND = "supervised"
GOAL = "Denoise noisy images; scored by mean PSNR (dB) against the clean originals."
METRIC = "psnr_db"
HIGHER_IS_BETTER = True
MIN_SCORE = 20.0


def _clean_image(rng, size=40):
    img = np.zeros((size, size))
    for _ in range(rng.randint(2, 4)):
        r0, c0 = rng.randint(0, size - 12, 2)
        h, w = rng.randint(8, 16, 2)
        img[r0:r0 + h, c0:c0 + w] = rng.uniform(0.5, 1.0)
    return img


def load():
    rng = np.random.RandomState(0)
    clean = np.array([_clean_image(rng) for _ in range(60)])
    noisy = clean + 0.2 * rng.randn(*clean.shape)
    cut = 30
    # train pairs are provided too, though good denoisers need no training
    return noisy[:cut], clean[:cut], noisy[cut:], clean[cut:]


def metric(y_true, y_pred):
    y_pred = np.clip(np.asarray(y_pred), 0.0, 1.0)
    mse = np.mean((y_true - y_pred) ** 2, axis=(1, 2))
    psnr = 10 * np.log10(1.0 / (mse + 1e-12))
    return float(np.mean(psnr))
