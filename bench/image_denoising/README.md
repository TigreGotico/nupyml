# image_denoising

**Goal:** recover clean images from noisy ones.

- **Kind:** supervised — `solve(X_train, y_train, X_test) -> y_pred`, images of shape `(n, 40, 40)`.
- **Data:** piecewise-constant images (random bright rectangles) plus Gaussian noise (σ=0.2). Clean images are the held-out target.
- **Metric:** mean PSNR in dB (higher is better).
- **QA floor:** 20.0 dB. The noisy input is ~16.5 dB; edge-preserving denoisers reach 22–25 dB.

Denoising needs no training — `solve` may ignore the train split. Total-variation, non-local means and the bilateral filter all preserve the edges a plain blur would smear.
