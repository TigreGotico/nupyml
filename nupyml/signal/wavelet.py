"""The discrete wavelet transform and wavelet denoising.

WHY WAVELETS, NOT FOURIER
-------------------------
The Fourier transform tells you WHICH frequencies are present but not WHEN -- it
is useless for a signal whose frequency content changes over time (a chirp, a
heartbeat, a spoken word). Wavelets give a TIME-FREQUENCY decomposition:
successive levels split the signal into a coarse APPROXIMATION and the DETAIL
lost at that scale, localised in time. Low levels capture fast, brief events;
high levels capture slow trends. That multi-resolution view is what makes
wavelets the tool for transient and non-stationary signals.
"""
import numpy as np

# decomposition low-pass filters; the high-pass and reconstruction filters are
# derived from these by the quadrature-mirror relations below.
_FILTERS = {
    "haar": [0.7071067811865476, 0.7071067811865476],
    "db2": [0.48296291314469025, 0.836516303737469,
            0.22414386804185735, -0.12940952255092145],
}


def _qmf(lo):
    """High-pass QMF ``h1[k] = (-1)^k h0[L-1-k]`` -- the mirror that makes the
    analysis operator ORTHONORMAL, so reconstruction is just its transpose."""
    lo = np.asarray(lo, float)
    L = len(lo)
    k = np.arange(L)
    return ((-1.0) ** k) * lo[L - 1 - k]


def _dwt_single(x, lo, hi):
    """One level: each output is a downsampled circular-shifted filter response.

    The rows of this operator are downsampled circular shifts of (lo, hi); for the
    QMF pair they are orthonormal, which is exactly why :func:`_idwt_single` (its
    transpose) reconstructs perfectly.
    """
    N = len(x)
    L = len(lo)
    half = N // 2
    # gather the length-L circular window starting at each even position
    starts = (2 * np.arange(half)[:, None] + np.arange(L)[None, :]) % N
    windows = x[starts]                             # (half, L)
    return windows @ lo, windows @ hi


def _idwt_single(a, d, lo, hi):
    """Transpose of :func:`_dwt_single`: scatter each coefficient back."""
    half = len(a)
    N = 2 * half
    L = len(lo)
    x = np.zeros(N)
    starts = (2 * np.arange(half)[:, None] + np.arange(L)[None, :]) % N
    for n in range(half):
        x[starts[n]] += lo * a[n] + hi * d[n]
    return x


def dwt(x, wavelet="db2", level=None):
    """Multilevel DWT. Returns ``[cA_L, cD_L, ..., cD_1]`` (pywt convention).

    Each level splits the running approximation into a coarser approximation and
    the detail at that scale; the final list holds the coarsest approximation
    followed by the details from coarse to fine.
    """
    x = np.asarray(x, float)
    lo = np.array(_FILTERS[wavelet]); hi = _qmf(lo)
    if level is None:
        level = max(1, int(np.log2(len(x))) - 2)
    coeffs = []
    a = x
    for _ in range(level):
        if len(a) < len(lo):
            break
        a, d = _dwt_single(a, lo, hi)
        coeffs.append(d)
    coeffs.append(a)
    return coeffs[::-1]                              # [cA_L, cD_L, ..., cD_1]


def idwt(coeffs, wavelet="db2"):
    """Invert :func:`dwt` back to the (approximate) original signal."""
    lo = np.array(_FILTERS[wavelet]); hi = _qmf(lo)
    a = coeffs[0]
    for d in coeffs[1:]:
        if len(d) != len(a):                        # trim length mismatch
            m = min(len(a), len(d))
            a, d = a[:m], d[:m]
        a = _idwt_single(a, d, lo, hi)
    return a


def wavelet_denoise(x, wavelet="db2", level=None, threshold=None):
    """Denoise by SOFT-THRESHOLDING the detail coefficients.

    THE INSIGHT (Donoho & Johnstone)
    --------------------------------
    In a wavelet basis, a smooth signal concentrates into a FEW large detail
    coefficients while white noise spreads THINLY across all of them. So shrinking
    the small details toward zero (soft thresholding) removes mostly noise and
    keeps the signal -- a near-optimal denoiser that, unlike a low-pass filter,
    preserves sharp edges (they live in the surviving large coefficients). The
    universal threshold ``sigma*sqrt(2 log n)`` uses a robust noise estimate from
    the finest details (median absolute deviation).
    """
    coeffs = dwt(x, wavelet, level)
    finest = coeffs[-1]
    sigma = np.median(np.abs(finest)) / 0.6745       # robust noise estimate (MAD)
    if threshold is None:
        threshold = sigma * np.sqrt(2 * np.log(len(x)))
    out = [coeffs[0]]                                # keep the approximation
    for d in coeffs[1:]:
        out.append(np.sign(d) * np.maximum(np.abs(d) - threshold, 0.0))
    rec = idwt(out, wavelet)
    return rec[:len(x)]


__all__ = ["dwt", "idwt", "wavelet_denoise"]
