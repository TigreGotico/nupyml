"""Empirical Mode Decomposition and simple detection utilities.

EMD is the data-driven alternative to Fourier/wavelet decomposition: it adapts
its basis to the signal instead of imposing sines or wavelets, which suits
NON-STATIONARY and NON-LINEAR signals.
"""
import numpy as np


def _extrema(x):
    d = np.diff(x)
    maxima = np.where((d[:-1] > 0) & (d[1:] <= 0))[0] + 1
    minima = np.where((d[:-1] < 0) & (d[1:] >= 0))[0] + 1
    return maxima, minima


def _envelope(idx, vals, n):
    from scipy.interpolate import CubicSpline
    # pad the ends so the spline does not swing wildly at the boundary
    idx = np.concatenate([[0], idx, [n - 1]])
    vals = np.concatenate([[vals[0]], vals, [vals[-1]]])
    idx, uniq = np.unique(idx, return_index=True)
    cs = CubicSpline(idx, vals[uniq])
    return cs(np.arange(n))


def emd(x, max_imfs=8, max_sift=50, tol=0.05):
    """Empirical Mode Decomposition into Intrinsic Mode Functions (Huang, 1998).

    THE IDEA
    --------
    Rather than project onto a FIXED basis (sines for Fourier, a chosen mother
    wavelet for the DWT), EMD extracts oscillatory components DEFINED BY THE DATA.
    Each Intrinsic Mode Function is found by "sifting": build the upper envelope
    through the maxima and the lower envelope through the minima (cubic splines),
    subtract their mean, and repeat until what remains is a symmetric zero-mean
    oscillation. Subtract that IMF from the signal and repeat on the residual.

    The signal is rebuilt EXACTLY as the sum of its IMFs plus a monotonic
    residual, from fastest oscillation to slowest -- an adaptive, local
    time-frequency decomposition ideal for non-stationary, non-linear signals
    where a fixed basis smears the content.

    Returns a list of IMFs; the last entry is the trend residual.
    """
    x = np.asarray(x, float)
    n = len(x)
    imfs = []
    residual = x.copy()
    for _ in range(max_imfs):
        h = residual.copy()
        for _ in range(max_sift):
            maxima, minima = _extrema(h)
            if len(maxima) < 2 or len(minima) < 2:
                break                                # too few extrema -> a trend
            upper = _envelope(maxima, h[maxima], n)
            lower = _envelope(minima, h[minima], n)
            mean = (upper + lower) / 2
            new_h = h - mean
            if np.mean(mean ** 2) < tol * np.mean(h ** 2):
                h = new_h
                break
            h = new_h
        imfs.append(h)
        residual = residual - h
        maxima, minima = _extrema(residual)
        if len(maxima) + len(minima) < 3:            # residual is monotonic
            break
    imfs.append(residual)                            # final trend
    return imfs


def find_peaks(x, height=None, distance=1):
    """Indices of local maxima, optionally above ``height`` and ``distance`` apart.

    A minimal peak picker: a sample greater than both neighbours is a candidate;
    the height filter drops small bumps, and the distance filter keeps only the
    tallest peak within any window (so one broad peak is not reported many times).
    """
    x = np.asarray(x, float)
    cand = np.where((x[1:-1] > x[:-2]) & (x[1:-1] > x[2:]))[0] + 1
    if height is not None:
        cand = cand[x[cand] >= height]
    if distance > 1 and len(cand):
        order = cand[np.argsort(-x[cand])]           # tallest first
        kept = []
        taken = np.zeros(len(x), bool)
        for p in order:
            if not taken[max(0, p - distance):p + distance + 1].any():
                kept.append(p); taken[p] = True
        cand = np.sort(kept)
    return np.asarray(cand, dtype=int)


def envelope(x):
    """Amplitude envelope via the analytic signal (|Hilbert transform|).

    The Hilbert transform builds the ANALYTIC signal whose magnitude is the
    instantaneous amplitude -- the smooth outline riding over an oscillation
    (an AM demodulator). Computed through the FFT: zero the negative frequencies,
    double the positive ones, invert, take the modulus.
    """
    from scipy.signal import hilbert
    return np.abs(hilbert(np.asarray(x, float)))


def zero_crossing_rate(x, frame_length=256, hop=None):
    """Fraction of sign changes per frame -- a cheap proxy for pitch/noisiness.

    Voiced speech and low tones cross zero rarely; noise and fricatives cross
    often. One of the oldest and cheapest audio features.
    """
    x = np.asarray(x, float)
    hop = hop or frame_length // 2
    n_frames = 1 + max(0, (len(x) - frame_length) // hop)
    rates = np.empty(n_frames)
    for i in range(n_frames):
        f = x[i * hop:i * hop + frame_length]
        rates[i] = np.mean(np.abs(np.diff(np.sign(f)))) / 2
    return rates


__all__ = ["emd", "find_peaks", "envelope", "zero_crossing_rate"]
