"""Matrix profile: the all-subsequences nearest-neighbour distance profile.

WHAT IT IS
----------
For a window length ``m``, the matrix profile stores, at each position ``i``, the
z-normalised Euclidean distance from the subsequence starting at ``i`` to its
NEAREST other subsequence in the series (its index is kept too). One array that
answers, for every location: "how unique is the shape here?".

WHY IT IS SO USEFUL
-------------------
Two reads of the same array give the two things you usually want:

* **Motifs** -- the LOWEST points of the profile are the pair of most-similar
  subsequences: a repeated pattern.
* **Discords** -- the HIGHEST point is the subsequence least like anything else
  in the series: the anomaly / novelty.

So motif discovery and anomaly detection fall out of a single computation, with
no training and no labels.

THE MASS TRICK
--------------
The z-normalised distance from one query to ALL windows is computed from a single
sliding dot product, which is a convolution -- so the FFT evaluates it in
``O(n log n)`` instead of the naive ``O(n*m)`` per query. Looping that over all
queries (the STAMP algorithm) gives the profile. A trivial-match exclusion zone
around ``i`` stops a subsequence matching its own near-neighbours.

Yeh et al. (2016).
"""
import numpy as np


def _mass(query, series, cumsum_precomp=None):
    """Distance profile: z-normalised Euclidean distance from ``query`` to every
    length-``len(query)`` window of ``series`` (via one FFT-based dot product)."""
    m = len(query)
    n = len(series)
    q = (query - query.mean()) / (query.std() + 1e-12)
    # sliding means and stds of the series windows
    s = np.concatenate([[0.0], np.cumsum(series)])
    s2 = np.concatenate([[0.0], np.cumsum(series ** 2)])
    win_sum = s[m:] - s[:-m]
    win_sum2 = s2[m:] - s2[:-m]
    win_mean = win_sum / m
    win_std = np.sqrt(np.maximum(win_sum2 / m - win_mean ** 2, 1e-12))
    # sliding dot product query . window via FFT convolution
    qr = q[::-1]
    conv = np.fft.irfft(np.fft.rfft(series, n) * np.fft.rfft(qr, n), n)
    dot = conv[m - 1:n]
    # z-normalised squared distance reduces to this closed form
    dist2 = 2 * m * (1 - (dot - m * win_mean * q.mean()) / (m * win_std))
    return np.sqrt(np.maximum(dist2, 0))


def matrix_profile(series, window):
    """Return (profile, index): for each window, its nearest-neighbour distance
    and the position of that neighbour. Trivial (adjacent) matches are excluded.
    """
    series = np.asarray(series, dtype=float)
    n = len(series)
    n_sub = n - window + 1
    if n_sub < 2:
        raise ValueError("series too short for this window")
    exclusion = max(1, window // 2)
    profile = np.full(n_sub, np.inf)
    index = np.zeros(n_sub, dtype=int)
    for i in range(n_sub):
        d = _mass(series[i:i + window], series)
        lo, hi = max(0, i - exclusion), min(n_sub, i + exclusion + 1)
        d[lo:hi] = np.inf                            # exclude trivial matches
        j = int(np.argmin(d))
        profile[i] = d[j]
        index[i] = j
    return profile, index


def motif(series, window):
    """Index pair (i, j) of the most similar subsequences -- a repeated motif."""
    profile, index = matrix_profile(series, window)
    i = int(np.argmin(profile))
    return i, int(index[i])


def discord(series, window):
    """Index of the most anomalous subsequence -- the largest profile value."""
    profile, _ = matrix_profile(series, window)
    return int(np.argmax(profile))


__all__ = ["matrix_profile", "motif", "discord"]
