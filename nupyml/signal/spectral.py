"""Short-time Fourier features: STFT, spectrogram, mel-spectrogram, MFCC.

A single Fourier transform assumes the signal is stationary. Speech and music are
not -- their spectrum changes constantly -- so we chop the signal into short
overlapping frames and Fourier-transform each. The result is a TIME x FREQUENCY
picture, the foundation of essentially all audio features.
"""
import numpy as np


def _frame(x, frame_length, hop):
    n_frames = 1 + max(0, (len(x) - frame_length) // hop)
    idx = np.arange(frame_length)[None, :] + hop * np.arange(n_frames)[:, None]
    return x[idx]                                    # (n_frames, frame_length)


def stft(x, frame_length=256, hop=None, window="hann"):
    """Short-time Fourier transform: FFT of each windowed frame.

    Returns a complex ``(n_frames, n_freqs)`` array. A window (Hann by default)
    tapers each frame to suppress the spectral leakage that a hard rectangular cut
    would cause. ``hop`` (default ``frame_length//4``) sets the overlap -- the
    time-frequency resolution trade-off lives here.
    """
    x = np.asarray(x, float)
    hop = hop or frame_length // 4
    frames = _frame(x, frame_length, hop)
    if window == "hann":
        w = 0.5 - 0.5 * np.cos(2 * np.pi * np.arange(frame_length) / frame_length)
    else:
        w = np.ones(frame_length)
    return np.fft.rfft(frames * w[None, :], axis=1)


def spectrogram(x, frame_length=256, hop=None, window="hann"):
    """Power spectrogram ``|STFT|^2`` -- energy at each time and frequency."""
    return np.abs(stft(x, frame_length, hop, window)) ** 2


def _mel_filterbank(n_filters, n_fft, sr):
    """Triangular filters equally spaced on the MEL scale (perceptual pitch)."""
    def hz_to_mel(f):
        return 2595 * np.log10(1 + f / 700.0)

    def mel_to_hz(m):
        return 700 * (10 ** (m / 2595.0) - 1)

    n_freqs = n_fft // 2 + 1
    mel_pts = np.linspace(hz_to_mel(0), hz_to_mel(sr / 2), n_filters + 2)
    hz_pts = mel_to_hz(mel_pts)
    bins = np.floor((n_fft + 1) * hz_pts / sr).astype(int)
    bins = np.clip(bins, 0, n_freqs - 1)
    fb = np.zeros((n_filters, n_freqs))
    for m in range(1, n_filters + 1):
        lo, mid, hi = bins[m - 1], bins[m], bins[m + 1]
        for k in range(lo, mid):
            if mid > lo:
                fb[m - 1, k] = (k - lo) / (mid - lo)
        for k in range(mid, hi):
            if hi > mid:
                fb[m - 1, k] = (hi - k) / (hi - mid)
    return fb


def mel_spectrogram(x, sr=16000, frame_length=256, hop=None, n_filters=26):
    """Spectrogram warped onto the mel scale -- frequency as the ear hears it.

    Human pitch perception is roughly logarithmic: we resolve low frequencies
    finely and high ones coarsely. Projecting the power spectrogram through
    triangular mel filters mimics that, compressing the spectrum into a
    perceptually meaningful, lower-dimensional representation -- the input to MFCC.
    """
    ps = spectrogram(x, frame_length, hop)
    fb = _mel_filterbank(n_filters, frame_length, sr)
    return ps @ fb.T                                 # (n_frames, n_filters)


def mfcc(x, sr=16000, frame_length=256, hop=None, n_filters=26, n_coeffs=13):
    """Mel-Frequency Cepstral Coefficients -- the classic speech feature.

    THE PIPELINE
    ------------
    mel-spectrogram -> log -> discrete cosine transform, keep the first few
    coefficients. The log mimics loudness perception; the DCT DECORRELATES the mel
    bands (adjacent mel filters overlap, so their energies are correlated) and
    compacts the spectral ENVELOPE -- the slowly-varying shape that encodes which
    phoneme was spoken -- into the low-order coefficients, discarding the fine
    detail (pitch, excitation). Keeping ~13 coefficients gives a compact,
    decorrelated descriptor that dominated speech recognition for decades.

    Returns ``(n_frames, n_coeffs)``.
    """
    from scipy.fft import dct
    mel = mel_spectrogram(x, sr, frame_length, hop, n_filters)
    log_mel = np.log(mel + 1e-10)
    return dct(log_mel, type=2, axis=1, norm="ortho")[:, :n_coeffs]


__all__ = ["stft", "spectrogram", "mel_spectrogram", "mfcc"]
