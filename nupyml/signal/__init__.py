"""Signal processing -- the 1-D counterpart to ``image``, feeding TSC and audio.

* ``wavelet`` -- the discrete wavelet transform (Haar, db2) and wavelet
  denoising: a time-frequency decomposition for transient, non-stationary signals
  where Fourier is blind to WHEN.
* ``spectral`` -- STFT, spectrogram, mel-spectrogram, and MFCC: the short-time
  Fourier features underlying essentially all audio recognition.
* ``decompose`` -- Empirical Mode Decomposition (a data-adaptive basis), plus
  peak finding, the Hilbert amplitude envelope, and the zero-crossing rate.

Only numpy and scipy are used.
"""
from .wavelet import dwt, idwt, wavelet_denoise
from .spectral import stft, spectrogram, mel_spectrogram, mfcc
from .decompose import emd, find_peaks, envelope, zero_crossing_rate

__all__ = ["dwt", "idwt", "wavelet_denoise",
           "stft", "spectrogram", "mel_spectrogram", "mfcc",
           "emd", "find_peaks", "envelope", "zero_crossing_rate"]
