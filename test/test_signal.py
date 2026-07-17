"""F10: wavelets, STFT/spectrogram/MFCC, EMD, and detection utilities.

Each is held to its property: the DWT reconstructs perfectly and denoising lowers
MSE; the spectrogram peaks at the true tone frequency; EMD sums back to the
signal and orders IMFs fast-to-slow; the envelope tracks AM modulation; the
zero-crossing rate rises with noisiness.
"""
import numpy as np
import pytest

from nupyml.signal import (dwt, idwt, wavelet_denoise, stft, spectrogram,
                          mel_spectrogram, mfcc, emd, find_peaks, envelope,
                          zero_crossing_rate)


# --- wavelets -------------------------------------------------------------

@pytest.mark.parametrize("wavelet", ["haar", "db2"])
def test_dwt_perfect_reconstruction(wavelet):
    rng = np.random.RandomState(0)
    x = rng.randn(128)
    coeffs = dwt(x, wavelet, level=3)
    rec = idwt(coeffs, wavelet)
    assert np.allclose(rec[:len(x)], x, atol=1e-9)


def test_dwt_levels_structure():
    x = np.random.RandomState(0).randn(64)
    coeffs = dwt(x, "haar", level=3)
    assert len(coeffs) == 4                          # cA3, cD3, cD2, cD1
    # detail lengths double from coarse to fine
    assert len(coeffs[1]) <= len(coeffs[2]) <= len(coeffs[3])


def test_wavelet_denoise_reduces_noise():
    rng = np.random.RandomState(0)
    t = np.linspace(0, 1, 512)
    clean = np.sin(2 * np.pi * 3 * t)
    noisy = clean + 0.4 * rng.randn(512)
    den = wavelet_denoise(noisy, "db2")
    assert np.mean((den - clean) ** 2) < np.mean((noisy - clean) ** 2)


# --- spectral -------------------------------------------------------------

def test_spectrogram_peaks_at_the_tone_frequency():
    sr = 8000
    t = np.arange(sr) / sr
    frame = 256
    x = np.sin(2 * np.pi * 440 * t)                  # pure 440 Hz tone
    S = spectrogram(x, frame_length=frame)
    dominant_bin = S.mean(axis=0).argmax()
    freq = dominant_bin * sr / frame
    assert abs(freq - 440) < sr / frame              # within one bin


def test_stft_and_mfcc_shapes():
    sr = 8000
    x = np.random.RandomState(0).randn(sr)
    S = stft(x, frame_length=256)
    assert S.shape[1] == 129 and np.iscomplexobj(S)
    M = mfcc(x, sr=sr, n_coeffs=13)
    assert M.shape[1] == 13 and np.all(np.isfinite(M))


def test_mel_spectrogram_is_nonnegative():
    x = np.random.RandomState(0).randn(4000)
    mel = mel_spectrogram(x, sr=8000, n_filters=26)
    assert (mel >= 0).all() and mel.shape[1] == 26


def test_mfcc_distinguishes_two_tones():
    sr = 8000
    t = np.arange(sr) / sr
    a = mfcc(np.sin(2 * np.pi * 300 * t), sr=sr).mean(axis=0)
    b = mfcc(np.sin(2 * np.pi * 2000 * t), sr=sr).mean(axis=0)
    assert np.linalg.norm(a - b) > 1.0               # different spectra -> different MFCC


# --- EMD ------------------------------------------------------------------

def test_emd_reconstructs_the_signal_exactly():
    t = np.linspace(0, 1, 400)
    x = np.sin(2 * np.pi * 5 * t) + 0.5 * np.sin(2 * np.pi * 40 * t) + 0.3 * t
    imfs = emd(x, max_imfs=6)
    assert np.allclose(np.sum(imfs, axis=0), x, atol=1e-8)


def test_emd_orders_imfs_fast_to_slow():
    t = np.linspace(0, 1, 400)
    x = np.sin(2 * np.pi * 5 * t) + 0.5 * np.sin(2 * np.pi * 40 * t)
    imfs = emd(x, max_imfs=6)

    def n_crossings(s):
        return np.sum(np.abs(np.diff(np.sign(s)))) / 2
    # the first IMF should oscillate faster than the trend residual
    assert n_crossings(imfs[0]) > n_crossings(imfs[-1])


# --- detection ------------------------------------------------------------

def test_find_peaks_height_and_distance():
    x = np.array([0, 2, 0, 3, 0, 1, 0.0])
    assert find_peaks(x, height=1.5).tolist() == [1, 3]
    # with a large distance only the tallest survives
    assert find_peaks(x, distance=5).tolist() == [3]


def test_envelope_tracks_am_modulation():
    t = np.linspace(0, 1, 800)
    modulation = 1 + 0.5 * np.sin(2 * np.pi * 3 * t)
    am = modulation * np.sin(2 * np.pi * 200 * t)
    env = envelope(am)
    assert np.corrcoef(env, modulation)[0, 1] > 0.9


def test_zero_crossing_rate_higher_for_noise_than_tone():
    rng = np.random.RandomState(0)
    t = np.linspace(0, 1, 2000)
    tone = np.sin(2 * np.pi * 5 * t)                 # slow -> few crossings
    noise = rng.randn(2000)                          # many crossings
    assert zero_crossing_rate(noise).mean() > zero_crossing_rate(tone).mean()
