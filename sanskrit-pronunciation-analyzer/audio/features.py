"""
features.py
-----------
Acoustic feature extraction for Sanskrit phoneme recognition.

Extracts Mel-Frequency Cepstral Coefficients (MFCCs), Spectral Centroid,
Spectral Rolloff, Zero-Crossing Rate (ZCR), and energy dynamics from raw audio
using standard numpy and scipy. No heavy external dependencies required.
"""

from typing import Tuple
import numpy as np
from scipy import signal
from scipy.fft import dct


def trim_silence(samples: np.ndarray, sample_rate: int, threshold_db: float = 25.0, min_speech_ms: float = 100.0) -> np.ndarray:
    """
    Perform Voice Activity Detection (VAD) / energy-based silence trimming.
    Isolates the active phoneme utterance from lead-in and trailing silence.
    """
    if len(samples) == 0:
        return samples

    # Frame size 20ms, hop 10ms
    frame_len = int(sample_rate * 0.02)
    hop_len = int(sample_rate * 0.01)

    if len(samples) < frame_len:
        return samples

    # Calculate frame energy (RMS in dB)
    num_frames = max(1, 1 + (len(samples) - frame_len) // hop_len)
    energies = np.zeros(num_frames)

    for i in range(num_frames):
        start = i * hop_len
        frame = samples[start : start + frame_len]
        rms = np.sqrt(np.mean(frame**2) + 1e-12)
        energies[i] = 20.0 * np.log10(rms + 1e-12)

    max_energy = np.max(energies)
    threshold = max_energy - threshold_db

    speech_frames = np.where(energies > threshold)[0]
    if len(speech_frames) == 0:
        return samples

    start_idx = speech_frames[0] * hop_len
    end_idx = min(len(samples), (speech_frames[-1] + 1) * hop_len + frame_len)

    min_samples = int(sample_rate * (min_speech_ms / 1000.0))
    if (end_idx - start_idx) < min_samples:
        return samples

    # Add small 50ms padding around speech if available
    pad_samples = int(sample_rate * 0.05)
    start_idx = max(0, start_idx - pad_samples)
    end_idx = min(len(samples), end_idx + pad_samples)

    return samples[start_idx:end_idx]


def hz_to_mel(hz: np.ndarray) -> np.ndarray:
    """Convert frequency in Hz to Mel scale."""
    return 2595.0 * np.log10(1.0 + hz / 700.0)


def mel_to_hz(mel: np.ndarray) -> np.ndarray:
    """Convert frequency in Mel scale to Hz."""
    return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)


def get_mel_filterbank(num_filters: int, n_fft: int, sample_rate: int, low_freq: float = 0.0, high_freq: float = None) -> np.ndarray:
    """Construct triangular Mel filterbank matrix."""
    if high_freq is None:
        high_freq = sample_rate / 2.0

    low_mel = hz_to_mel(np.array([low_freq]))[0]
    high_mel = hz_to_mel(np.array([high_freq]))[0]
    mel_points = np.linspace(low_mel, high_mel, num_filters + 2)
    hz_points = mel_to_hz(mel_points)

    bin_points = np.floor((n_fft + 1) * hz_points / sample_rate).astype(int)
    num_bins = n_fft // 2 + 1
    filterbank = np.zeros((num_filters, num_bins))

    for m in range(1, num_filters + 1):
        f_m_minus = bin_points[m - 1]
        f_m = bin_points[m]
        f_m_plus = bin_points[m + 1]

        for k in range(f_m_minus, f_m):
            if f_m > f_m_minus:
                filterbank[m - 1, k] = (k - f_m_minus) / (f_m - f_m_minus)

        for k in range(f_m, f_m_plus):
            if f_m_plus > f_m:
                filterbank[m - 1, k] = (f_m_plus - k) / (f_m_plus - f_m)

    return filterbank


def compute_mfcc(
    samples: np.ndarray,
    sample_rate: int = 16000,
    n_mfcc: int = 13,
    n_filters: int = 26,
    n_fft: int = 512,
    hop_length: int = 160,
    win_length: int = 400,
) -> np.ndarray:
    """
    Compute Mel-Frequency Cepstral Coefficients (MFCCs) for an audio signal.
    Returns: 2D array of shape (n_mfcc, num_frames).
    """
    if len(samples) < win_length:
        # Pad short signals
        samples = np.pad(samples, (0, win_length - len(samples)), mode="constant")

    # 1. Pre-emphasis filter
    pre_emphasized = np.append(samples[0], samples[1:] - 0.97 * samples[:-1])

    # 2. Framing and Windowing
    num_frames = 1 + (len(pre_emphasized) - win_length) // hop_length
    window = np.hamming(win_length)

    frames = np.zeros((num_frames, win_length))
    for i in range(num_frames):
        start = i * hop_length
        frames[i] = pre_emphasized[start : start + win_length] * window

    # 3. FFT Magnitude and Power Spectrum
    mag_frames = np.abs(np.fft.rfft(frames, n=n_fft))
    power_frames = (mag_frames**2) / n_fft

    # 4. Mel Filterbank Application
    fb = get_mel_filterbank(n_filters, n_fft, sample_rate)
    mel_energy = np.dot(power_frames, fb.T)
    mel_energy = np.where(mel_energy == 0, np.finfo(float).eps, mel_energy)
    log_mel_energy = np.log(mel_energy)

    # 5. Discrete Cosine Transform (DCT)
    mfcc = dct(log_mel_energy, type=2, axis=1, norm="ortho")[:, :n_mfcc]

    return mfcc.T  # Shape: (n_mfcc, num_frames)


def compute_deltas(feature_matrix: np.ndarray, width: int = 5) -> np.ndarray:
    """
    Compute delta (derivative) coefficients over time frames.
    feature_matrix shape: (n_features, n_frames).
    """
    n_feats, n_frames = feature_matrix.shape
    if n_frames < width:
        return np.zeros_like(feature_matrix)

    half_w = width // 2
    padded = np.pad(feature_matrix, ((0, 0), (half_w, half_w)), mode="edge")
    deltas = np.zeros_like(feature_matrix)

    denominator = 2 * sum(i**2 for i in range(1, half_w + 1))
    for i in range(n_frames):
        delta_val = np.zeros(n_feats)
        for w in range(1, half_w + 1):
            delta_val += w * (padded[:, i + half_w + w] - padded[:, i + half_w - w])
        deltas[:, i] = delta_val / denominator

    return deltas


def compute_spectral_features(samples: np.ndarray, sample_rate: int = 16000, n_fft: int = 512, hop_length: int = 160) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute Spectral Centroid, Spectral Rolloff (85%), and Zero-Crossing Rate.
    """
    if len(samples) < n_fft:
        samples = np.pad(samples, (0, n_fft - len(samples)), mode="constant")

    num_frames = 1 + (len(samples) - n_fft) // hop_length
    freqs = np.fft.rfftfreq(n_fft, d=1.0 / sample_rate)

    centroids = np.zeros(num_frames)
    rolloffs = np.zeros(num_frames)
    zcrs = np.zeros(num_frames)

    for i in range(num_frames):
        start = i * hop_length
        frame = samples[start : start + n_fft]

        # Zero crossing rate
        zcr = np.mean(np.abs(np.diff(np.sign(frame)))) / 2.0
        zcrs[i] = zcr

        # Magnitude spectrum
        mag = np.abs(np.fft.rfft(frame * np.hamming(n_fft)))
        mag_sum = np.sum(mag) + 1e-12

        # Centroid
        centroids[i] = np.sum(freqs * mag) / mag_sum

        # Rolloff (85% of total spectral energy)
        cum_energy = np.cumsum(mag)
        threshold = 0.85 * mag_sum
        rolloff_idx = np.where(cum_energy >= threshold)[0]
        rolloffs[i] = freqs[rolloff_idx[0]] if len(rolloff_idx) > 0 else freqs[-1]

    return centroids, rolloffs, zcrs


def extract_feature_vector(samples: np.ndarray, sample_rate: int = 16000) -> np.ndarray:
    """
    Extract a comprehensive, fixed-length 1D feature vector for acoustic classification.
    Features include:
      - MFCCs (13 coefficients): mean, std, min, max
      - Delta MFCCs (13 coefficients): mean, std
      - Delta-Delta MFCCs (13 coefficients): mean, std
      - Spectral Centroid: mean, std, max
      - Spectral Rolloff: mean, std
      - Zero-Crossing Rate: mean, std, max
      - RMS Energy: mean, std, max
    Total feature length: 13*4 + 13*2 + 13*2 + 3 + 2 + 3 + 3 = 117 features.
    """
    # 1. Voice activity detection / silence trimming
    active_samples = trim_silence(samples, sample_rate)
    if len(active_samples) < 320:  # < 20ms
        active_samples = samples if len(samples) >= 320 else np.pad(samples, (0, 320 - len(samples)))

    # 2. MFCCs + Deltas
    mfccs = compute_mfcc(active_samples, sample_rate=sample_rate, n_mfcc=13)
    deltas = compute_deltas(mfccs)
    delta_deltas = compute_deltas(deltas)

    # 3. Spectral features
    centroids, rolloffs, zcrs = compute_spectral_features(active_samples, sample_rate=sample_rate)

    # 4. Energy
    frame_len = 400
    hop_len = 160
    num_frames = max(1, 1 + (len(active_samples) - frame_len) // hop_len)
    rms_energies = np.array([
        np.sqrt(np.mean(active_samples[i * hop_len : i * hop_len + frame_len] ** 2) + 1e-12)
        for i in range(num_frames)
    ])

    # 5. Statistical aggregations
    features = [
        np.mean(mfccs, axis=1),
        np.std(mfccs, axis=1),
        np.min(mfccs, axis=1),
        np.max(mfccs, axis=1),
        np.mean(deltas, axis=1),
        np.std(deltas, axis=1),
        np.mean(delta_deltas, axis=1),
        np.std(delta_deltas, axis=1),
        [np.mean(centroids), np.std(centroids), np.max(centroids)],
        [np.mean(rolloffs), np.std(rolloffs)],
        [np.mean(zcrs), np.std(zcrs), np.max(zcrs)],
        [np.mean(rms_energies), np.std(rms_energies), np.max(rms_energies)],
    ]

    return np.hstack(features).astype(np.float32)
