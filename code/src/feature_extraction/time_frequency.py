"""
Time-Frequency Feature Extraction Module

Extracts various time-frequency features from an audio file, including:
STFT, Mel spectrogram (log-scale), CQT, chromagram, and tempogram.
Aggregates mean, std, skewness, and kurtosis for each feature where applicable.
"""

import numpy as np
import librosa
from scipy.stats import skew, kurtosis

def extract_time_frequency_features(file_path: str,
                                    sr: int = 22050,
                                    n_fft: int = 1024,
                                    hop_length: int = 256,
                                    n_mels: int = 40,
                                    n_cqt_bins: int = 60,
                                    var_thresh: float = 1e-6) -> dict:
    """
    Extracts time-frequency features from an audio file.

    Args:
        file_path (str): Path to the audio file.
        sr (int): Sampling rate for loading audio.
        n_fft (int): FFT window size for STFT and Mel spectrogram.
        hop_length (int): Hop length for frame-based features.
        n_mels (int): Number of Mel bands.
        n_cqt_bins (int): Number of CQT bins.
        var_thresh (float): Minimum variance threshold for skew/kurtosis.

    Returns:
        dict: Dictionary of aggregated time-frequency features.
    """
    y, _ = librosa.load(file_path, sr=sr, mono=True)
    if len(y) < n_fft:
        y = np.pad(y, (0, n_fft - len(y)), mode="constant")

    feats = {}

    # Helper function to compute statistics
    def compute_stats(matrix, prefix):
        mean = np.mean(matrix, axis=1)
        std = np.std(matrix, axis=1)
        var = np.var(matrix, axis=1)
        skewness = np.array([skew(row) if v >= var_thresh else 0.0 for row, v in zip(matrix, var)])
        kurt = np.array([kurtosis(row) if v >= var_thresh else 0.0 for row, v in zip(matrix, var)])
        stats = {}
        for i, (m, s, sk, k) in enumerate(zip(mean, std, skewness, kurt)):
            stats.update({
                f"{prefix}_mean_{i}": m,
                f"{prefix}_std_{i}": s,
                f"{prefix}_skew_{i}": sk,
                f"{prefix}_kurt_{i}": k
            })
        return stats

    # --- STFT ---
    stft = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop_length))
    feats.update(compute_stats(stft, "stft"))

    # --- Mel / Log-Mel Spectrogram ---
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    feats.update(compute_stats(mel_db, "mel"))

    # --- Constant-Q Transform (CQT) ---
    cqt = np.abs(librosa.cqt(y=y, sr=sr, n_bins=n_cqt_bins, hop_length=hop_length))
    feats.update({f"cqt_mean_{i}": np.mean(row) for i, row in enumerate(cqt)})
    feats.update({f"cqt_var_{i}": np.var(row) for i, row in enumerate(cqt)})

    # --- Chromagram ---
    chroma = librosa.feature.chroma_stft(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length)
    feats.update(compute_stats(chroma, "chroma"))

    # --- Tempogram ---
    onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop_length)
    tempogram = librosa.feature.tempogram(onset_envelope=onset_env, sr=sr, hop_length=hop_length)
    feats.update({f"tempogram_mean_{i}": np.mean(row) for i, row in enumerate(tempogram)})
    feats.update({f"tempogram_var_{i}": np.var(row) for i, row in enumerate(tempogram)})

    return feats
