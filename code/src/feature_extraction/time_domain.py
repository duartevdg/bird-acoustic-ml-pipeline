"""
Time-Domain Feature Extraction Module

Extracts various statistical and waveform features from an audio file, 
including amplitude statistics, envelope/RMS, zero-crossing rate, autocorrelation, 
skewness, kurtosis, and signal entropy.
"""

import numpy as np
import librosa
from scipy.stats import skew, kurtosis

def extract_time_features(file_path: str,
                          hop_length: int = 512,
                          n_fft: int = 2048,
                          fixed_sr: int = 22050,
                          var_thresh: float = 1e-10) -> dict:
    """
    Extracts time-domain features from an audio file.

    Args:
        file_path (str): Path to the audio file.
        hop_length (int, optional): Hop length for frame-based features. Defaults to 512.
        n_fft (int, optional): Frame length for RMS computation. Defaults to 2048.
        fixed_sr (int, optional): Sampling rate to resample audio. Defaults to 22050.
        var_thresh (float, optional): Minimum variance threshold for skew/kurtosis. Defaults to 1e-10.

    Returns:
        dict: Dictionary of aggregated time-domain features, or None if the signal is too short or flat.
    """
    # Load audio
    y, sr = librosa.load(file_path, sr=fixed_sr, mono=True)
    if len(y) < 2 or np.allclose(y, y[0]):
        return None  # flat or too short signal

    feats = {}
    N = len(y)
    mean_y = np.mean(y)
    y_centered = y - mean_y
    rms_vals = librosa.feature.rms(y=y, frame_length=n_fft, hop_length=hop_length)[0]

    # Helper: compute aggregate statistics
    def stats(vec):
        v = np.var(vec)
        return {
            "mean": np.mean(vec),
            "std": np.std(vec),
            "skew": skew(vec) if v >= var_thresh else 0.0,
            "kurt": kurtosis(vec) if v >= var_thresh else 0.0
        }

    # --- Amplitude statistics ---
    feats.update({
        "mean_amplitude": mean_y,
        "median_amplitude": np.median(y),
        "std_amplitude": np.std(y),
        "min_amplitude": np.min(y),
        "max_amplitude": np.max(y),
        "peak_amplitude": np.max(np.abs(y)),
        "peak_to_peak": np.ptp(y),
        "variance": np.var(y),
        "mean_absolute_deviation": np.mean(np.abs(y_centered)),
        "signal_energy": np.mean(y**2),
        "nrmsd": np.sqrt(np.mean(y_centered**2)) / (np.sqrt(np.mean(y**2)) + 1e-10),
        "snr": 10 * np.log10(np.sum(y**2) / (np.sum(y_centered**2) + 1e-10)),
        "peak_to_rms_ratio": np.max(np.abs(y)) / (np.mean(rms_vals) + 1e-10)
    })

    # --- Envelope / RMS statistics ---
    env_stats = stats(rms_vals)
    feats.update({f"envelope_{k}": v for k, v in env_stats.items()})

    # --- Zero-crossing rate statistics ---
    zcr = librosa.feature.zero_crossing_rate(y, hop_length=hop_length)[0]
    zcr_stats = stats(zcr)
    feats.update({f"zcr_{k}": v for k, v in zcr_stats.items()})

    # --- Autocorrelation (lag-1) ---
    feats["autocorrelation"] = np.corrcoef(y[:-1], y[1:])[0, 1]

    # --- Skewness and kurtosis of waveform ---
    feats["skewness"] = skew(y, bias=False)
    feats["kurtosis"] = kurtosis(y, bias=False)

    # --- Entropy ---
    hist, _ = np.histogram(y, bins=100, density=True)
    hist = hist[hist > 0]
    feats["entropy"] = -np.sum(hist * np.log2(hist))

    return feats
