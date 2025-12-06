"""
Frequency-Domain Feature Extraction Module

Extracts various spectral features from an audio file, including spectral
centroid, bandwidth, flatness, rolloff, contrast, flux, entropy, and tonal
features (Tonnetz). Aggregates frame-level features into statistics.
"""

import numpy as np
import librosa
from scipy.stats import skew, kurtosis

def extract_frequency_features(file_path: str,
                               sr: int = 22050,
                               n_fft: int = 1024,
                               hop_length: int = 256,
                               n_mfcc: int = 40,
                               var_thresh: float = 1e-6) -> dict:
    """
    Extracts frequency-domain features from an audio file.

    Args:
        file_path (str): Path to the audio file.
        sr (int, optional): Target sampling rate. Defaults to 22050.
        n_fft (int, optional): FFT window size. Defaults to 1024.
        hop_length (int, optional): Hop length for STFT. Defaults to 256.
        n_mfcc (int, optional): Number of MFCCs (not used in this function but reserved). Defaults to 40.
        var_thresh (float, optional): Minimum variance threshold to compute skew/kurtosis. Defaults to 1e-6.

    Returns:
        dict: Dictionary of aggregated spectral features.
    """
    # Load audio
    y, _ = librosa.load(file_path, sr=sr, mono=True)
    if len(y) < n_fft:
        y = np.pad(y, (0, n_fft - len(y)), mode="constant")

    feats = {}

    # Compute magnitude spectrogram
    S = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop_length))

    # Helper: compute aggregate statistics
    def stats(vec):
        v = np.var(vec)
        return {
            "mean": np.mean(vec),
            "std": np.std(vec),
            "skew": skew(vec) if v >= var_thresh else 0.0,
            "kurt": kurtosis(vec) if v >= var_thresh else 0.0
        }

    # Spectral features
    spectral_features = [
        ("centroid", librosa.feature.spectral_centroid(S=S, sr=sr)),
        ("bandwidth", librosa.feature.spectral_bandwidth(S=S, sr=sr)),
        ("flatness", librosa.feature.spectral_flatness(S=S)),
        ("rolloff", librosa.feature.spectral_rolloff(S=S, sr=sr, roll_percent=0.85)),
        ("contrast", librosa.feature.spectral_contrast(S=S, sr=sr))
    ]

    for name, mat in spectral_features:
        st = stats(mat[0])
        feats.update({f"{name}_{k}": v for k, v in st.items()})

    # Spectral flux (onset strength)
    flux = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop_length)
    st = stats(flux)
    feats.update({f"flux_{k}": v for k, v in st.items()})

    # Spectral entropy
    S_norm = S / (np.sum(S, axis=0, keepdims=True) + np.finfo(float).eps)
    entropy = -np.sum(S_norm * np.log2(S_norm + np.finfo(float).eps), axis=0)
    st = stats(entropy)
    feats.update({f"entropy_{k}": v for k, v in st.items()})

    # Tonnetz (tonal features)
    chroma = librosa.feature.chroma_stft(y=y, sr=sr, hop_length=hop_length)
    tonnetz = librosa.feature.tonnetz(chroma=chroma, sr=sr)
    for i, row in enumerate(tonnetz):
        st = stats(row)
        feats.update({f"tonnetz{i+1}_{k}": v for k, v in st.items()})

    return feats
