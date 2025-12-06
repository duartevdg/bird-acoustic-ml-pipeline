"""
Cepstral Feature Extraction Module

Extracts MFCC, delta, delta-delta, and LPC features from an audio file, and
aggregates them into statistical descriptors (mean, std, skewness, kurtosis)
per feature dimension.
"""

import numpy as np
import librosa
from scipy.stats import skew, kurtosis

def extract_cepstral_features(file_path: str,
                              n_mfcc: int = 20,
                              lpc_order: int = 4,
                              delta_frames: int = 2,
                              var_thresh: float = 1e-10) -> dict:
    """
    Extracts cepstral features from an audio file.

    Args:
        file_path (str): Path to the audio file.
        n_mfcc (int, optional): Number of MFCC coefficients. Defaults to 20.
        lpc_order (int, optional): Order of LPC coefficients. Defaults to 4.
        delta_frames (int, optional): Window size for delta calculation. Defaults to 2.
        var_thresh (float, optional): Variance threshold below which skew/kurtosis is set to 0. Defaults to 1e-10.

    Returns:
        dict: Aggregated feature dictionary with keys of the form:
              'cepstral_<idx>_mean', 'cepstral_<idx>_std', 'cepstral_<idx>_skew', 'cepstral_<idx>_kurt'.
    """
    # Load audio
    signal, sr = librosa.load(file_path, sr=None, mono=True)

    # --- MFCC features ---
    mfcc = librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=n_mfcc)
    n_frames = mfcc.shape[1]

    # --- Delta and Delta-Delta ---
    mfcc_delta = librosa.feature.delta(mfcc, order=1, width=2*delta_frames+1)
    mfcc_delta2 = librosa.feature.delta(mfcc, order=2, width=2*delta_frames+1)
    mfcc_delta = mfcc_delta[:, :n_frames]
    mfcc_delta2 = mfcc_delta2[:, :n_frames]

    # --- LPC features per frame ---
    frame_len, hop_len = int(0.025*sr), int(0.010*sr)
    frames = librosa.util.frame(signal, frame_length=frame_len, hop_length=hop_len).T

    lpc_feat = []
    for frame in frames:
        try:
            coeffs = librosa.lpc(frame, order=lpc_order)
            lpc_feat.append(coeffs[1:])  # drop first coefficient
        except np.linalg.LinAlgError:
            lpc_feat.append(np.zeros(lpc_order))
    lpc_feat = np.array(lpc_feat)

    # Interpolate LPC to match MFCC frames if necessary
    if lpc_feat.shape[0] != n_frames:
        lpc_feat = np.array([
            np.interp(np.linspace(0, lpc_feat.shape[0]-1, n_frames),
                      np.arange(lpc_feat.shape[0]),
                      lpc_feat[:, i])
            for i in range(lpc_order)
        ]).T

    # --- Combine features ---
    all_features = np.hstack([mfcc.T, mfcc_delta.T, mfcc_delta2.T, lpc_feat])

    # --- Aggregate statistics ---
    feat_dict = {}
    for i in range(all_features.shape[1]):
        col = all_features[:, i]
        v = np.var(col)
        feat_dict.update({
            f"cepstral_{i+1}_mean": np.mean(col),
            f"cepstral_{i+1}_std": np.std(col),
            f"cepstral_{i+1}_skew": skew(col) if v >= var_thresh else 0.0,
            f"cepstral_{i+1}_kurt": kurtosis(col) if v >= var_thresh else 0.0
        })

    return feat_dict
