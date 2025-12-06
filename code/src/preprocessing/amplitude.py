"""
Amplitude Normalization for Bird Audio Dataset
==============================================

This script recursively normalizes the amplitude of audio files in a dataset.
Supported formats: WAV, MP3, FLAC, OGG. Output files are saved in WAV format
with the same directory structure.

Normalization methods:
- "none" : No normalization.
- "peak" : Normalize by peak amplitude.
- "rms"  : Normalize by root-mean-square (RMS) amplitude.
"""

import os
from glob import glob
from pathlib import Path
import logging

import librosa
import numpy as np
import soundfile as sf
from joblib import Parallel, delayed

# ==============================
# LOGGING SETUP
# ==============================
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ==============================
# HELPER FUNCTIONS
# ==============================
def normalize_amplitude(y: np.ndarray, method: str = "none") -> np.ndarray:
    """
    Normalize audio signal amplitude.

    Parameters
    ----------
    y : np.ndarray
        Audio signal array (mono or multi-channel).
    method : str
        Normalization method: "none", "peak", or "rms".

    Returns
    -------
    np.ndarray
        Normalized audio array.
    """
    if method == "none":
        return y
    if method == "peak":
        peak = np.max(np.abs(y))
        return y / peak if peak > 0 else y
    if method == "rms":
        rms = np.sqrt(np.mean(y ** 2))
        return y / rms if rms > 0 else y
    raise ValueError(f"Unknown normalization method: {method}")


def normalize_file(input_path: str, output_path: str, method: str = "peak") -> str:
    """
    Normalize a single audio file and save as WAV.

    Parameters
    ----------
    input_path : str
        Path to the input audio file.
    output_path : str
        Path to save the normalized audio file.
    method : str
        Normalization method.

    Returns
    -------
    str
        Path to the saved normalized file.
    """
    y, sr = librosa.load(input_path, sr=None, mono=False)
    y = normalize_amplitude(y, method)
    if y.ndim > 1:  # librosa loads multi-channel as (channels, samples)
        y = y.T
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    sf.write(output_path, y, sr)
    return output_path


def amp_normalize_dataset(input_folder: str, output_folder: str,
                          method: str = "peak", n_jobs: int = -1) -> list[str]:
    """
    Recursively normalize all audio files in a dataset folder.

    Parameters
    ----------
    input_folder : str
        Root folder of raw audio files.
    output_folder : str
        Folder to save normalized audio files (maintains directory structure).
    method : str
        Normalization method: "none", "peak", or "rms".
    n_jobs : int
        Number of parallel jobs (-1 uses all cores).

    Returns
    -------
    list[str]
        List of successfully normalized files.
    """
    input_folder = os.path.abspath(input_folder)
    output_folder = os.path.abspath(output_folder)
    os.makedirs(output_folder, exist_ok=True)

    # Recursively collect audio files
    audio_files = [os.path.abspath(f) for f in glob(os.path.join(input_folder, '**', '*.*'), recursive=True)
                   if f.lower().endswith(('.wav', '.mp3', '.flac', '.ogg'))]

    if not audio_files:
        logger.warning(f"No audio files found in {input_folder}")
        return []

    def process_file(file_path: str) -> tuple[str, str]:
        try:
            rel_path = Path(file_path).relative_to(input_folder)
            out_path = Path(output_folder) / rel_path.with_suffix(".wav")  # force WAV output
            normalize_file(file_path, out_path, method)
            return str(out_path), "success"
        except Exception as e:
            return str(file_path), f"failed: {e}"

    # Parallel processing
    results = Parallel(n_jobs=n_jobs, prefer="processes")(
        delayed(process_file)(f) for f in audio_files
    )

    # Separate successes and failures
    success_files = [f for f, status in results if status == "success"]
    failed_files = [f for f, status in results if status != "success"]

    if failed_files:
        logger.warning(f"{len(failed_files)} files failed during amplitude normalization:")
        for f in failed_files:
            logger.warning(f"  {f}")

    logger.info(f"Amplitude normalization complete: {len(success_files)}/{len(audio_files)} files processed successfully")
    return success_files

