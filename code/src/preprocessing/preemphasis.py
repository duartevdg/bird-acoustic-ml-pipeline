"""
Pre-emphasis Audio Processing Toolkit
=============================

This module applies or removes pre-emphasis to audio files. It supports:
- Single-file processing
- Dataset-level processing with parallel execution
- Automatic WAV output with preserved sample rate
"""

import numpy as np
import os
from pathlib import Path
from joblib import Parallel, delayed
import logging
import librosa
import soundfile as sf
from typing import List

# ==============================
# LOGGING SETUP
# ==============================
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ==============================
# PRE-EMPHASIS FUNCTIONS
# ==============================
def apply_preemphasis(y: np.ndarray, coeff: float = 0.97) -> np.ndarray:
    """Apply pre-emphasis filter to a waveform."""
    if coeff in (0.0, None):
        return y
    return np.append(y[0], y[1:] - coeff * y[:-1])

def undo_preemphasis(y: np.ndarray, coeff: float = 0.97) -> np.ndarray:
    """Reconstruct the original waveform from a pre-emphasized signal."""
    if coeff in (0.0, None):
        return y
    out = np.zeros_like(y)
    out[0] = y[0]
    for n in range(1, len(y)):
        out[n] = y[n] + coeff * out[n - 1]
    return out

def apply_preemphasis_to_file(input_path: str, output_path: str, coeff: float = 0.97):
    """Apply pre-emphasis to a single audio file and save as WAV."""
    y, sr = librosa.load(input_path, sr=None, mono=True)
    y_preemph = apply_preemphasis(y, coeff)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    sf.write(output_path, y_preemph, sr)

# ==============================
# DATASET-LEVEL PROCESSING
# ==============================
def apply_preemphasis_to_dataset(input_folder: str, output_folder: str, coeff: float = 0.97, n_jobs: int = -1) -> List[str]:
    """
    Apply pre-emphasis to all audio files in a dataset.

    Args:
        input_folder: Path to raw audio files
        output_folder: Path to save pre-emphasized files
        coeff: Pre-emphasis coefficient
        n_jobs: Number of parallel jobs (-1 uses all cores)

    Returns:
        List of successfully processed file paths
    """
    audio_files = [f for f in Path(input_folder).rglob("*.*") 
                   if f.suffix.lower() in [".wav", ".mp3", ".flac", ".ogg"]]

    def process_file(file_path: Path):
        try:
            rel_path = file_path.relative_to(input_folder)
            out_path = Path(output_folder) / rel_path.with_suffix(".wav")
            os.makedirs(out_path.parent, exist_ok=True)
            apply_preemphasis_to_file(str(file_path), str(out_path), coeff)
            return str(out_path), "success"
        except Exception as e:
            return str(file_path), f"failed: {e}"

    results = Parallel(n_jobs=n_jobs)(delayed(process_file)(f) for f in audio_files)

    success_files = [f for f, status in results if status == "success"]
    failed_files = [f for f, status in results if status != "success"]

    if failed_files:
        logger.warning(f"{len(failed_files)} files failed during pre-emphasis:")
        for f in failed_files:
            logger.warning(f"  {f}")

    logger.info(f"Pre-emphasis applied to {len(success_files)}/{len(audio_files)} files successfully")
    return success_files