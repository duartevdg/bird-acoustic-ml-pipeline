"""
Bird Audio Normalization Toolkit
================================

This module performs audio normalization, including:
- Resampling to a target sample rate
- Converting to mono or stereo
- Format and bit depth conversion
"""

import os
from glob import glob
import numpy as np
import librosa
import soundfile as sf
from joblib import Parallel, delayed
import logging
from typing import List, Dict

# ==============================
# LOGGING SETUP
# ==============================
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ==============================
# HELPER FUNCTIONS
# ==============================
def load_audio(input_path: str):
    """Load audio with librosa, preserving original sample rate and channels."""
    return librosa.load(input_path, sr=None, mono=False)

def resample_audio(y: np.ndarray, orig_sr: int, target_sr) -> tuple[np.ndarray, int]:
    """Resample audio if target sample rate is specified."""
    if target_sr == "none" or target_sr is None or orig_sr == target_sr:
        return y, orig_sr
    return librosa.resample(y, orig_sr=orig_sr, target_sr=target_sr), target_sr

def convert_channels(y: np.ndarray, target_channels: str) -> np.ndarray:
    """Convert audio to mono or stereo if required."""
    if target_channels == "none":
        return y
    if target_channels == "mono" and y.ndim > 1:
        return librosa.to_mono(y)
    if target_channels == "stereo" and y.ndim == 1:
        return np.stack([y, y], axis=0)
    return y

def save_audio(y: np.ndarray, sr: int, path: str, format_out: str, bit_depth: int):
    """Save audio file in desired format and bit depth."""
    if y.ndim > 1:
        y = y.T
    bit_depth_map = {16: "PCM_16", 24: "PCM_24", 32: "PCM_32"}
    subtype = bit_depth_map.get(bit_depth, "PCM_16")
    sf.write(path, y, sr, format=format_out.upper(), subtype=subtype)

def normalize_audio(input_path: str, output_path: str, config: Dict):
    """Perform full audio normalization based on configuration."""
    format_out = config.get("format", "wav")
    target_sr = config.get("sampling_rate", None)
    bit_depth = config.get("bit_depth", 24)
    target_channels = config.get("channels", "mono")

    y, orig_sr = load_audio(input_path)
    y, sr = resample_audio(y, orig_sr, target_sr)
    y = convert_channels(y, target_channels)

    # Ensure output path has correct extension
    if not output_path.lower().endswith(f".{format_out}"):
        output_path = f"{output_path}.{format_out}"

    save_audio(y, sr, output_path, format_out, bit_depth)

# ==============================
# DATASET-LEVEL NORMALIZATION
# ==============================
def normalize_dataset(input_folder: str, output_folder: str, norm_config: Dict, n_jobs: int = -1) -> List[str]:
    """
    Normalize all audio files in a folder recursively.
    Returns a list of successfully normalized files.
    """
    os.makedirs(output_folder, exist_ok=True)

    audio_files = [f for f in glob(os.path.join(input_folder, '**', '*.*'), recursive=True)
                   if f.lower().endswith(('.wav', '.mp3', '.flac', '.ogg'))]

    if not audio_files:
        logger.warning(f"No audio files found in {input_folder}")
        return []

    def process_file(file_path):
        try:
            rel_path = os.path.relpath(file_path, input_folder)
            out_dir = os.path.join(output_folder, os.path.dirname(rel_path))
            os.makedirs(out_dir, exist_ok=True)
            out_path = os.path.join(out_dir, os.path.splitext(os.path.basename(file_path))[0])
            normalize_audio(file_path, out_path, norm_config)
            return out_path + f".{norm_config.get('format', 'wav')}", "success"
        except Exception as e:
            return file_path, f"failed: {str(e)}"

    results = Parallel(n_jobs=n_jobs, prefer="processes")(delayed(process_file)(f) for f in audio_files)

    failed = [r for r in results if r[1] != "success"]
    if failed:
        logger.warning(f"{len(failed)} files failed during normalization:")
        for f, msg in failed:
            logger.warning(f"  {f}: {msg}")

    success_files = [f for f, status in results if status == "success"]
    logger.info(f"Normalized {len(success_files)}/{len(audio_files)} files successfully")
    return success_files

