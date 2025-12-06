"""
Bird Audio Augmentation Toolkit
===============================

This module provides multiple waveform-level augmentation techniques for bird audio
datasets. Supported augmentations include:

- Pitch shift
- Time stretch / speed perturbation
- Noise injection (Gaussian or random)
- Time shifting
- Dynamic range compression
- Reverberation
- Time/frequency masking
- Mixup and CutMix
- Waveform mixing and time warping
"""

import os
import random
from glob import glob
from typing import Optional, List

import numpy as np
import librosa
import soundfile as sf
import scipy.signal
from tqdm import tqdm
import logging

# ==============================
# LOGGING SETUP
# ==============================
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ==============================
# AUGMENTATION FUNCTIONS
# ==============================
def add_gaussian_noise(y: np.ndarray, snr_db: float = 10) -> np.ndarray:
    """Add Gaussian noise to a waveform given a target SNR in dB."""
    rms_signal = np.sqrt(np.mean(y ** 2))
    snr_linear = 10 ** (snr_db / 10)
    rms_noise = rms_signal / np.sqrt(snr_linear)
    noise = np.random.normal(0, rms_noise, y.shape)
    return y + noise

def noise_injection(y: np.ndarray, noise_level: float = 0.005) -> np.ndarray:
    """Add low-level random noise to a waveform."""
    return y + np.random.randn(len(y)) * noise_level

def time_stretch_audio(y: np.ndarray, rate: float = 1.0) -> np.ndarray:
    """Stretch or compress waveform in time (1D mono signal only)."""
    if y.ndim != 1:
        raise ValueError("Time stretch requires a 1D mono waveform.")
    return librosa.effects.time_stretch(y, rate)

def pitch_shift(y: np.ndarray, sr: int, n_steps: int = 0) -> np.ndarray:
    """Shift the pitch of a waveform by n_steps (semitones)."""
    return librosa.effects.pitch_shift(y=y, sr=sr, n_steps=n_steps)

def time_shift(y: np.ndarray, shift_max: float = 0.2) -> np.ndarray:
    """Randomly shift a waveform in time."""
    shift_amt = int(len(y) * random.uniform(-shift_max, shift_max))
    return np.roll(y, shift_amt)

def random_crop(y: np.ndarray, crop_ratio: float = 0.8) -> np.ndarray:
    """Randomly crop a portion of the waveform."""
    crop_len = int(len(y) * crop_ratio)
    start = random.randint(0, len(y) - crop_len)
    return y[start:start + crop_len]

def dynamic_range_compression(y: np.ndarray, C: float = 1.0, clip_val: float = 0.9) -> np.ndarray:
    """Apply logarithmic dynamic range compression to waveform."""
    compressed = C * np.log1p(np.abs(y)) * np.sign(y)
    return np.clip(compressed, -clip_val, clip_val)

def reverberate(y: np.ndarray, sr: int, reverb_scale: float = 0.3) -> np.ndarray:
    """Apply a simple reverberation effect using a decaying impulse response."""
    ir_len = int(sr * 0.03)  # 30 ms IR
    ir = np.logspace(0, -3, ir_len) * reverb_scale
    return scipy.signal.convolve(y, ir, mode='full')[:len(y)]

def time_mask(y: np.ndarray, time_mask_param: int = 50) -> np.ndarray:
    """Randomly zero out a segment of the waveform (time masking)."""
    t = random.randint(0, time_mask_param)
    t0 = random.randint(0, len(y) - t)
    y_masked = y.copy()
    y_masked[t0:t0 + t] = 0
    return y_masked

def frequency_mask(y: np.ndarray, sr: int, freq_mask_param: int = 15) -> np.ndarray:
    """Randomly mask a frequency band in the STFT magnitude."""
    Y = librosa.stft(y)
    S = np.abs(Y)
    num_freq_bins = S.shape[0]
    f = random.randint(0, freq_mask_param)
    f0 = random.randint(0, num_freq_bins - f)
    S[f0:f0 + f, :] = 0
    S_complex = S * np.exp(1j * np.angle(Y))
    return librosa.istft(S_complex)

# ==============================
# DATASET AUGMENTATION
# ==============================
def augment_dataset(input_folder: str,
                    output_folder: Optional[str] = None,
                    sr: int = 22050,
                    num_augments: int = 2,
                    methods: Optional[List[str]] = None) -> str:
    """
    Apply random augmentations to all WAV files in a dataset folder.

    Parameters
    ----------
    input_folder : str
        Path to folder containing WAV files.
    output_folder : str, optional
        Folder to save augmented files. Defaults to input_folder+"_augmented".
    sr : int
        Sampling rate to load audio.
    num_augments : int
        Number of augmented versions per file.
    methods : list[str], optional
        List of augmentation methods to apply (subset of available methods).

    Returns
    -------
    str
        Path to the folder containing augmented files.
    """
    if methods is None:
        methods = [
            "pitch_shift", "time_stretch", "noise_injection", "add_gaussian_noise",
            "time_shift", "dynamic_range_compression", "reverberate",
            "time_mask", "frequency_mask"
        ]

    if output_folder is None:
        output_folder = input_folder.rstrip("/\\") + "_augmented"
    os.makedirs(output_folder, exist_ok=True)

    wav_files = glob(os.path.join(input_folder, "**", "*.wav"), recursive=True)

    if not wav_files:
        logger.warning(f"No WAV files found in {input_folder}")
        return output_folder

    aug_funcs = {
        "pitch_shift": lambda y: (pitch_shift(y, sr, n_steps=random.choice([-2, -1, 1, 2])), f"ps"),
        "time_stretch": lambda y: (time_stretch_audio(y, rate=random.uniform(0.8, 1.2)), f"ts"),
        "noise_injection": lambda y: (noise_injection(y, noise_level=0.005), "ni"),
        "add_gaussian_noise": lambda y: (add_gaussian_noise(y, snr_db=10), "agn"),
        "time_shift": lambda y: (time_shift(y), "tshift"),
        "dynamic_range_compression": lambda y: (dynamic_range_compression(y), "drc"),
        "reverberate": lambda y: (reverberate(y, sr), "rvb"),
        "time_mask": lambda y: (time_mask(y.copy()), "tmask"),
        "frequency_mask": lambda y: (frequency_mask(y, sr), "fmask"),
    }

    for file_path in tqdm(wav_files, desc="Augmenting dataset"):
        y, _ = librosa.load(file_path, sr=sr)
        base_name = os.path.splitext(os.path.basename(file_path))[0]
        relative_dir = os.path.dirname(os.path.relpath(file_path, input_folder))
        target_dir = os.path.join(output_folder, relative_dir)
        os.makedirs(target_dir, exist_ok=True)

        for i in range(num_augments):
            method = random.choice(methods)
            y_aug, tag = aug_funcs.get(method, lambda y: (y, "none"))(y)
            out_path = os.path.join(target_dir, f"{base_name}_aug_{tag}_{i}.wav")
            sf.write(out_path, y_aug, sr)

    logger.info(f"Augmentation complete. Augmented dataset saved at: {output_folder}")
    return output_folder
