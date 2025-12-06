"""
Bird Audio Denoising Toolkit
============================

This module provides multiple audio denoising techniques suitable for bird
recordings. Supported methods include:

- Lowpass filtering
- Bandpass filtering
- Spectral subtraction
- Wiener filtering
- Wavelet denoising
- NoiseReduce library denoising

"""

import os
import logging
from pathlib import Path
from joblib import Parallel, delayed
from typing import List, Optional

import numpy as np
import pywt
import noisereduce as nr
from scipy.signal import butter, sosfilt, wiener
from scipy.fftpack import fft, ifft
from pyAudioAnalysis import audioBasicIO as aIO
import soundfile as sf

# ==============================
# LOGGING SETUP
# ==============================
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ==============================
# DENOISING METHODS
# ==============================
def lowpass_filter(data: np.ndarray, fs: int, cutoff: int = 4000, order: int = 5) -> np.ndarray:
    sos = butter(order, cutoff / (0.5 * fs), btype='low', output='sos')
    return sosfilt(sos, data)

def bandpass_filter(signal: np.ndarray, fs: int, lowcut: int = 1000, highcut: int = 8000, order: int = 4) -> np.ndarray:
    sos = butter(order, [lowcut / (0.5 * fs), highcut / (0.5 * fs)], btype='band', output='sos')
    return sosfilt(sos, signal)

def spectral_subtraction(signal: np.ndarray, fs: int, noise_duration: float = 0.5,
                         frame_len_ms: float = 20, hop_ms: float = 10) -> np.ndarray:
    frame_len = int(fs * frame_len_ms / 1000)
    hop_len = int(fs * hop_ms / 1000)
    noise_samples = int(fs * noise_duration)
    noise_spec = np.abs(fft(signal[:noise_samples], n=frame_len))
    
    output = np.zeros_like(signal)
    window = np.hamming(frame_len)
    win_sum = np.zeros_like(signal)

    for i in range(0, len(signal)-frame_len, hop_len):
        frame = signal[i:i+frame_len] * window
        spectrum = fft(frame)
        mag, phase = np.abs(spectrum), np.angle(spectrum)
        clean_mag = np.maximum(mag - noise_spec, 0)
        output[i:i+frame_len] += np.real(ifft(clean_mag * np.exp(1j*phase))) * window
        win_sum[i:i+frame_len] += window**2

    win_sum[win_sum == 0] = 1
    return output / win_sum

def wiener_denoise(signal: np.ndarray) -> np.ndarray:
    try:
        if not np.isfinite(signal).all() or np.var(signal) < 1e-12:
            return signal
        result = wiener(signal)
        return result if np.isfinite(result).all() else signal
    except Exception:
        return signal

def wavelet(signal: np.ndarray, wavelet_name: str = 'db8', level: int = 4) -> np.ndarray:
    coeffs = pywt.wavedec(signal, wavelet_name, level=level)
    sigma = np.median(np.abs(coeffs[-1])) / 0.6745
    uthresh = sigma * np.sqrt(2 * np.log(len(signal)))
    denoised_coeffs = [coeffs[0]] + [pywt.threshold(c, uthresh, mode='soft') for c in coeffs[1:]]
    denoised = pywt.waverec(denoised_coeffs, wavelet_name)
    if len(denoised) > len(signal):
        denoised = denoised[:len(signal)]
    else:
        denoised = np.pad(denoised, (0, len(signal) - len(denoised)), mode='constant')
    return denoised

def noisereduce(signal: np.ndarray, fs: int) -> np.ndarray:
    return nr.reduce_noise(y=signal, sr=fs)

# ==============================
# FILE-LEVEL DENOISING
# ==============================
def denoise_audio(input_path: str, output_path: str, method: str = "wiener") -> Optional[str]:
    """
    Denoise a single audio file using the specified method.
    """
    try:
        sr, signal = aIO.read_audio_file(str(input_path))
        signal = signal.astype(np.float32)
        if signal.ndim > 1:
            signal = signal.mean(axis=1)

        methods = {
            "lowpass": lambda x, fs: lowpass_filter(x, fs),
            "bandpass": lambda x, fs: bandpass_filter(x, fs),
            "spectral_sub": lambda x, fs: spectral_subtraction(x, fs),
            "wiener": lambda x, fs: wiener_denoise(x),
            "wavelet": lambda x, fs: wavelet(x),
            "noisereduce": lambda x, fs: noisereduce(x, fs)
        }

        if method not in methods:
            raise ValueError(f"Unsupported denoising method: {method}")

        denoised = methods[method](signal, sr)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        sf.write(output_path, denoised, sr)
        return output_path
    except Exception as e:
        logger.warning(f"Skipping {input_path}: {e}")
        return None

# ==============================
# DATASET-LEVEL DENOISING
# ==============================
def denoise_dataset(input_folder: str, output_folder: str, method: str = "wiener", n_jobs: int = -1) -> List[str]:
    """
    Denoise all audio files in a folder recursively.
    Returns a list of successfully denoised file paths.
    """
    os.makedirs(output_folder, exist_ok=True)
    audio_files = [f for f in Path(input_folder).rglob("*.*")
                   if f.suffix.lower() in [".wav", ".mp3", ".flac", ".ogg"]]

    if not audio_files:
        logger.warning(f"No audio files found in {input_folder}")
        return []

    def process_file(file_path):
        try:
            rel_path = os.path.relpath(file_path, input_folder)
            out_path = os.path.join(output_folder, os.path.splitext(rel_path)[0] + ".wav")
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            return denoise_audio(file_path, out_path, method), "success"
        except Exception as e:
            logger.warning(f"Skipping {file_path}: {e}")
            return None, "failed"

    results = Parallel(n_jobs=n_jobs, prefer="processes")(delayed(process_file)(f) for f in audio_files)
    success_files = [f for f, status in results if status == "success"]
    failed_files = [f for f, status in results if status != "success" and f is not None]

    if failed_files:
        logger.warning(f"{len(failed_files)} files failed during denoising:")
        for f in failed_files:
            logger.warning(f"  {f}")

    logger.info(f"Denoised {len(success_files)}/{len(audio_files)} files successfully")
    return success_files

