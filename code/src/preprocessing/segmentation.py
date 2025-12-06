"""
Audio Segmentation Module
=========================

Supports:
- Time-based segmentation
- Event-based (silence) segmentation
- Adaptive parameter optimization for silence removal
- Dataset-level parallel processing
- WAV output of individual segments
"""

import os
from pathlib import Path
from joblib import Parallel, delayed
import logging
import soundfile as sf

from pyAudioAnalysis import audioBasicIO as aIO
from pyAudioAnalysis import audioSegmentation as aS

try:
    from tqdm import tqdm
    USE_TQDM = True
except ImportError:
    USE_TQDM = False

# ==============================
# LOGGING SETUP
# ==============================
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ==============================
# UTILITY FUNCTIONS
# ==============================
def load_audio(filepath):
    Fs, signal = aIO.read_audio_file(filepath)
    return Fs, signal

def save_segments_to_wav(signal, Fs, segments, output_folder, prefix="segment"):
    """Save each segment as a separate WAV file."""
    os.makedirs(output_folder, exist_ok=True)
    for i, (start_sec, end_sec) in enumerate(segments):
        start_sample = int(start_sec * Fs)
        end_sample = int(end_sec * Fs)
        segment_signal = signal[start_sample:end_sample]
        filename = os.path.join(output_folder, f"{prefix}_{i+1:03d}.wav")
        sf.write(filename, segment_signal, Fs)

def merge_close_segments(segments, merge_threshold=0.5):
    """Merge segments closer than merge_threshold (seconds)."""
    if not segments:
        return []
    segments = sorted(segments, key=lambda x: x[0])
    merged = [list(segments[0])]
    for start, end in segments[1:]:
        if start - merged[-1][1] <= merge_threshold:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return [tuple(seg) for seg in merged]

# ==============================
# TIME-BASED SEGMENTATION
# ==============================
def time_based_segmentation(signal, Fs, segment_length_sec):
    total_duration_sec = len(signal) / Fs
    segments = []
    start = 0.0
    while start < total_duration_sec:
        end = min(start + segment_length_sec, total_duration_sec)
        segments.append((start, end))
        start += segment_length_sec
    return segments

# ==============================
# EVENT-BASED SEGMENTATION
# ==============================
def simple_silence_removal(signal, Fs, short_window=0.02, step=0.02,
                           smooth_window=1.0, weight=0.5, merge_threshold=0.5,
                           min_duration=1.0, plot=False):
    segments = aS.silence_removal(signal, Fs, short_window, step,
                                  smooth_window=smooth_window,
                                  weight=weight, plot=plot)
    segments = [s for s in segments if s[1]-s[0] >= min_duration]
    segments = merge_close_segments(segments, merge_threshold)
    return segments if segments else [(0.0, len(signal)/Fs)]

def adaptive_silence_removal(signal, Fs, short_window=0.02, step=0.02,
                              smooth_window_values=[0.5,1.0,2.0],
                              weight_values=[0.3,0.5,0.7],
                              merge_threshold=0.5, min_duration=1.0,
                              plot=False):
    """Grid-search best smooth_window and weight for silence removal."""
    best, _ = None, float('inf')
    for s in smooth_window_values:
        for w in weight_values:
            segs = aS.silence_removal(signal, Fs, short_window, step,
                                      smooth_window=s, weight=w, plot=False)
            segs = [seg for seg in segs if seg[1]-seg[0]>=min_duration]
            merged = merge_close_segments(segs, merge_threshold)
            if len(merged) < _:
                best, _ = (s, w, merged), len(merged)
    if best is None:
        return [(0.0, len(signal)/Fs)]
    s, w, _ = best
    segments = aS.silence_removal(signal, Fs, short_window, step,
                                  smooth_window=s, weight=w, plot=plot)
    segments = [seg for seg in segments if seg[1]-seg[0] >= min_duration]
    return merge_close_segments(segments, merge_threshold)

# ==============================
# HIGH-LEVEL FUNCTIONS
# ==============================
def segment_file_safe(audio_path, output_dir, method="time", segment_length_sec=5.0,
                      smooth_window=1.0, weight=0.3, plot=False, min_duration=1.0):
    try:
        Fs, signal = load_audio(audio_path)
        base_filename = os.path.splitext(os.path.basename(audio_path))[0]

        if method == "time":
            segments = time_based_segmentation(signal, Fs, segment_length_sec)
        elif method == "event":
            segments = simple_silence_removal(signal, Fs, short_window=0.020, step=0.020, min_duration=min_duration)
        else:
            raise ValueError("Method must be 'time' or 'event'")

        os.makedirs(output_dir, exist_ok=True)
        save_segments_to_wav(signal, Fs, segments, output_dir, prefix=base_filename)
        return audio_path, "success"
    except Exception as e:
        logger.warning(f"[SKIP] Failed to segment {audio_path}: {e}")
        return audio_path, "failed"

def segment_dataset(input_folder, output_dir="segments", method="time",
                    segment_length_sec=5.0, smooth_window=1.0, weight=0.3,
                    plot=False, min_duration=1.0, n_jobs=-1):
    """Segment all audio files in a dataset, saving each segment as a WAV."""
    audio_files = [f for f in Path(input_folder).rglob("*.*")
                   if f.suffix.lower() in [".wav", ".mp3", ".flac", ".ogg", ".aiff"]]

    if not audio_files:
        logger.warning(f"No audio files found in {input_folder}")
        return []

    results = Parallel(n_jobs=n_jobs, prefer="processes")(
        delayed(segment_file_safe)(
            str(f),
            output_dir=os.path.join(output_dir, f.parent.name),
            method=method,
            segment_length_sec=segment_length_sec,
            smooth_window=smooth_window,
            weight=weight,
            plot=plot,
            min_duration=min_duration
        ) for f in audio_files
    )

    success_files = [f for f, status in results if status == "success"]
    failed_files = [f for f, status in results if status != "success" and f is not None]

    if failed_files:
        logger.warning(f"{len(failed_files)} files failed segmentation:")
        for f in failed_files:
            logger.warning(f"  {f}")

    logger.info(f"Segmented {len(success_files)}/{len(audio_files)} files successfully")
    return success_files