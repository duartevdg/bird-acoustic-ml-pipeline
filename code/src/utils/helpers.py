"""
Bird Audio Pipeline Utilities
=============================

This module contains helper functions for merging feature CSVs, filtering preprocessing
combinations, logging, cleaning temporary files, saving results, and notifications.

Functions:
- merge_feature_csvs: Merge multiple feature CSV files into a single DataFrame.
- passes_filters: Check if a preprocessing combo meets selected run options.
- setup_file_logging: Configure a logger to write logs to a file.
- safe_rmtree: Delete a folder with retries to handle temporary file locks.
- clean_temp_except_splits: Clean temporary folder while preserving train/test splits.
- save_results: Save or append experiment results while avoiding duplicates.
- append_result_row: Append a single row to a CSV with optional deduplication.
- notify: Send a simple notification (e.g., via ntfy.sh).
- combo_code: Generate a short code summarizing preprocessing steps.
"""

import os
import time
import shutil
import logging
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Any
import pandas as pd
import requests


# ======================
# CSV MERGING
# ======================
def merge_feature_csvs(feature_folder: str, output_csv: str, logger: Optional[logging.Logger] = None) -> Optional[pd.DataFrame]:
    """
    Merge all CSV feature files in a folder into a single DataFrame.
    
    Parameters
    ----------
    feature_folder : str
        Path containing feature CSV files.
    output_csv : str
        Output path for the merged CSV.
    logger : logging.Logger, optional
        Logger for info/warning messages.
    
    Returns
    -------
    pd.DataFrame or None
        Merged DataFrame if files exist, otherwise None.
    """
    feature_folder = Path(feature_folder)
    output_csv = Path(output_csv)
    csv_files = list(feature_folder.glob("*.csv"))
    if not csv_files:
        if logger: logger.warning(f"No CSV files found in {feature_folder}, skipping merge")
        return None

    if logger: logger.info(f"Found {len(csv_files)} CSVs in {feature_folder}")

    merged = None
    for f in csv_files:
        df = pd.read_csv(f)

        metadata_cols = [c for c in ["file_id", "file_path", "group", "species", "class"] if c in df.columns]
        feature_cols = [c for c in df.columns if c not in metadata_cols]

        if logger:
            logger.info(f"Reading {f.name}: metadata cols={len(metadata_cols)}, feature cols={len(feature_cols)}")

        if "file_id" not in df.columns:
            if logger: logger.warning(f"Skipping {f.name}: missing 'file_id'")
            continue

        if merged is None:
            merged = df
            if logger: logger.info(f"Initialized merged DataFrame from {f.name}, shape {merged.shape}")
        else:
            drop_cols = [c for c in metadata_cols if c in df.columns and c in merged.columns and c != "file_id"]
            df = df.drop(columns=drop_cols, errors="ignore")
            merged = pd.merge(merged, df, on="file_id", how="outer")
            if logger: logger.info(f"Merged {f.name}, merged shape {merged.shape}")

    if merged is not None and not merged.empty:
        merged = merged.fillna(0)
        merged.to_csv(output_csv, index=False)
        if logger: logger.info(f"Saved merged CSV to {output_csv} with shape {merged.shape}")
    else:
        if logger: logger.warning("Merged DataFrame is empty, CSV not created")

    return merged


# ======================
# FILTERING COMBOS
# ======================
def passes_filters(combo: Tuple[Any, ...], run_options: Dict[str, List[Any]]) -> bool:
    """
    Check if a preprocessing combination passes the selected run options.
    
    Parameters
    ----------
    combo : tuple
        Tuple of parameters: (format, samplerate, bitdepth, channels, amplitude, pre_emphasis, denoise, segment, augment)
    run_options : dict
        Dictionary of allowed values for each parameter.
    
    Returns
    -------
    bool
        True if the combo is allowed, False otherwise.
    """
    fmt, sr, bd, ch, amp, preemph, denoise, segment, augment = combo

    def filter_numeric(param_value, allowed_list, cast_type):
        if allowed_list:
            if str(param_value).lower() == "none":
                return "none" in [str(v).lower() for v in allowed_list]
            try:
                param_casted = cast_type(param_value)
                allowed_casted = [cast_type(v) for v in allowed_list if str(v).lower() != "none"]
                return param_casted in allowed_casted
            except (ValueError, TypeError):
                return False
        return True

    def filter_string(param_value, allowed_list):
        if allowed_list:
            if str(param_value).lower() == "none":
                return "none" in [str(v).lower() for v in allowed_list]
            return param_value in allowed_list
        return True

    return (
        filter_string(fmt, run_options.get("formats")) and
        filter_numeric(sr, run_options.get("sampling_rates"), int) and
        filter_numeric(bd, run_options.get("bit_depths"), int) and
        filter_string(ch, run_options.get("channels")) and
        filter_string(amp, run_options.get("amplitude_methods")) and
        filter_numeric(preemph, run_options.get("pre_emphasis_coefficients"), float) and
        filter_string(denoise, run_options.get("denoising_methods")) and
        filter_string(segment, run_options.get("segmentation_methods")) and
        filter_string(augment, run_options.get("augmentation_methods"))
    )


# ======================
# LOGGING
# ======================
def setup_file_logging(logger: logging.Logger, logs_path: str, name: str = "pipeline") -> None:
    """
    Configure file-based logging.
    """
    os.makedirs(logs_path, exist_ok=True)
    log_file = os.path.join(logs_path, f"{name}.log")
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s — %(levelname)s — %(message)s')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.propagate = False


# ======================
# TEMP FILE CLEANUP
# ======================
def safe_rmtree(folder: str, logger: Optional[logging.Logger] = None, max_retries: int = 5, delay: float = 2) -> None:
    """
    Safely delete a folder with retry attempts to handle temporary file locks.
    """
    for attempt in range(max_retries):
        try:
            shutil.rmtree(folder)
            return
        except Exception as e:
            msg = f"Failed to delete {folder} on attempt {attempt+1}: {e}"
            if logger: logger.warning(msg)
            else: print(msg)
            time.sleep(delay)
    if logger:
        logger.error(f"Could not remove {folder} after {max_retries} attempts")
    else:
        print(f"Could not remove {folder} after {max_retries} attempts")


def clean_temp_except_splits(temp_path: str, subset: str, logger: logging.Logger) -> None:
    """
    Clean a temp folder but preserve split folders like "{subset}_train" and "{subset}_test".
    """
    keep_folders = {f"{subset}_train", f"{subset}_test"}
    for entry in os.listdir(temp_path):
        full_path = os.path.join(temp_path, entry)
        if entry in keep_folders:
            logger.info(f"Keeping split folder: {entry}")
            continue
        if os.path.isdir(full_path):
            safe_rmtree(full_path, logger)
            logger.info(f"Deleted folder: {entry}")
        else:
            try:
                os.remove(full_path)
                logger.info(f"Deleted file: {entry}")
            except Exception as e:
                logger.error(f"Failed to delete {entry}: {e}")


# ======================
# RESULT HANDLING
# ======================
def save_results(all_results: List[Dict], results_path: str, key_columns: List[str]) -> int:
    """
    Save experiment results to a CSV, avoiding duplicates based on key columns.
    """
    results_path = Path(results_path)
    results_path.mkdir(parents=True, exist_ok=True)
    results_csv = results_path / "final_results.csv"

    all_results_df = pd.DataFrame(all_results)
    if results_csv.exists() and results_csv.stat().st_size > 0:
        existing_df = pd.read_csv(results_csv)
        combined_df = pd.concat([existing_df, all_results_df], ignore_index=True)
        combined_df.drop_duplicates(subset=key_columns, keep="first", inplace=True)
    else:
        combined_df = all_results_df
        existing_df = pd.DataFrame()

    combined_df.sort_values(by=key_columns, ignore_index=True, inplace=True)
    combined_df.to_csv(results_csv, index=False)
    return len(combined_df) - len(existing_df) if not existing_df.empty else len(all_results_df)


def append_result_row(row: Dict, path: str, key_columns: List[str], deduplicate: bool = True) -> None:
    """
    Append a single result row to a CSV file with optional deduplication.
    """
    df_row = pd.DataFrame([row])
    path = Path(path)
    if deduplicate and path.exists() and path.stat().st_size > 0:
        existing_df = pd.read_csv(path)
        if tuple(row[k] for k in key_columns) in set(map(tuple, existing_df[key_columns].to_numpy())):
            return
    df_row.to_csv(path, mode='a' if path.exists() else 'w', header=not path.exists(), index=False)


# ======================
# NOTIFICATIONS
# ======================
def notify(message: str = "Job finished!", endpoint_url: str = "") -> None:
    """
    Send a notification via a user-specified endpoint (e.g., ntfy.sh).

    Parameters
    ----------
    message : str
        Message to send.
    endpoint_url : str
        Full URL of the notification endpoint. Example: "https://ntfy.sh/your_topic"
        If empty, the function will not send anything.

    Notes
    -----
    Users must provide their own endpoint URL. This function will fail silently
    if the request cannot be sent.
    """
    if not endpoint_url:
        print(f"[notify] No endpoint URL provided. Message not sent: {message}")
        return

    try:
        response = requests.post(endpoint_url, data=message.encode())
        if response.status_code != 200:
            print(f"[notify] Failed to send notification, status code: {response.status_code}")
    except Exception as e:
        print(f"[notify] Notification failed: {e}")


# ======================
# UTILITY
# ======================
def combo_code(combo_dict: Dict[str, Any]) -> str:
    """
    Generate a compact code summarizing key preprocessing steps.
    
    Example: {'pre_emphasis':'0.95','denoising':'none','segmentation':'time','augmentation':'pitch_shift'}
    -> 'PE0.95-Dnone-Stime-Apitch_shift'
    """
    return (
        f"PE{combo_dict['pre_emphasis']}-D{combo_dict['denoising']}-"
        f"S{combo_dict['segmentation']}-A{combo_dict['augmentation']}"
    )
