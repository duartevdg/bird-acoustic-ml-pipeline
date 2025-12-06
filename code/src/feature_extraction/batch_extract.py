"""
Feature Extraction Module

Provides utilities to extract features from audio files stored in a pandas DataFrame.
Supports parallel extraction with ProcessPoolExecutor and saves results as CSV.
"""

import os
import csv
import logging
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
import pandas as pd

logger = logging.getLogger(__name__)

# ----------------------------
# Safe feature extraction
# ----------------------------
def _extract_features_safe(feature_extractor, file_path, class_label, group):
    """
    Safely extracts features from a single audio file, returning a dictionary
    with metadata and features. Errors are logged without interrupting execution.
    """
    try:
        features = feature_extractor(file_path)
        file_id = os.path.splitext(os.path.basename(file_path))[0]
        features['file_id'] = file_id
        features['file_path'] = file_path
        features['class'] = class_label
        features['group'] = group
        return features
    except Exception as e:
        logger.error(f"Error extracting features from {file_path}: {e}")
        return None

# ----------------------------
# Folder-based extraction
# ----------------------------
def extract_features_from_folder(metadata_df: pd.DataFrame, feature_extractor, output_csv_path: str,
                                 max_workers=None, extractor_name=None):
    """
    Extracts features from a list of audio files provided in a DataFrame.

    Args:
        metadata_df (pd.DataFrame): Must contain 'file_path', 'species', 'group'.
        feature_extractor (callable): Function that takes file_path and returns a dict of features.
        output_csv_path (str): Path to save the resulting CSV.
        max_workers (int, optional): Number of parallel workers. Defaults to os.cpu_count().
        extractor_name (str, optional): Name used in progress description.

    Returns:
        None
    """
    audio_files = metadata_df['file_path'].tolist()
    file_class_map = dict(zip(metadata_df['file_path'], metadata_df['species']))
    file_group_map = dict(zip(metadata_df['file_path'], metadata_df['group']))

    if not audio_files:
        logger.error("No audio files found in DataFrame.")
        return

    logger.info(f"Starting feature extraction for {len(audio_files)} audio files "
                f"with up to {max_workers or os.cpu_count()} workers.")

    missing_group = [f for f in audio_files if file_group_map.get(f) is None]
    if missing_group:
        logger.warning(f"{len(missing_group)} files missing group info. Example: {missing_group[:5]}")
    else:
        logger.info("All files matched successfully to group values.")

    all_features = []
    desc = f"Extracting features ({extractor_name or feature_extractor.__name__})"
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(_extract_features_safe,
                            feature_extractor,
                            file,
                            file_class_map.get(file),
                            file_group_map.get(file))
            : file for file in audio_files
        }

        for future in tqdm(as_completed(futures), total=len(audio_files), desc=desc):
            file_path = futures[future]
            features = future.result()
            if features is not None:
                all_features.append(features)
            else:
                logger.warning(f"Failed to extract features from {file_path}.")

    if not all_features:
        logger.error("No features were successfully extracted.")
        return

    # Consolidate keys and maintain order
    keys = set()
    for feat in all_features:
        keys.update(feat.keys())
    for k in ['file_id', 'file_path', 'class', 'group']:
        keys.discard(k)
    sorted_keys = ['file_id', 'file_path', 'class', 'group'] + sorted(keys)

    # Save to CSV
    os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)
    with open(output_csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=sorted_keys)
        writer.writeheader()
        for feat in all_features:
            writer.writerow(feat)

    logger.info(f"Features saved to {output_csv_path}.")

# ----------------------------
# Multiple extractor support
# ----------------------------
def extract_multiple(metadata_df: pd.DataFrame, output_folder: str, extractors: list[tuple], n_workers=2):
    """
    Runs multiple feature extractors sequentially and saves each to CSV.

    Args:
        metadata_df (pd.DataFrame): Must contain 'file_path', 'species', 'group'.
        output_folder (str): Folder where CSVs will be saved.
        extractors (list[tuple]): List of (feature_extractor_function, output_csv_name)
        n_workers (int, optional): Number of parallel workers for each extractor. Defaults to 2.
    """
    os.makedirs(output_folder, exist_ok=True)

    for func, name in extractors:
        out_csv = os.path.join(output_folder, name)
        extract_features_from_folder(
            metadata_df,
            func,
            out_csv,
            max_workers=n_workers,
            extractor_name=name
        )
