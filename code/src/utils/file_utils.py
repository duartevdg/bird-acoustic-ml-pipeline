"""
Audio File Utility Functions
============================

Provides helper functions to load YAML configs, search for audio files recursively,
and temporarily flatten WAV files while preserving directory structure.

Usage:
------
- Use `load_all_configs` to load pipeline paths, main config, and hyperparameters.
- Use `find_audio_files_recursive` or `find_wav_files_recursive` to list audio files.
- Use `flatten_wav_files_with_structure` to copy WAV files into a temporary folder while
  keeping relative paths and keeping a mapping of original to flattened paths.
"""

import os
import yaml
from string import Template
import shutil
import tempfile
from typing import List, Tuple, Dict, Optional
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# -------------------------
# CONFIG LOADING FUNCTIONS
# -------------------------

def load_config(config_path: str = 'config.yaml') -> dict:
    """
    Load a YAML configuration file.
    
    Parameters
    ----------
    config_path : str
        Path to the YAML file.
    
    Returns
    -------
    dict
        Parsed YAML content.
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def load_all_configs(paths_file: str = "config/paths.yaml", 
                     main_config_file: str = "config/main.yaml", 
                     hp_file: str = "config/hyperparameters.yaml") -> Tuple[dict, dict, dict]:
    """
    Load and resolve all pipeline configuration files.
    
    Resolves paths in `paths_file` that reference `base_path` using string.Template.
    
    Parameters
    ----------
    paths_file : str
        YAML file containing folder paths.
    main_config_file : str
        Main experiment configuration YAML.
    hp_file : str
        Hyperparameters YAML file.
    
    Returns
    -------
    Tuple[dict, dict, dict]
        resolved_paths, pipeline_config, hyperparameters
    """
    # Load paths and resolve placeholders
    raw_paths = load_config(paths_file)
    base_path = raw_paths.get("base_path", "")
    resolved_paths = {
        k: Template(v).substitute(base_path=base_path) if isinstance(v, str) and "${base_path}" in v else v
        for k, v in raw_paths.items()
    }
    logging.info(f"[load_all_configs] Loaded and resolved paths from {paths_file}")

    # Load main pipeline config and hyperparameters
    pipeline_config = load_config(main_config_file)
    hyperparameters = load_config(hp_file)
    logging.info(f"[load_all_configs] Loaded pipeline config from {main_config_file} and hyperparameters from {hp_file}")

    return resolved_paths, pipeline_config, hyperparameters

# -------------------------
# AUDIO FILE SEARCH FUNCTIONS
# -------------------------

def find_audio_files_recursive(folder: str, extensions: Optional[List[str]] = None) -> List[str]:
    """
    Recursively find audio files with specified extensions.
    
    Parameters
    ----------
    folder : str
        Root folder to search.
    extensions : list of str, optional
        List of file extensions to include (default: common audio formats).
    
    Returns
    -------
    List[str]
        Full paths to audio files found.
    """
    if extensions is None:
        extensions = ['.wav', '.mp3', '.flac', '.ogg', '.m4a', '.aac']

    audio_files = [
        os.path.join(root, f)
        for root, _, files in os.walk(folder)
        for f in files
        if any(f.lower().endswith(ext) for ext in extensions)
    ]
    logging.info(f"[find_audio_files_recursive] Found {len(audio_files)} audio files in {folder}")
    return audio_files


def find_wav_files_recursive(folder: str) -> List[str]:
    """
    Recursively find all WAV files in a folder.
    
    Parameters
    ----------
    folder : str
        Root folder to search.
    
    Returns
    -------
    List[str]
        Full paths to WAV files found.
    """
    wav_files = find_audio_files_recursive(folder, extensions=['.wav'])
    logging.info(f"[find_wav_files_recursive] Found {len(wav_files)} WAV files in {folder}")
    return wav_files

# -------------------------
# TEMPORARY WAV FLATTENING
# -------------------------

def flatten_wav_files_with_structure(wav_files: List[str], base_folder: str) -> Tuple[str, Dict[str, str]]:
    """
    Copy WAV files into a temporary folder while preserving relative directory structure.
    
    Parameters
    ----------
    wav_files : List[str]
        List of WAV file paths.
    base_folder : str
        Base folder to compute relative paths from.
    
    Returns
    -------
    Tuple[str, dict]
        temp_flat_folder : str
            Path to temporary folder containing copied WAVs.
        path_map : dict
            Mapping from flattened paths to original paths.
    """
    temp_flat_folder = tempfile.mkdtemp(prefix="wav_flatten_")
    path_map: Dict[str, str] = {}

    for wav_path in wav_files:
        rel_path = os.path.relpath(wav_path, base_folder)
        dest_path = os.path.join(temp_flat_folder, rel_path)
        os.makedirs(os.path.dirname(dest_path), exist_ok=True)
        shutil.copy2(wav_path, dest_path)
        path_map[os.path.normpath(dest_path)] = os.path.normpath(wav_path)

    logging.info(f"[flatten_wav_files_with_structure] Flattened {len(wav_files)} WAV files to {temp_flat_folder}")
    return temp_flat_folder, path_map
