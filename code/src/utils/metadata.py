"""
Bird Audio Metadata Loader
==========================

This script scans a folder containing bird audio recordings, extracts audio properties,
and fetches metadata from the Xeno-Canto API. The resulting DataFrame contains both
audio features and recording metadata, suitable for downstream analysis or machine
learning pipelines.

Usage:
- Set your Xeno-Canto API key below.
- Call `load_metadata_for_folder(folder_path)` to get a DataFrame with metadata.
"""

import os
import time
import random
import logging
from typing import Tuple, Union, List
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
import pandas as pd
from pydub.utils import mediainfo
from requests.adapters import HTTPAdapter, Retry

# ======================
# CONFIGURATION
# ======================

# Supported audio file extensions
AUDIO_EXTS: Tuple[str, ...] = (".mp3", ".wav", ".flac", ".ogg")

# API endpoint
API_URL: str = "https://xeno-canto.org/api/3/recordings"

# Required fields from Xeno-Canto metadata
API_FIELDS: List[str] = [
    "id", "gen", "sp", "ssp", "en", "rec", "cnt", "loc",
    "lat", "lon", "alt", "type", "sex", "q", "time",
    "date", "uploaded", "file-name"
]

# Audio properties to extract
AUDIO_FIELDS: List[str] = ["samplerate", "bitdepth", "channels", "duration", "format", "size_bytes"]

# Load API key from environment variable for security
API_KEY: str = "YOUR_API_KEY_HERE"  # <-- replace "YOUR_API_KEY_HERE" with your actual key

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# ======================
# FUNCTIONS
# ======================

def _make_session(total_retries: int = 3, backoff: float = 0.5) -> requests.Session:
    """
    Create a requests session with retry logic.
    
    Parameters
    ----------
    total_retries : int
        Number of retry attempts for failed requests.
    backoff : float
        Factor for exponential backoff between retries.
    
    Returns
    -------
    requests.Session
        Configured HTTP session.
    """
    session = requests.Session()
    retry_strategy = Retry(
        total=total_retries,
        backoff_factor=backoff,
        status_forcelist=[429, 500, 502, 503, 504]
    )
    session.mount("https://", HTTPAdapter(max_retries=retry_strategy))
    return session


def _fetch_metadata_xc(recording_id: str, session: requests.Session, api_key: str, max_retries: int = 10) -> Union[dict, None]:
    """
    Fetch metadata from Xeno-Canto API for a given recording ID.
    
    Parameters
    ----------
    recording_id : str
        Xeno-Canto recording identifier.
    session : requests.Session
        HTTP session with retry logic.
    api_key : str
        API key for authentication.
    max_retries : int
        Maximum retry attempts.
    
    Returns
    -------
    dict or None
        Metadata dictionary if successful, None otherwise.
    """
    params = {"query": f"nr:{recording_id}", "key": api_key}
    backoff_base = 5

    for attempt in range(max_retries + 1):
        try:
            time.sleep(random.uniform(0.05, 0.2))  # small random delay
            response = session.get(API_URL, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            if data.get("recordings"):
                return data["recordings"][0]
            return None
        except requests.exceptions.HTTPError as e:
            if response.status_code in (429, 503):
                sleep_time = backoff_base * 2**attempt + random.uniform(0, 0.5)
                logging.warning(f"HTTP {response.status_code}, retrying in {sleep_time:.1f}s...")
                time.sleep(sleep_time)
            else:
                logging.error(f"HTTP error: {e}")
                break
        except requests.exceptions.RequestException as e:
            logging.error(f"Request exception: {e}")
            break
    return None


def _audio_props(filepath: str) -> dict:
    """
    Extract audio properties using pydub.
    
    Parameters
    ----------
    filepath : str
        Path to audio file.
    
    Returns
    -------
    dict
        Dictionary with audio properties.
    """
    try:
        info = mediainfo(filepath)
        return {
            "samplerate": int(info.get("sample_rate", 0)),
            "bitdepth": int(info.get("bits_per_sample", 0)),
            "channels": int(info.get("channels", 0)),
            "duration": float(info.get("duration", 0)),
            "format": info.get("format_name", "unknown"),
            "size_bytes": os.path.getsize(filepath),
        }
    except Exception as e:
        logging.warning(f"Failed to read audio info for {filepath}: {e}")
        return {k: 0 for k in AUDIO_FIELDS}


def _process_file(filepath: str, session: requests.Session, api_key: str) -> Union[dict, None]:
    """
    Process a single audio file: fetch Xeno-Canto metadata and audio properties.
    
    Parameters
    ----------
    filepath : str
        Path to audio file.
    session : requests.Session
        HTTP session.
    api_key : str
        Xeno-Canto API key.
    
    Returns
    -------
    dict or None
        Combined metadata and audio properties.
    """
    # Extract recording ID from filename (assumes filenames like XC12345.wav)
    rec_id = os.path.splitext(os.path.basename(filepath))[0].lstrip("XC")
    xc_meta = _fetch_metadata_xc(rec_id, session, api_key)
    if not xc_meta:
        return None

    data = {field: xc_meta.get(field, "") for field in API_FIELDS}
    data.update(_audio_props(filepath))
    data.update({
        "file_path": filepath,
        "class": os.path.basename(os.path.dirname(filepath)),
        "author": xc_meta.get("rec", "unknown")
    })
    return data


def load_metadata_for_folder(
    folder_path: str,
    extensions: Tuple[str, ...] = AUDIO_EXTS,
    n_workers: int = None,
    max_retry_rounds: int = 3,
    group_by: Union[str, List[str], None] = None
) -> pd.DataFrame:
    """
    Load metadata for all audio files in a folder.
    
    Parameters
    ----------
    folder_path : str
        Path to folder containing audio files.
    extensions : tuple of str
        Supported audio file extensions.
    n_workers : int, optional
        Number of parallel workers (defaults to CPU cores * 2, max 32).
    max_retry_rounds : int
        Number of retry rounds for failed files.
    group_by : str or list of str, optional
        Columns to group by (e.g., 'author').
    
    Returns
    -------
    pd.DataFrame
        DataFrame containing metadata for all audio files.
    """
    # Find all audio files
    files = [os.path.join(root, f) for root, _, fs in os.walk(folder_path) 
             for f in fs if f.lower().endswith(extensions)]
    if not files:
        raise FileNotFoundError(f"No audio files found in {folder_path}")

    session = _make_session()
    workers = n_workers or min((os.cpu_count() or 4) * 2, 32)
    results, pending = [], files.copy()

    for round_num in range(1, max_retry_rounds + 1):
        failed, round_results = [], []
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(_process_file, f, session, API_KEY): f for f in pending}
            for future in as_completed(futures):
                res = future.result()
                if res:
                    round_results.append(res)
                else:
                    failed.append(futures[future])

        results.extend(round_results)
        pending = failed
        if not pending:
            break
        logging.info(f"Retrying {len(pending)} failed files, round {round_num}...")
        time.sleep(round_num * 7)  # incremental wait

    if pending:
        logging.warning(f"{len(pending)} files could not fetch metadata.")

    df = pd.DataFrame(results)
    df['filename'] = df['file_path'].apply(lambda x: os.path.splitext(os.path.basename(x))[0])
    cols = ["file_path", "filename", "class", "author"] + list(API_FIELDS) + list(AUDIO_FIELDS)
    df = df[[c for c in cols if c in df.columns]]

    if group_by:
        group_cols = [group_by] if isinstance(group_by, str) else group_by
        missing_cols = [c for c in group_cols if c not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing grouping columns: {missing_cols}")
        df['group'] = df[group_cols].astype(str).agg('_'.join, axis=1)

    return df
