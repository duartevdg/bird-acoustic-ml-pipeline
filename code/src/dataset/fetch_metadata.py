"""
Bird Audio Metadata Extractor
=============================

This script scans a folder of bird audio recordings, fetches metadata from
Xeno-Canto using the `xenopy` API (via direct HTTP requests), and extracts
local audio properties using pydub. The results are saved as a CSV for
further analysis.

Instructions:
- Replace `API_KEY` with your own Xeno-Canto API key.
- Set `AUDIO_DIR` to your dataset folder path.
- Set `CSV_OUTPUT_PATH` and `FAILED_JSON_PATH` to desired output paths.
"""

import os
import time
import random
import json
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
import pandas as pd
from pydub.utils import mediainfo
from requests.adapters import HTTPAdapter, Retry
from tqdm import tqdm

# ==============================
# USER CONFIGURATION
# ==============================
API_KEY = "API_KEY"  # <-- Replace with API key
DATASET_NAME = "DATASET_NAME"  # <-- Replace with dataset name
AUDIO_DIR = "DATASET_PATH"  # <-- Replace with audio dataset path
CSV_OUTPUT_PATH = rf"C:\Users\User\Desktop\Duarte\results\dataset_analysis\metadata_files\{DATASET_NAME}_metadata.csv"
FAILED_JSON_PATH = rf"C:\Users\User\Desktop\Duarte\results\dataset_analysis\metadata_files\{DATASET_NAME}_failed.json"

os.makedirs(os.path.dirname(CSV_OUTPUT_PATH), exist_ok=True)

# Fields to extract (XC metadata + audio info)
FIELDS = [
    "id", "gen", "sp", "ssp", "grp", "en", "rec", "cnt", "loc",
    "lat", "lon", "alt", "type", "sex", "stage", "method", "q",
    "length", "time", "date", "uploaded", "rmk", "animal-seen",
    "playback-used", "auto", "dvc", "mic", "smp", "file-name",
    "samplerate", "bitdepth", "channels", "duration", "format", "size_bytes"
]

# ==============================
# SETUP
# ==============================
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

session = requests.Session()
retries = Retry(total=10, backoff_factor=0.5, status_forcelist=[429, 500, 502, 503, 504])
session.mount('https://', HTTPAdapter(max_retries=retries))

# ==============================
# HELPER FUNCTIONS
# ==============================
def fetch_metadata_xc(recording_id: str, max_retries: int = 8) -> dict | None:
    """Fetch Xeno-Canto metadata for a given recording ID with retries."""
    base_url = 'https://xeno-canto.org/api/3/recordings'
    params = {'query': f'nr:{recording_id}', 'key': API_KEY}

    for attempt in range(max_retries + 1):
        try:
            time.sleep(random.uniform(0.05, 0.2))
            resp = session.get(base_url, params=params, timeout=10)
            resp.raise_for_status()
            data = resp.json()
            if 'recordings' in data and len(data['recordings']) > 0:
                return data['recordings'][0]
            return None
        except requests.exceptions.RequestException:
            wait = (2 ** attempt) + random.uniform(0, 1)
            time.sleep(wait)
    return None

def extract_audio_info(filepath: str) -> dict:
    """Extract basic audio information using pydub."""
    try:
        info = mediainfo(filepath)
        return {
            "samplerate": int(info.get("sample_rate", 0)),
            "bitdepth": int(info.get("bits_per_sample", 0)),
            "channels": int(info.get("channels", 0)),
            "duration": float(info.get("duration", 0)),
            "format": info.get("format_name", "unknown"),
            "size_bytes": os.path.getsize(filepath)
        }
    except Exception:
        return {k: 0 if k != "format" else "error" for k in ["samplerate","bitdepth","channels","duration","size_bytes","format"]}

def process_file(filepath: str) -> dict:
    """Process a single audio file: fetch metadata and extract local audio info."""
    file_name = os.path.basename(filepath)
    recording_id = os.path.splitext(file_name)[0].lstrip("XC").strip()

    metadata = fetch_metadata_xc(recording_id)
    audio_info = extract_audio_info(filepath)

    row = {field: metadata.get(field, "") if metadata else "" for field in FIELDS}
    row.update(audio_info)
    row["id"] = recording_id
    row["file-name"] = file_name

    return row

# ==============================
# MAIN FUNCTION
# ==============================
def main():
    # Collect audio files
    audio_files = [os.path.join(root, f)
                   for root, _, files in os.walk(AUDIO_DIR)
                   for f in files if f.lower().endswith(".mp3")]

    logger.info(f"Total MP3 files found: {len(audio_files)}")

    # Load previously failed files
    if os.path.exists(FAILED_JSON_PATH):
        with open(FAILED_JSON_PATH, "r") as f:
            failed_files = json.load(f)
        logger.info(f"Resuming with {len(failed_files)} previously failed files")
    else:
        failed_files = audio_files.copy()

    results = []

    # Retry rounds
    MAX_RETRY_ROUNDS = 5
    num_cores = os.cpu_count() or 4
    max_workers = min(num_cores * 2, 32)
    logger.info(f"Using {max_workers} workers based on CPU cores ({num_cores})")

    for round_idx in range(1, MAX_RETRY_ROUNDS + 1):
        if not failed_files:
            break

        logger.info(f"=== Round {round_idx}: {len(failed_files)} files to process ===")
        round_results, round_failed = [], []

        batch_size = 50
        for i in range(0, len(failed_files), batch_size):
            batch_files = failed_files[i:i+batch_size]
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = {executor.submit(process_file, f): f for f in batch_files}
                for future in tqdm(as_completed(futures), total=len(futures),
                                   desc=f"Round {round_idx} batch {i//batch_size + 1}"):
                    try:
                        res = future.result()
                        if res is None:
                            round_failed.append(futures[future])
                        else:
                            round_results.append(res)
                    except Exception as e:
                        logger.warning(f"Error processing {futures[future]}: {e}")
                        round_failed.append(futures[future])

        results.extend(round_results)
        failed_files = round_failed

        # Save intermediate CSV
        pd.DataFrame(results).to_csv(CSV_OUTPUT_PATH, index=False)
        logger.info(f"Intermediate CSV saved with {len(results)} entries")

        # Save failed files
        with open(FAILED_JSON_PATH, "w") as f:
            json.dump(failed_files, f)

        if not failed_files:
            logger.info("All files processed successfully!")
            break

        wait_time = 10 * round_idx
        logger.info(f"Waiting {wait_time}s before next retry round...")
        time.sleep(wait_time)

    logger.info(f"Metadata extraction finished. CSV saved to {CSV_OUTPUT_PATH}")
    if failed_files:
        logger.warning(f"{len(failed_files)} files could not be processed. See {FAILED_JSON_PATH}")

if __name__ == "__main__":
    main()
