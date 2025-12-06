"""
Bird Audio Dataset Subset Builder
=================================

This script generates nested training subsets (S10, S20, S40, S80) and an 
independent testing set from a folder of bird audio recordings. The subsets 
are defined based on a species CSV file and sampling rules.

Instructions:
- Set `PROJECT_DIR` to your project folder.
- Set `DATASET_DIR` to your full bird audio dataset folder.
- Ensure `SPECIES_FILE` points to a CSV listing species and subset memberships.
- Run the script to generate CSV files for all subsets and testing.
"""

import os
import random
import logging
from collections import Counter
import pandas as pd

# ==============================
# CONFIGURATION (replace with your paths)
# ==============================
PROJECT_DIR = r"/path/to/project"  # e.g., "/home/user/bird_project"
DATA_DIR = os.path.join(PROJECT_DIR, "data")
DATASET_DIR = r"/path/to/full_dataset"  # e.g., "/home/user/bird_project/full_dataset"
SPECIES_FILE = os.path.join(PROJECT_DIR, "misc", "bird_species_subsets.csv")

MAX_SAMPLES = 750
TEST_RATIO = 0.15  # ~15% of recordings per species for independent testing
SEED = 42

SUBSETS = [] # List of subset names

os.makedirs(DATA_DIR, exist_ok=True)
random.seed(SEED)

# ==============================
# LOGGING SETUP
# ==============================
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ==============================
# FUNCTIONS
# ==============================
def build_testing_and_nested_subsets():
    """Generate testing and nested subset CSVs from dataset."""
    if not os.path.isfile(SPECIES_FILE):
        raise FileNotFoundError(f"Species file not found: {SPECIES_FILE}")

    species_df = pd.read_csv(SPECIES_FILE)
    subset_rows = {subset: [] for subset in SUBSETS}
    testing_rows = []

    for _, row_info in species_df.iterrows():
        folder_name = row_info["Scientific Name"]
        folder_path = os.path.join(DATASET_DIR, folder_name)
        if not os.path.isdir(folder_path):
            logger.warning(f"Folder not found for species {folder_name}")
            continue

        # Accept common audio formats
        files = [f for f in os.listdir(folder_path) if f.lower().endswith((".mp3", ".wav", ".flac", ".ogg"))]
        files.sort()  # deterministic order

        if not files:
            logger.warning(f"No audio files found for {folder_name}")
            continue

        # --- Step 1: select testing files (independent) ---
        n_test = max(1, min(int(len(files) * TEST_RATIO), len(files) - 1))
        testing_files = random.sample(files, n_test)
        for f in testing_files:
            file_id = os.path.splitext(f)[0]
            testing_rows.append({
                "species": folder_name,
                "file_id": file_id,
                "file_path": os.path.join(folder_path, f)
            })

        # Remaining files for subsets
        remaining_files = [f for f in files if f not in testing_files]

        # --- Step 2: sample for nested subsets ---
        species_subsets = [s for s in SUBSETS if row_info.get(s) == 1]
        if not species_subsets:
            continue

        # Limit per species
        sampled_files = remaining_files.copy()
        if len(remaining_files) > MAX_SAMPLES:
            sampled_files = random.sample(remaining_files, MAX_SAMPLES)

        # Assign files to subsets
        for subset in species_subsets:
            for f in sampled_files:
                file_id = os.path.splitext(f)[0]
                subset_rows[subset].append({
                    "species": folder_name,
                    "file_id": file_id,
                    "file_path": os.path.join(folder_path, f)
                })

    # --- Save testing CSV ---
    testing_csv = os.path.join(DATA_DIR, "testing.csv")
    pd.DataFrame(testing_rows).to_csv(testing_csv, index=False)
    logger.info(f"✅ Independent testing CSV saved: {testing_csv}")
    for species, count in Counter([r["species"] for r in testing_rows]).items():
        logger.info(f"Testing: {species}: {count} files")

    # --- Save nested subset CSVs ---
    for subset, rows in subset_rows.items():
        subset_csv = os.path.join(DATA_DIR, f"{subset}.csv")
        pd.DataFrame(rows).to_csv(subset_csv, index=False)
        logger.info(f"✅ Nested subset CSV saved: {subset_csv}")
        for species, count in Counter([r["species"] for r in rows]).items():
            logger.info(f"{subset}: {species}: {count} files")

# ==============================
# MAIN EXECUTION
# ==============================
if __name__ == "__main__":
    build_testing_and_nested_subsets()
