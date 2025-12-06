"""
Bird Audio Dataset Group Mapper
===============================

This script processes a cleaned bird audio metadata CSV, generates numeric
groups for recordings based on species, recorder, and location, and updates
subset CSV files with the corresponding group information.

Usage:
- Set `METADATA_FILE` to your cleaned metadata CSV.
- Set `DATA_DIR` to the folder containing your subset CSV files.
- Run the script to automatically update all subset CSVs with numeric group IDs.
"""

import os
import pandas as pd
import logging

# ==============================
# CONFIGURATION (replace with your paths)
# ==============================
DATA_DIR = r"/path/to/subset_csvs"  # e.g., "/home/user/bird_project/data"
METADATA_FILE = r"/path/to/cleaned_metadata.csv"  # e.g., "/home/user/bird_project/metadata/cleaned_metadata.csv"

# Subset CSV files to update
SUBSET_FILES = ["S10.csv", "S20.csv", "S40.csv", "S80.csv", "testing.csv"]

# ==============================
# LOGGING SETUP
# ==============================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

# ==============================
# HELPER FUNCTIONS
# ==============================
def load_and_prepare_metadata(metadata_path: str) -> pd.DataFrame:
    """
    Load the cleaned metadata and create string-based and numeric group IDs.

    Parameters
    ----------
    metadata_path : str
        Path to the cleaned metadata CSV.

    Returns
    -------
    pd.DataFrame
        Metadata DataFrame with 'group_str' and numeric 'group' columns.
    """
    if not os.path.isfile(metadata_path):
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

    metadata = pd.read_csv(metadata_path)

    # Fill missing recorder/location info and standardize strings
    metadata['rec'] = metadata['rec'].fillna('unknown_rec').str.strip().str.lower()
    metadata['loc'] = metadata['loc'].fillna('unknown_loc').str.strip().str.lower()

    # Create string-based group identifier
    metadata['group_str'] = (
        metadata['full_name'].str.replace(" ", "_").str.lower() +
        "_" + metadata['rec'] + "_" + metadata['loc']
    )

    # Map string groups to numeric IDs
    metadata['group'] = pd.factorize(metadata['group_str'])[0]

    logger.info(f"Loaded metadata with {len(metadata)} entries")
    logger.info(f"Total unique groups: {metadata['group'].nunique()}")

    return metadata

def update_subset_csvs(data_dir: str, metadata_df: pd.DataFrame, subset_files: list[str]) -> None:
    """
    Update subset CSV files with numeric group information from metadata.

    Parameters
    ----------
    data_dir : str
        Directory containing the subset CSV files.
    metadata_df : pd.DataFrame
        Metadata DataFrame with 'id' and 'group' columns.
    subset_files : list[str]
        List of subset CSV filenames to update.
    """
    metadata_df['id'] = metadata_df['id'].astype(str)
    metadata_df['group'] = metadata_df['group'].astype(str)

    for subset_file in subset_files:
        subset_path = os.path.join(data_dir, subset_file)
        if not os.path.isfile(subset_path):
            logger.warning(f"File not found: {subset_file}, skipping.")
            continue

        subset_df = pd.read_csv(subset_path)
        subset_df['file_id'] = subset_df['file_id'].astype(str)

        # Merge with metadata to assign groups
        merged_df = subset_df.merge(
            metadata_df[['id', 'group']],
            left_on='file_id',
            right_on='id',
            how='left'
        )

        # Keep only relevant columns
        final_df = merged_df[['species', 'file_id', 'group']]

        # Save updated CSV (overwrite)
        final_df.to_csv(subset_path, index=False)
        logger.info(f"Updated {subset_file} with group info ({len(final_df)} rows)")

# ==============================
# MAIN EXECUTION
# ==============================
def main():
    metadata_df = load_and_prepare_metadata(METADATA_FILE)
    update_subset_csvs(DATA_DIR, metadata_df, SUBSET_FILES)
    logger.info("All subset CSVs updated successfully.")

if __name__ == "__main__":
    main()
