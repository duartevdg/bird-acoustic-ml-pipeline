"""
Xeno-Canto Bird Recordings Downloader
=====================================

This script downloads audio recordings for a given bird species from Xeno-Canto
using the `xenopy` library. Metadata is retrieved, and recordings are saved
to a user-specified folder.

Instructions:
- Set `SPECIES_NAME` to the desired species (scientific name).
- Set `DEST_DIR` to your desired download folder.
- Adjust `QUALITY_THRESHOLD` to filter recordings (A is best, D is low quality).
"""

import os
import logging
from tqdm import tqdm
from xenopy import Query

# -----------------------------
# USER CONFIGURATION
# -----------------------------
SPECIES_NAME = "Turdus iliacus"  # Example species
DEST_DIR = r"C:\path\to\your\download\folder"  # Change to your folder
QUALITY_THRESHOLD = "D"  # Only download recordings better than this

# Number of parallel processes for downloading
NUM_PROCESSES = 10
MAX_ATTEMPTS = 10

# -----------------------------
# SETUP LOGGING
# -----------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s — %(levelname)s — %(message)s"
)
logger = logging.getLogger(__name__)

# -----------------------------
# FUNCTION DEFINITIONS
# -----------------------------
def download_species_recordings(species_name: str, dest_dir: str,
                                quality_threshold: str = "D",
                                num_processes: int = 10,
                                max_attempts: int = 10) -> None:
    """
    Download all audio recordings for a given species from Xeno-Canto.

    Parameters
    ----------
    species_name : str
        Scientific name of the bird species (e.g., "Turdus iliacus").
    dest_dir : str
        Path to folder where recordings will be saved.
    quality_threshold : str
        Minimum quality of recordings to download (A, B, C, D).
    num_processes : int
        Number of parallel processes to use for downloads.
    max_attempts : int
        Maximum number of retry attempts per recording.
    """
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)
        logger.info(f"Created destination folder: {dest_dir}")

    logger.info(f"Retrieving metadata for species: {species_name} (quality > {quality_threshold})")
    q = Query(name=species_name, q_gt=quality_threshold)
    meta = q.retrieve_meta(verbose=True)
    total_recordings = len(meta)

    if total_recordings == 0:
        logger.warning(f"No recordings found for species '{species_name}' with quality > {quality_threshold}")
        return

    pages_needed = (total_recordings // 300) + (1 if total_recordings % 300 else 0)
    logger.info(f"Found {total_recordings} recordings across {pages_needed} pages")

    with tqdm(total=pages_needed, desc=f"Downloading {species_name}", unit="pages") as progress_bar:
        for page in range(1, pages_needed + 1):
            q.page = page
            q.retrieve_recordings(
                multiprocess=True,
                nproc=num_processes,
                attempts=max_attempts,
                outdir=dest_dir
            )
            progress_bar.update(1)

    logger.info(f"Download completed for species: {species_name}")

# -----------------------------
# MAIN EXECUTION
# -----------------------------
if __name__ == "__main__":
    download_species_recordings(
        species_name=SPECIES_NAME,
        dest_dir=DEST_DIR,
        quality_threshold=QUALITY_THRESHOLD,
        num_processes=NUM_PROCESSES,
        max_attempts=MAX_ATTEMPTS
    )
