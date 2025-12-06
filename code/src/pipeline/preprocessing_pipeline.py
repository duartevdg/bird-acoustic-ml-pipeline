"""
Preprocessing Pipeline
======================

A reusable pipeline for running sequential audio preprocessing steps on:
- folders of audio files
- lists of individual audio file paths

Supports:
- automatic temporary folder management
- cleanup of intermediate steps
- detailed logging of progress and outputs
"""

import os
import shutil
import logging
from utils.helpers import safe_rmtree

class PreprocessingPipeline:
    """
    Runs a series of preprocessing steps on audio data.
    Each step is a tuple of (function, config_dict).
    """

    def __init__(self, steps, logger=None, temp_dir=None):
        """
        Initialize the pipeline.

        Args:
            steps (list): List of tuples (function, config_dict) in execution order.
            logger (logging.Logger, optional): Logger to use. Defaults to root logger.
            temp_dir (str): Temporary folder for intermediate outputs. Must be provided.
        """
        self.steps = steps
        self.logger = logger or logging.getLogger(__name__)
        if temp_dir is None:
            raise ValueError("Temporary directory must be provided")
        self.temp_dir = os.path.abspath(temp_dir)
        os.makedirs(self.temp_dir, exist_ok=True)

    # ----------------------------
    # Folder-based processing
    # ----------------------------
    def run(self, input_folder: str) -> str:
        """
        Run the pipeline on all files in a folder.

        Args:
            input_folder (str): Path to the input folder.

        Returns:
            str: Path to the folder containing the final step output.
        """
        current_input = os.path.abspath(input_folder)
        for idx, (func, config) in enumerate(self.steps):
            step_output = os.path.join(self.temp_dir, f"step_{idx}_{func.__name__}")
            os.makedirs(step_output, exist_ok=True)

            try:
                if func.__name__ == "normalize_dataset":
                    func(current_input, step_output, config)
                else:
                    func(current_input, step_output, **config)

                # Delete previous step folder if not the original input
                if idx > 0 and current_input != input_folder and os.path.exists(current_input):
                    safe_rmtree(current_input, self.logger)
                    self.logger.info(f"Deleted previous step folder: {current_input}")

                current_input = step_output

            except Exception as e:
                self.logger.error(f"Error running step {func.__name__}: {e}", exc_info=True)
                raise

        return current_input

    # ----------------------------
    # File-list-based processing
    # ----------------------------
    def run_files(self, file_paths: list[str]) -> str:
        """
        Run the pipeline on a list of individual audio file paths.

        Args:
            file_paths (list[str]): List of audio file paths.

        Returns:
            str: Path to the folder containing the final step output.
        """
        current_input = [os.path.abspath(f) for f in file_paths]

        for idx, (func, config) in enumerate(self.steps):
            step_output = os.path.join(self.temp_dir, f"step_{idx}_{func.__name__}")
            os.makedirs(step_output, exist_ok=True)

            try:
                if func.__name__ == "normalize_dataset":
                    # Copy files to temporary folder for processing
                    temp_folder = os.path.join(self.temp_dir, f"temp_input_{idx}")
                    os.makedirs(temp_folder, exist_ok=True)
                    for f in current_input:
                        shutil.copy2(f, os.path.join(temp_folder, os.path.basename(f)))

                    # Ensure output is WAV
                    config["format"] = "wav"
                    func(temp_folder, step_output, config)
                    safe_rmtree(temp_folder, self.logger)

                else:
                    func(current_input, step_output, **config)

                # Log some files after this step
                files_after_step = []
                for root, _, files in os.walk(step_output):
                    files_after_step.extend(files)
                self.logger.info(
                    f"[DEBUG] Step {idx} ({func.__name__}) output files: "
                    f"{files_after_step[:5]}{'...' if len(files_after_step) > 5 else ''}"
                )

                current_input = step_output

            except Exception as e:
                self.logger.error(f"Error running step {func.__name__}: {e}", exc_info=True)
                raise

        if not os.listdir(current_input):
            self.logger.warning(f"No files in final step output: {current_input}")

        return current_input
