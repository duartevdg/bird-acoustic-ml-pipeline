import os
import itertools
import logging
import pandas as pd
from utils.helpers import append_result_row, combo_code, setup_file_logging, safe_rmtree, merge_feature_csvs, passes_filters, clean_temp_except_splits
from utils.file_utils import load_config, load_all_configs, find_wav_files_recursive
from pipeline.preprocessing_pipeline import PreprocessingPipeline
from models.train import train_model_monte_carlo
from models.evaluate import evaluate_model_direct
from preprocessing import normalization, denoising, preemphasis, amplitude, segmentation, augmentation
from feature_extraction.batch_extract import extract_multiple
from feature_extraction.time_domain import extract_time_features
from feature_extraction.frequency_domain import extract_frequency_features
from feature_extraction.time_frequency import extract_time_frequency_features
from feature_extraction.cepstral_domain import extract_cepstral_features
from sklearn.model_selection import GroupShuffleSplit
from feature_extraction.feature_normalization import normalize_and_select
from tqdm import tqdm
from utils.helpers import notify
import time

logging.basicConfig(level=logging.INFO,format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

FEATURE_EXTRACTORS_MAP = {
    "time": (extract_time_features, "features_time.csv"),
    "frequency": (extract_frequency_features, "features_frequency.csv"),
    "time_frequency": (extract_time_frequency_features, "features_time_frequency.csv"),
    "cepstral": (extract_cepstral_features, "features_cepstral.csv")
}


def run_all_combinations_pipeline(n_jobs=4):
    logger.info("[START] Running all combinations pipeline")

    if n_jobs == -1:
        n_jobs = os.cpu_count() or 1
        logger.info(f"Using all available CPUs: {n_jobs} workers")

    paths, pipeline_config, _ = load_all_configs()
    run_options = load_config("config/run_options.yaml")

    logger.info(f"[DEBUG] Run Options Loaded:\n{run_options}")

    data_path = paths["data_path"]
    features_folder = paths["features_path"]
    models_path = paths["models_path"]
    logs_path = paths["logs_path"]
    results_path = paths["results_path"]
    hp_config_path = os.path.join(paths["configs_path"], "hyperparameters.yaml")
    temp_path = os.path.join(paths["temp_path"])

    setup_file_logging(logger, logs_path, name="final_pipeline")

    subsets = run_options.get("data_subsets", [])
    allowed_extractors = run_options.get("feature_extractors", list(FEATURE_EXTRACTORS_MAP.keys()))
    model_names = run_options.get("model_names", [])
    all_aug_methods = run_options.get("augmentation_methods", [])

    all_results = []

    data_csv_files = [os.path.join(data_path, f"{subset}.csv") for subset in subsets]
    logger.info(f"Found {len(data_csv_files)} subsets to process.")

    for csv_idx, csv_file in enumerate(data_csv_files, 1):
        subset = os.path.splitext(os.path.basename(csv_file))[0]
        metadata_df = pd.read_csv(csv_file)
        logger.info(f"[{csv_idx}/{len(data_csv_files)}] Subset '{subset}' contains {len(metadata_df)} recordings")

        file_group_map = {row['file_path']: row['group'] for _, row in metadata_df.iterrows()}

        # Train/test split
        gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
        train_idx, test_idx = next(gss.split(metadata_df, groups=metadata_df["group"]))
        train_df = metadata_df.iloc[train_idx]
        test_df = metadata_df.iloc[test_idx]

        split_train_paths = train_df["file_path"].tolist()
        split_test_paths = test_df["file_path"].tolist()

        logger.info(f"Train/Test split done for subset {subset}: {len(split_train_paths)} train, {len(split_test_paths)} test")

        # Generate preprocessing/model combinations
        combos = list(itertools.product(
            run_options['formats'],
            run_options['sampling_rates'],
            run_options['bit_depths'],
            run_options['channels'],
            run_options['amplitude_methods'],
            run_options['pre_emphasis_coefficients'],
            run_options['denoising_methods'],
            run_options['segmentation_methods'],
            run_options['augmentation_methods']
        ))
        logger.info(f"Processing {len(combos)} preprocessing/model combinations for subset {subset}")

        num_valid_combos = sum(1 for c in combos if passes_filters(c, run_options))
        num_domains = len([d for d in allowed_extractors])
        total_tasks = num_valid_combos * num_domains * len(model_names)
        task_counter = 0

        # Iterate over all combinations with progress
        for combo in tqdm(combos, desc=f"[{subset}] Combos", unit="combo"):
            if not passes_filters(combo, run_options):
                logger.info(f"[SKIP] Combo filtered out: {combo}")
                continue

            fmt, sr, bd, ch, amp, preemph, denoise, segment, augment = combo
            combo_tag = f"{subset}_{fmt}_{sr}Hz_{bd}bit_{ch}_D{denoise}__PE{preemph}_Amp{amp}_S{segment}_Aug{augment}"
            logger.info(f"\n[COMBO] {combo_tag} starting")
            notify(f"\n[COMBO] {combo_tag} starting")

            combo_dict = {
                "subset": subset,
                "format": fmt,
                "sampling_rate": sr,
                "bit_depth": bd,
                "channels": ch,
                "amplitude_method": amp,
                "pre_emphasis": preemph,
                "denoising": denoise,
                "segmentation": segment,
                "augmentation": augment,
            }

            # Build preprocessing pipelines
            train_steps = []
            if fmt != "none":
                train_steps.append((normalization.normalize_dataset, {"format": fmt, "sampling_rate": int(sr), "bit_depth": int(bd), "channels": ch}))
            if denoise != "none":
                train_steps.append((denoising.denoise_dataset, {"method": denoise}))
            if preemph != "none":
                train_steps.append((preemphasis.apply_preemphasis_to_dataset, {"coeff": float(preemph)}))
            if amp != "none":
                train_steps.append((amplitude.amp_normalize_dataset, {"method": amp}))
            if segment != "none":
                train_steps.append((segmentation.segment_dataset, {"method": segment}))
            if augment != "none":
                train_steps.append((augmentation.augment_dataset, {"methods": all_aug_methods}))

            test_steps = train_steps[:-1] if augment != "none" else train_steps.copy()

            train_pipeline = PreprocessingPipeline(train_steps, logger, temp_dir=os.path.join(temp_path, f"{combo_tag}_train"))
            test_pipeline = PreprocessingPipeline(test_steps, logger, temp_dir=os.path.join(temp_path, f"{combo_tag}_test"))

            try:
                logger.info(f"[{combo_tag}] Preprocessing train data")
                train_preprocessed_folder = train_pipeline.run_files(split_train_paths)
                logger.info(f"[{combo_tag}] Preprocessing test data")
                test_preprocessed_folder = test_pipeline.run_files(split_test_paths)

                train_wav_files = find_wav_files_recursive(train_preprocessed_folder)
                test_wav_files = find_wav_files_recursive(test_preprocessed_folder)
                if not train_wav_files or not test_wav_files:
                    logger.warning(f"[{combo_tag}] No audio files found after preprocessing, skipping.")
                    continue

                # Feature folders
                train_feature_folder = os.path.join(features_folder, combo_tag, "train")
                test_feature_folder = os.path.join(features_folder, combo_tag, "test")
                os.makedirs(train_feature_folder, exist_ok=True)
                os.makedirs(test_feature_folder, exist_ok=True)

                def get_base_file_id(f):
                    return os.path.splitext(os.path.basename(f))[0].split('_')[0]

                def get_species_from_path(f, df):
                    file_id = get_base_file_id(f)
                    return df.loc[df['file_id'].astype(str) == file_id, 'species'].values[0]

                def get_group_from_path(f, df):
                    file_id = get_base_file_id(f)
                    return df.loc[df['file_id'].astype(str) == file_id, 'group'].values[0]

                def get_id_from_path(f, df):
                    return get_base_file_id(f)

                train_df_min = pd.DataFrame({
                    "file_path": train_wav_files,
                    "species": [get_species_from_path(f,metadata_df) for f in train_wav_files],
                    "group": [get_group_from_path(f,metadata_df) for f in train_wav_files],
                    "file_id": [get_id_from_path(f,metadata_df) for f in train_wav_files] 
                })

                test_df_min = pd.DataFrame({
                    "file_path": test_wav_files,
                    "species": [get_species_from_path(f,metadata_df) for f in test_wav_files],
                    "group": [get_group_from_path(f,metadata_df) for f in test_wav_files],
                    "file_id": [get_id_from_path(f,metadata_df) for f in test_wav_files] 
                })

                # Extract features with progress bar
                extractors = [(FEATURE_EXTRACTORS_MAP[ext][0], FEATURE_EXTRACTORS_MAP[ext][1]) for ext in allowed_extractors if ext in FEATURE_EXTRACTORS_MAP]
                logger.info(f"[{combo_tag}] Extracting features from train data")
                extract_multiple(train_df_min, train_feature_folder, extractors, n_workers=n_jobs)
                logger.info(f"[{combo_tag}] Extracting features from test data")
                extract_multiple(test_df_min, test_feature_folder, extractors, n_workers=n_jobs)
                logger.info(f"[{combo_tag}] Feature extraction completed")

                # Clean up
                safe_rmtree(train_preprocessed_folder, logger)
                safe_rmtree(test_preprocessed_folder, logger)
                clean_temp_except_splits(temp_path, subset, logger)

                # Domains to train
                domains_to_run = allowed_extractors.copy()
                logger.info(f"[DEBUG] Domains to run: {domains_to_run}")

                if "combined" in domains_to_run:
                    train_features_csv_combined = os.path.join(train_feature_folder, "combined_features.csv")
                    test_features_csv_combined = os.path.join(test_feature_folder, "combined_features.csv")

                    merged_train_df = merge_feature_csvs(train_feature_folder, train_features_csv_combined, logger=logger)
                    merged_test_df = merge_feature_csvs(test_feature_folder, test_features_csv_combined, logger=logger)

                for domain in domains_to_run:
                    if domain == "combined":
                        train_features_csv = train_features_csv_combined
                        test_features_csv = test_features_csv_combined
                    else:
                        train_features_csv = os.path.join(train_feature_folder, FEATURE_EXTRACTORS_MAP[domain][1])
                        test_features_csv = os.path.join(test_feature_folder, FEATURE_EXTRACTORS_MAP[domain][1])

                    if not os.path.exists(train_features_csv) or not os.path.exists(test_features_csv):
                        logger.warning(f"[{combo_tag}] Skipping domain {domain}: CSV missing")
                        continue

                    train_df_features = pd.read_csv(train_features_csv)
                    test_df_features = pd.read_csv(test_features_csv)
                    
                    normalize_flag= run_options.get("normalize", False)
                    select_flag=run_options.get("select",False)
                    k = run_options.get("top_k_features", 50)

                    train_df_features, test_df_features = normalize_and_select(
                        train_df_features,
                        test_df_features,
                        label_col="class",
                        metadata_cols=[c for c in ["file_id","file_path","group"] if c in train_df_features.columns],
                        normalize=normalize_flag,
                        feature_selection=select_flag,
                        k=k
                    )
                    
                    metadata_cols_to_drop = [c for c in ["file_id","file_path","group"] if c in train_df_features.columns]
                    groups_array = train_df_features["group"].values
                    train_df_model = train_df_features.drop(columns=[c for c in metadata_cols_to_drop if c in train_df_features.columns])
                    test_df_model = test_df_features.drop(columns=[c for c in metadata_cols_to_drop if c in test_df_features.columns])

                    train_df_model.to_csv(train_features_csv, index=False)
                    test_df_model.to_csv(test_features_csv, index=False)

                    # Train models
                    for model_name in model_names:
                        try:
                            all_models = run_options.get("models", {})
                            kwargs = all_models.get(model_name, {})
                            logger.info(f"[{combo_tag}] Training model '{model_name}' on domain '{domain}' with params {kwargs if kwargs else 'N/A'}")

                            model_output_path = os.path.join(models_path, f"{combo_tag}_{model_name}_{domain}.joblib")
                            start_time=time.time()
                            metrics, best_model, le = train_model_monte_carlo(
                                features_csv=train_features_csv,
                                model_name=model_name,
                                n_splits=2,
                                test_size=0.2,
                                random_seed=42,
                                save_model_path=model_output_path,
                                save_metrics_path=None,
                                tune=True,
                                config_path=hp_config_path,
                                groups=groups_array,
                                **kwargs
                            )
                            end_time = time.time()
                            train_duration = end_time - start_time  # seconds
                            test_metrics = evaluate_model_direct(best_model, test_features_csv, label_column="class", encoder=le)
                            logger.info(f"[TRAIN EVAL] {metrics}")
                            logger.info(f"[TEST EVAL] {test_metrics}")

                            row = {
                                **combo_dict,
                                "model": model_name,
                                "domain": domain,
                                **metrics,
                                **{f"test_{k}": v for k, v in test_metrics.items()},
                                "normalized": normalize_flag,
                                "selected_top_k": k if select_flag else None,
                                "train_time_sec": train_duration
                            }

                            key_columns = ["subset", "format", "sampling_rate", "bit_depth", "channels",
                                        "amplitude_method", "pre_emphasis", "denoising", "segmentation",
                                        "augmentation", "model", "domain", "normalized", "selected_top_k"]
                            append_result_row(row, os.path.join(results_path, "final.csv"), key_columns=key_columns)
                            task_counter += 1
                            notify(f"✅ ({task_counter}/{total_tasks}) {subset} | {model_name} | {domain} | {combo_code(combo_dict)} | Test Acc: {test_metrics.get('accuracy', 'N/A')}")


                        except Exception as e:
                            logger.error(f"[ERROR] Model training/evaluation failed for domain {domain}: {e}")
                            continue

            except Exception as e:
                logger.error(f"[ERROR] Pipeline failed for combo {combo_tag}: {e}")
                continue

