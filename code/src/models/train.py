"""
Train Models Module

This module provides utilities for training machine learning models on tabular or
time-series features, with support for:

- Standard train/test split evaluation
- Monte Carlo cross-validation (with optional group-aware splitting)
- Hyperparameter tuning via grid search or Optuna
- Automatic label encoding for models requiring numeric labels
- Feature scaling pipelines for robust training
- Saving trained models and evaluation metrics for reproducibility
"""

import time
from typing import Optional, Tuple, Any, Dict

import numpy as np
import pandas as pd
import optuna
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import GroupShuffleSplit, StratifiedKFold, GroupKFold
from sklearn.base import clone
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from models.classifiers import get_model
from models.base import load_data, split_dataset, save_model
from models.tune import grid_search_tuning
from utils.file_utils import load_config


# ==========================
# --- Helper functions ---
# ==========================
def _eval_on_split(estimator, X_train, y_train, X_val, y_val) -> Dict[str, float]:
    """
    Fit estimator on training data and evaluate on validation data.

    Args:
        estimator: sklearn-like estimator with fit/predict interface.
        X_train, y_train: Training features and labels.
        X_val, y_val: Validation features and labels.

    Returns:
        Dict with metrics: accuracy, weighted F1, precision, recall.
    """
    est = clone(estimator)
    est.fit(X_train, y_train)
    y_pred = est.predict(X_val)
    return {
        "accuracy": float(accuracy_score(y_val, y_pred)),
        "f1": float(f1_score(y_val, y_pred, average="weighted")),
        "precision": float(precision_score(y_val, y_pred, average="weighted", zero_division=0)),
        "recall": float(recall_score(y_val, y_pred, average="weighted", zero_division=0))
    }


def _aggregate_metrics(list_of_dicts: list) -> dict:
    """
    Aggregate a list of metric dictionaries into mean and standard deviation.

    Args:
        list_of_dicts: List of dicts containing numeric metrics.

    Returns:
        Dict with keys of form 'metric_mean' and 'metric_std'.
    """
    if not list_of_dicts:
        return {}
    keys = list_of_dicts[0].keys()
    agg = {}
    for k in keys:
        agg[f"{k}_mean"] = float(np.mean([d[k] for d in list_of_dicts]))
        agg[f"{k}_std"] = float(np.std([d[k] for d in list_of_dicts], ddof=1))
    return agg


# ==========================
# --- Standard Train/Test Split ---
# ==========================
def train_model_simple(
    features_csv: str,
    model_name: str = 'random_forest',
    test_size: float = 0.2,
    random_state: int = 42,
    save_model_path: Optional[str] = None,
    save_metrics_path: Optional[str] = None,
    tune: bool = False,
    config_path: str = 'hyperparameters.yaml',
    **kwargs
) -> Tuple[Any, Tuple[pd.DataFrame, pd.Series]]:
    """
    Train a model using a standard train/test split.

    Args:
        features_csv: Path to CSV containing features and 'class' column.
        model_name: Model identifier (used by get_model).
        test_size: Fraction of data used for testing.
        random_state: Random seed for reproducibility.
        save_model_path: Optional path to save trained model.
        save_metrics_path: Optional path to save evaluation metrics CSV.
        tune: Whether to perform grid search tuning.
        config_path: Path to hyperparameter config file (YAML).
        kwargs: Additional parameters to pass to model constructor.

    Returns:
        Tuple of (trained model, (X_test, y_test)).
    """
    config = load_config(config_path) if tune else None
    X, y = load_data(features_csv)

    # Encode labels for models requiring numeric labels
    le = None
    if model_name.lower() in ['xgboost', 'm3gp', 'lightgbm']:
        le = LabelEncoder()
        y = le.fit_transform(y)

    X_train, X_test, y_train, y_test = split_dataset(X, y, test_size, random_state, stratify=y)

    # Hyperparameter tuning if requested
    if tune and config:
        param_grid = config.get(model_name, {}).get('param_grid', {})
        base = get_model(model_name, **kwargs)
        model = grid_search_tuning(base, param_grid, X_train, y_train, cv=5, scoring='f1_weighted', n_jobs=-1)
    else:
        model = get_model(model_name, **kwargs).fit(X_train, y_train)

    # Save model
    if save_model_path:
        save_model(model, save_model_path)

    # Save metrics
    if save_metrics_path:
        y_pred = model.predict(X_test)
        if le: y_test, y_pred = le.inverse_transform(y_test), le.inverse_transform(y_pred)
        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "f1_weighted": f1_score(y_test, y_pred, average='weighted'),
            "precision_weighted": precision_score(y_test, y_pred, average='weighted'),
            "recall_weighted": recall_score(y_test, y_pred, average='weighted')
        }
        pd.DataFrame([metrics]).to_csv(save_metrics_path, index=False)

    return model, (X_test, y_test)


# ==========================
# --- Monte Carlo CV with optional group-aware splits ---
# ==========================
def train_model_monte_carlo(
    features_csv: str,
    model_name: str = 'random_forest',
    n_splits: int = 10,
    test_size: float = 0.2,
    random_seed: int = 42,
    save_model_path: Optional[str] = None,
    save_metrics_path: Optional[str] = None,
    tune: bool = False,
    config_path: str = 'hyperparameters.yaml',
    group_column: Optional[str] = None,
    n_trials: int = 50,
    groups: Optional[pd.Series] = None,
    optuna_timeout: Optional[int] = None,
    **kwargs
) -> Tuple[dict, Any, Optional[LabelEncoder]]:
    """
    Train a model using Monte Carlo cross-validation, with optional hyperparameter tuning
    via Optuna and group-aware splitting.

    Args:
        features_csv: Path to CSV with features and 'class' column.
        model_name: Model identifier.
        n_splits: Number of Monte Carlo splits.
        test_size: Fraction of data reserved for testing in each split.
        random_seed: Random seed for reproducibility.
        save_model_path: Optional path to save final trained model.
        save_metrics_path: Optional path to save aggregated metrics.
        tune: Whether to perform hyperparameter tuning.
        config_path: Path to hyperparameter configuration (YAML).
        group_column: Column name to use for group-aware splitting.
        n_trials: Number of Optuna trials for hyperparameter search.
        groups: Optional explicit group array for group-aware splitting.
        optuna_timeout: Optional maximum runtime for Optuna search in seconds.
        kwargs: Additional parameters passed to model constructor.

    Returns:
        Tuple containing:
        - Aggregated metrics dictionary (mean/std of Monte Carlo CV)
        - Final best-fitted model
        - Optional LabelEncoder (if labels were encoded)
    """
    # --- Load features ---
    df = pd.read_csv(features_csv)
    if "class" not in df.columns:
        raise ValueError("Feature CSV must contain a 'class' column.")

    non_features = ['file_path', 'filename', 'group', 'Unnamed: 0']
    X = df.drop(columns=['class'] + [c for c in non_features if c in df.columns])
    y = df["class"].copy()

    # --- Label encoding ---
    le = None
    if model_name.lower() in ['xgboost', 'm3gp', 'lightgbm']:
        le = LabelEncoder()
        y = le.fit_transform(y)

    # --- Group-aware splitting ---
    if groups is None:
        groups = df[group_column] if group_column and group_column in df.columns else None

    # --- Build pipeline ---
    def build_pipeline(model_params):
        estimator = get_model(model_name, model_params=model_params, **kwargs)
        return Pipeline([("scaler", StandardScaler()), ("estimator", estimator)])

    # --- Hyperparameter tuning with Optuna ---
    best_params = kwargs.get("model_params", {})
    if tune:
        config = load_config(config_path)
        model_cfg = config.get(model_name, {})
        search_space = model_cfg.get("optuna_search") or {k: {"type": "categorical", "values": v} 
                                                         for k, v in model_cfg.get("param_grid", {}).items()}

        if search_space:
            def _sampler(trial):
                params = {}
                for p, spec in search_space.items():
                    t = spec.get("type", "categorical")
                    if t == "int":
                        params[p] = trial.suggest_int(p, spec["low"], spec["high"])
                    elif t == "float":
                        params[p] = trial.suggest_float(p, spec["low"], spec["high"], log=spec.get("log", False))
                    elif t == "categorical":
                        params[p] = trial.suggest_categorical(p, spec["values"])
                    elif t == "bool":
                        params[p] = trial.suggest_categorical(p, [False, True])
                    else:
                        raise ValueError(f"Unsupported param type {t} for {p}")
                return params

            def objective(trial):
                params = _sampler(trial)
                pipe = build_pipeline(params)
                # Inner CV
                if groups is not None and group_column in df.columns:
                    unique_groups = df[group_column].nunique()
                    n_splits_inner = min(3, max(2, unique_groups // 5))
                    cv_splitter = GroupKFold(n_splits=n_splits_inner)
                    scores = []
                    for tr_idx, val_idx in cv_splitter.split(X, y, groups):
                        scores.append(f1_score(y.iloc[val_idx], pipe.fit(X.iloc[tr_idx], y.iloc[tr_idx]).predict(X.iloc[val_idx]), average="weighted"))
                    return float(np.mean(scores))
                else:
                    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=random_seed)
                    scores = []
                    for tr_idx, val_idx in cv.split(X, y):
                        scores.append(f1_score(y.iloc[val_idx], pipe.fit(X.iloc[tr_idx], y.iloc[tr_idx]).predict(X.iloc[val_idx]), average="weighted"))
                    return float(np.mean(scores))

            study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=random_seed))
            study.optimize(objective, n_trials=n_trials, timeout=optuna_timeout)
            best_params = study.best_params

    # --- Monte Carlo CV ---
    metrics_list, fitted_models, f1_list = [], [], []
    splitter = GroupShuffleSplit(n_splits=n_splits, test_size=test_size, random_state=random_seed)
    split_iterator = splitter.split(X, y, groups) if groups is not None else splitter.split(X, y)

    for train_idx, val_idx in split_iterator:
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        pipe = build_pipeline(best_params)
        start = time.time()
        pipe.fit(X_train, y_train)
        duration = time.time() - start

        val_metrics = _eval_on_split(pipe, X_train, y_train, X_val, y_val)
        val_metrics["train_time_sec"] = duration

        metrics_list.append(val_metrics)
        fitted_models.append(pipe)
        f1_list.append(val_metrics["f1"])

    aggregated = _aggregate_metrics(metrics_list)
    best_idx = int(np.argmax(f1_list))
    best_model = clone(fitted_models[best_idx])
    best_model.fit(X, y)  # refit on full dataset
    aggregated["best_params"] = best_params

    # --- Save outputs ---
    if save_model_path:
        save_model(best_model, save_model_path)
    if save_metrics_path:
        pd.DataFrame([aggregated]).to_csv(save_metrics_path, index=False)

    return aggregated, best_model, le
