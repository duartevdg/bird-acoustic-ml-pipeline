"""
Evaluate Models Module

This module provides a robust implementation of Monte Carlo cross-validation for
evaluating machine learning models on tabular or time-series features. It supports
group-aware splitting to avoid leakage, parallelized model training, and aggregation
of metrics over multiple splits.

Features:
- Group-aware or standard Monte Carlo splitting
- Parallel training and evaluation with joblib
- Aggregation of metrics including mean, std, and confusion matrices
- Automatic best-model selection based on F1 score
- Handles both pandas DataFrames and numpy arrays
- Optionally aggregates predictions at a file/group level
- Designed for integration with any sklearn-like estimator or custom model factory
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import ShuffleSplit, GroupShuffleSplit
from sklearn.metrics import confusion_matrix
from joblib import Parallel, delayed
from tqdm import tqdm
from typing import Callable, Optional, Tuple, Dict, Any, Union
from models.base import evaluate_model


def evaluate_model_direct(
    model,
    features_csv: str,
    label_column: str = "class",
    encoder=None
) -> Dict[str, float]:
    """
    Evaluate a trained model on a features CSV. Optionally aggregates predictions by file/group.

    Args:
        model: Trained model implementing a `.predict()` method.
        features_csv: Path to CSV file containing features and labels.
        label_column: Column name of the target variable.
        encoder: Optional label encoder to inverse-transform predictions for categorical targets.

    Returns:
        A dictionary of evaluation metrics (accuracy, precision, recall, F1 score).
    """
    df = pd.read_csv(features_csv)
    if label_column not in df.columns:
        raise ValueError(f"[ERROR] Label column '{label_column}' not found in features CSV.")

    # Drop irrelevant columns
    X = df.drop(columns=[label_column, "file_path", "filename", "group"], errors="ignore")
    y_true = df[label_column]

    # Make predictions
    y_pred = model.predict(X)
    if encoder:
        y_pred = encoder.inverse_transform(y_pred)

    # Aggregate by file_id if present
    if "file_id" in df.columns:
        agg_true = df.groupby("file_id")[label_column].first()
        agg_pred = pd.Series(y_pred, index=df.index).groupby(df["file_id"]).agg(
            lambda x: x.value_counts().idxmax()
        )
        y_true, y_pred = agg_true.values, agg_pred.values

    labels = sorted(np.unique(y_true))
    return evaluate_model(y_true, y_pred, labels=labels)


def _safe_index(arr: Union[pd.DataFrame, pd.Series, np.ndarray], idx):
    """Safe indexing helper compatible with pandas and numpy arrays."""
    return arr.iloc[idx] if hasattr(arr, "iloc") else arr[idx]


def _train_and_evaluate(
    model_factory: Callable,
    model_kwargs: Optional[Dict[str, Any]],
    X_train, y_train,
    X_test, y_test,
    labels,
    random_seed: int = 42
) -> Optional[Tuple[Dict[str, float], np.ndarray, Any]]:
    """
    Train and evaluate a single model on a single split.

    Returns:
        Tuple of (metrics dict, confusion matrix, trained model) or None if training fails.
    """
    try:
        model_kwargs = model_kwargs.copy() if model_kwargs else {}
        if "random_state" not in model_kwargs and random_seed is not None:
            model_kwargs["random_state"] = random_seed

        model = model_factory(**model_kwargs)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        metrics = evaluate_model(y_test, y_pred, labels=labels)
        cm = confusion_matrix(y_test, y_pred, labels=labels)
        return metrics, cm, model

    except Exception as e:
        print(f"[ERROR] Training/evaluation failed: {e}")
        return None


def monte_carlo_cross_validation(
    model_factory: Callable,
    X: Union[pd.DataFrame, np.ndarray],
    y: Union[pd.Series, np.ndarray],
    n_splits: int = 10,
    test_size: float = 0.2,
    decimals: int = 4,
    random_seed: int = 42,
    model_kwargs: Optional[Dict[str, Any]] = None,
    n_jobs: int = -1,
    groups: Optional[Union[pd.Series, np.ndarray]] = None
) -> Tuple[Dict[str, float], Any]:
    """
    Perform Monte Carlo cross-validation with optional group-aware splitting.

    Args:
        model_factory: Callable that returns a new model instance.
        X: Feature matrix (pandas DataFrame or numpy array).
        y: Target labels (pandas Series or numpy array).
        n_splits: Number of Monte Carlo splits.
        test_size: Fraction of data used as test set.
        decimals: Decimal precision for aggregated metrics.
        random_seed: Seed for reproducibility.
        model_kwargs: Optional parameters for model instantiation.
        n_jobs: Number of parallel workers for joblib.
        groups: Optional array for group-aware splits to prevent leakage.

    Returns:
        Tuple containing:
        - Aggregated metrics dictionary (mean, std, confusion matrix)
        - Best performing trained model (by F1 score)
    """
    model_kwargs = model_kwargs or {}
    labels = sorted(np.unique(y))

    # Build splits
    if groups is not None:
        cv = GroupShuffleSplit(n_splits=n_splits, test_size=test_size, random_state=random_seed)
        splits = list(cv.split(X, y, groups))
        # Check for group leakage
        for i, (train_idx, test_idx) in enumerate(splits):
            overlap = set(groups[train_idx]).intersection(groups[test_idx])
            if overlap:
                raise AssertionError(f"Group leakage detected in split {i}: {overlap}")
    else:
        cv = ShuffleSplit(n_splits=n_splits, test_size=test_size, random_state=random_seed)
        splits = list(cv.split(X))

    # Parallel training/evaluation
    results = Parallel(n_jobs=n_jobs)(
        delayed(_train_and_evaluate)(
            model_factory, model_kwargs,
            _safe_index(X, train_idx), _safe_index(y, train_idx),
            _safe_index(X, test_idx), _safe_index(y, test_idx),
            labels, random_seed
        )
        for train_idx, test_idx in tqdm(splits, desc="Monte Carlo CV")
    )

    results = [r for r in results if r is not None]
    if not results:
        raise RuntimeError("All training/evaluation runs failed.")

    metrics_list, cms, models = zip(*results)

    # Aggregate metrics
    agg_metrics = {
        f"{k}_mean": round(np.mean([m[k] for m in metrics_list]), decimals)
        for k in metrics_list[0]
    }
    agg_metrics.update({
        f"{k}_std": round(np.std([m[k] for m in metrics_list]), decimals)
        for k in metrics_list[0]
    })
    agg_metrics["confusion_matrix_mean"] = np.round(np.mean(cms, axis=0), decimals)

    # Select best model by F1 score
    best_model = models[np.argmax([m["f1_score"] for m in metrics_list])]

    return agg_metrics, best_model
