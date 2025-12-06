"""
Tune Models Module

This module provides utilities for hyperparameter tuning of scikit-learn compatible models
using grid search or randomized search.

Functions:
- grid_search_tuning: Standard exhaustive GridSearchCV with printing and returning the fitted grid object.
- grid_search_tuning_best: Returns only the best estimator from GridSearchCV.
"""

from typing import Any, Dict, Optional
from sklearn.model_selection import GridSearchCV
import pandas as pd


def grid_search_tuning(
    model: Any,
    param_grid: Dict[str, list],
    X: pd.DataFrame,
    y: pd.Series,
    cv: int = 5,
    scoring: str = 'f1_weighted',
    n_jobs: int = -1,
    verbose: int = 1
) -> GridSearchCV:
    """
    Perform exhaustive grid search cross-validation for hyperparameter tuning.

    Args:
        model: scikit-learn compatible estimator.
        param_grid: Dictionary of parameters to search. Keys are parameter names, values are lists of options.
        X: Feature matrix (DataFrame or array-like).
        y: Target vector (Series or array-like).
        cv: Number of cross-validation folds.
        scoring: Metric for evaluation (default: 'f1_weighted').
        n_jobs: Number of parallel jobs (-1 uses all cores).
        verbose: Verbosity level for GridSearchCV.

    Returns:
        Fitted GridSearchCV object with results accessible via attributes like `best_params_` and `best_score_`.

    Example:
        >>> grid = grid_search_tuning(RandomForestClassifier(), {'n_estimators': [100, 200]}, X_train, y_train)
        >>> print(grid.best_params_)
    """
    grid = GridSearchCV(model, param_grid, cv=cv, scoring=scoring, n_jobs=n_jobs, verbose=verbose)
    grid.fit(X, y)

    print(f"[INFO] Best parameters: {grid.best_params_}")
    print(f"[INFO] Best {scoring}: {grid.best_score_:.4f}")

    return grid


def grid_search_tuning_best(
    model: Any,
    param_grid: Dict[str, list],
    X: pd.DataFrame,
    y: pd.Series,
    cv: int = 5,
    scoring: str = 'f1_weighted',
    n_jobs: int = -1,
    verbose: int = 1
) -> Any:
    """
    Perform grid search and return only the best estimator.

    Args:
        model: scikit-learn compatible estimator.
        param_grid: Dictionary of parameters to search.
        X: Feature matrix.
        y: Target vector.
        cv: Number of cross-validation folds.
        scoring: Metric for evaluation.
        n_jobs: Number of parallel jobs.
        verbose: Verbosity level.

    Returns:
        The estimator with the best parameters found during grid search.

    Example:
        >>> best_model = grid_search_tuning_best(RandomForestClassifier(), {'n_estimators': [100, 200]}, X_train, y_train)
    """
    grid = grid_search_tuning(model, param_grid, X, y, cv=cv, scoring=scoring, n_jobs=n_jobs, verbose=verbose)
    return grid.best_estimator_
