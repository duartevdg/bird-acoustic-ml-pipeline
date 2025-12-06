"""
Model Utilities Module

Functions for loading data, splitting datasets, saving/loading models,
and evaluating classification performance.
"""

import os
import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import train_test_split

def load_data(features_csv: str):
    """
    Load features and labels from a CSV file.

    Args:
        features_csv (str): Path to CSV containing features and 'class' column.

    Returns:
        X (pd.DataFrame): Feature matrix.
        y (pd.Series): Class labels.
    """
    df = pd.read_csv(features_csv)
    X = df.drop(columns=['file_path', 'class'], errors='ignore')
    y = df['class']
    return X, y

def split_dataset(X, y, test_size: float = 0.2, random_state: int = 42, stratify=None):
    """
    Split data into train and test sets.

    Args:
        X (pd.DataFrame): Feature matrix.
        y (pd.Series): Labels.
        test_size (float): Fraction of data for testing.
        random_state (int): Random seed.
        stratify: Stratification labels for split (usually y).

    Returns:
        X_train, X_test, y_train, y_test
    """
    return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=stratify)

def save_model(model, filepath: str):
    """
    Save a trained model using joblib.

    Args:
        model: Trained scikit-learn or compatible model.
        filepath (str): Destination file path.
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    joblib.dump(model, filepath)

def load_model(filepath: str):
    """
    Load a saved model.

    Args:
        filepath (str): Path to saved joblib model.

    Returns:
        Loaded model object.
    """
    return joblib.load(filepath)

def evaluate_model(y_true, y_pred, labels=None):
    """
    Evaluate classification performance using weighted metrics.

    Args:
        y_true: Ground-truth labels.
        y_pred: Predicted labels.
        labels: Optional list of labels to include in metrics.

    Returns:
        dict: Dictionary containing accuracy, precision, recall, and f1_score.
    """
    acc = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average='weighted', zero_division=0
    )
    return {
        'accuracy': acc,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
    }
