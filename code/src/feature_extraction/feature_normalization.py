"""
Data Normalization and Feature Selection Module

Preprocesses training and testing datasets by optionally normalizing features,
imputing missing values, and performing feature selection or dimensionality
reduction (SelectKBest or PCA).
"""

import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.decomposition import PCA

def normalize_and_select(train_df: pd.DataFrame,
                         test_df: pd.DataFrame,
                         label_col: str,
                         metadata_cols: list = None,
                         normalize: bool = True,
                         feature_selection: bool = True,
                         k: int = 50,
                         selection_method: str = 'f_classif') -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Preprocesses datasets by imputing missing values, normalizing features,
    and performing feature selection or dimensionality reduction.

    Args:
        train_df (pd.DataFrame): Training dataset including features and labels.
        test_df (pd.DataFrame): Test dataset including features and labels.
        label_col (str): Column name of the label/class.
        metadata_cols (list, optional): Columns to exclude from processing. Defaults to [].
        normalize (bool, optional): Whether to standardize features. Defaults to True.
        feature_selection (bool, optional): Whether to select top k features or apply PCA. Defaults to True.
        k (int, optional): Number of features/components to select. Defaults to 50.
        selection_method (str, optional): 'f_classif', 'mutual_info', or 'pca'. Defaults to 'f_classif'.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]: Processed training and testing DataFrames with labels and metadata preserved.
    """
    if metadata_cols is None:
        metadata_cols = []

    # ----------------------------
    # Separate features and labels
    # ----------------------------
    exclude = [label_col] + metadata_cols
    X_train, y_train = train_df.drop(columns=exclude), train_df[label_col]
    X_test = test_df.drop(columns=exclude)

    # ----------------------------
    # Impute missing values
    # ----------------------------
    imputer = SimpleImputer(strategy="mean")
    X_train = pd.DataFrame(imputer.fit_transform(X_train), columns=X_train.columns)
    X_test = pd.DataFrame(imputer.transform(X_test), columns=X_test.columns)

    # ----------------------------
    # Normalize features
    # ----------------------------
    if normalize:
        scaler = StandardScaler()
        X_train = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)
        X_test = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)

    # ----------------------------
    # Feature selection or dimensionality reduction
    # ----------------------------
    if feature_selection:
        n_features = min(k, X_train.shape[1])
        if selection_method == 'f_classif':
            selector = SelectKBest(score_func=f_classif, k=n_features)
            X_train = pd.DataFrame(selector.fit_transform(X_train, y_train))
            X_test = pd.DataFrame(selector.transform(X_test))
        elif selection_method == 'mutual_info':
            selector = SelectKBest(score_func=mutual_info_classif, k=n_features)
            X_train = pd.DataFrame(selector.fit_transform(X_train, y_train))
            X_test = pd.DataFrame(selector.transform(X_test))
        elif selection_method == 'pca':
            pca = PCA(n_components=n_features)
            X_train = pd.DataFrame(pca.fit_transform(X_train))
            X_test = pd.DataFrame(pca.transform(X_test))
        else:
            raise ValueError(f"Unknown selection_method: {selection_method}")

    # ----------------------------
    # Combine features with labels/metadata
    # ----------------------------
    train_df_processed = pd.concat([train_df[[label_col] + metadata_cols].reset_index(drop=True),
                                    X_train.reset_index(drop=True)], axis=1)
    test_df_processed = pd.concat([test_df[[label_col] + metadata_cols].reset_index(drop=True),
                                   X_test.reset_index(drop=True)], axis=1)

    return train_df_processed, test_df_processed
