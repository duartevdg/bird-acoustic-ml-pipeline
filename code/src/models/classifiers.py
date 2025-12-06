"""
Model Factory Module

Utility to return ML models by name with optional parameters.
New models can be added to the `models` dictionary.
"""

from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from m3gp import M3GP
from sktime.transformations.panel.rocket import MiniRocket

def get_model(model_name: str, model_params: dict = None, **kwargs):
    """
    Return an instantiated machine learning model.

    Args:
        model_name (str): Name of the model (case-insensitive).
        model_params (dict): Parameters to pass to the model constructor.
        **kwargs: Additional parameters.

    Returns:
        Instantiated model object.
    """
    model_name = model_name.lower()
    model_params = model_params or {}
    final_params = {**kwargs, **model_params}

    # Special case: M3GP wraps a RandomForest base
    if model_name == "m3gp":
        base_model = RandomForestClassifier(max_depth=model_params.get("max_depth", 6))
        return M3GP(model_class=base_model, fitnessType="2FOLD", **kwargs)

    # Dictionary of standard models
    models = {
        "random_forest": RandomForestClassifier,
        "xgboost": XGBClassifier,
        "minirocket": MiniRocket,
        "lightgbm": LGBMClassifier
    }

    if model_name not in models:
        raise ValueError(f"Unknown model: {model_name}. Available models: {list(models.keys())}")

    return models[model_name](**final_params)