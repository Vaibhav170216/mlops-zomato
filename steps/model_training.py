import logging
import pandas as pd
import mlflow
from zenml.client import Client
from zenml import step
from src.model_development import LinearRegressionModel
from sklearn.base import RegressorMixin
from .config import ModelNameConfig

experiment_tracker = Client().active_stack.experiment_tracker

@step(experiment_tracker = experiment_tracker.name)
def model_train(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    config: ModelNameConfig,
) -> RegressorMixin:
    
    try:
        model = None
        if config.model_name == "LinearRegression":
            mlflow.sklearn.autolog()
            model = LinearRegressionModel()
            trained_model = model.train(X_train, y_train)
            return trained_model
        else:
            raise ValueError(f"Model {config.model_name} is not supported.")
    except Exception as e:
        logging.error(f"Error in training model {e}.")
        raise e
    

    