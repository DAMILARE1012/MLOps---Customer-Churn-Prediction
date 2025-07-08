import logging
import mlflow
import pandas as pd
from src.train_model import LogisticReg
from sklearn.base import ClassifierMixin

from zenml import step
from .config import ModelName
from zenml.client import Client

experiment_tracker = Client().active_stack.experiment_tracker
print(f"MLflow URI: {experiment_tracker.get_tracking_uri()}")

#Define a step called train_model
@step(experiment_tracker = experiment_tracker.name, enable_cache=False)
def train_model(
    X_train:pd.DataFrame,
    y_train:pd.Series,
    config:ModelName
    ):
    """
    rains the data based on the configured model
    Args:
        X_train: pd.DataFrame = Independent training data,
        y_train: pd.Series = Dependent training data.
        
    """
    try:
        model = None
        if config.model_name == "logistic regression":
            # Automatically log details about the model - scores, models etc
            mlflow.sklearn.autolog()
            model = LogisticReg()
        else:
            raise ValueError("Model name is not supported")
        
        trained_model = model.train(X_train=X_train,y_train=y_train)
        logging.info(f"Model trained successfully")
        return trained_model
    
    except Exception as e:
        logging.error("Error in step training model",e)
        raise e