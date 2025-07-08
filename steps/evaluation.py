import logging
import pandas as pd
import numpy as np
from zenml import step
from src.evaluate_model import  F1_Score, Accuracy_score, Precision_Score
from typing import Tuple
from typing_extensions import Annotated
from sklearn.base import ClassifierMixin
from sklearn.metrics import confusion_matrix, classification_report, recall_score
from zenml.client import Client

experiment_tracker = Client().active_stack.experiment_tracker

@step(experiment_tracker = experiment_tracker.name, enable_cache=False)
def evaluate_model(
    model: ClassifierMixin,
    X_test: pd.DataFrame,
    y_test: pd.Series
) -> Tuple[
    Annotated[float,"accuracy_score"],
    Annotated[float,"precision_score"]
]:
    """
    Evaluate a machine learning model's performance using common metrics.
    """
    try:
        y_pred = model.predict(X_test)

        precision_score_class = Precision_Score()
        precision_score = precision_score_class.evaluate_model(y_pred=y_pred, y_true=y_test)

        accuracy_score_class = Accuracy_score()
        accuracy_score = accuracy_score_class.evaluate_model(y_true=y_test, y_pred=y_pred)

        recall = recall_score(y_test, y_pred)
        conf_matrix = confusion_matrix(y_test, y_pred)
        class_report = classification_report(y_test, y_pred)

        return accuracy_score, precision_score

    except Exception as e:
        logging.error("Error in evaluating model",e)
        raise e