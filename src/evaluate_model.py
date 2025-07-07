import logging
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_score, f1_score
from abc import ABC, abstractmethod
import numpy as np

# Abstract class for model evaluation
class Evaluate(ABC):
    @abstractmethod
    def evaluate_model(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Abstract method to evaluate a machine learning model's performance.

        Args:
            y_true (np.ndarray): True labels.
            y_pred (np.ndarray): Predicted labels.

        Returns:
            float: Evaluation result.
        """
        pass


#Class to calculate accuracy score
class Accuracy_score(Evaluate):
    """
        Calculates and returns the accuracy score for a model's predictions.

    """
        
    def evaluate_model(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        try:
            accuracy_scr = accuracy_score(y_true=y_true, y_pred=y_pred) * 100
            logging.info(f"Accuracy_score: {accuracy_scr}")  
            return accuracy_scr  
        except Exception as e:
            logging.error("Error in evaluating the accuracy of the model",e)
            raise e
#Class to calculate Precision score
class Precision_Score(Evaluate):
    def evaluate_model(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Generates and returns a precision score for a model's predictions.

        """
        try:
            precision = precision_score(y_true=y_true,y_pred=y_pred)
            logging.info(f"Precision score: {precision}")
            return float(precision)
        except Exception as e:
            logging.error("Error in calculation of precision_score",e)
            raise e

class F1_Score(Evaluate):
    def evaluate_model(self, y_true: np.ndarray, y_pred: np.ndarray):
        """
        Generates and returns an F1 score for a model's predictions.
        
        """
        try:
            f1_scr = f1_score(y_pred=y_pred, y_true=y_true)
            logging.info(f"F1 score: {f1_scr}") 
            return f1_scr
        except Exception as e:
            logging.error("Error in calculating F1 score", e)
            raise e