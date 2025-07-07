import pandas as pd
from sklearn.linear_model import LogisticRegression
from abc import ABC, abstractmethod
import logging


#Abstract model
class Model(ABC):
    """
    We define an abstract Model class with a ‘train’ method that all models must implement. 
    The LogisticReg class is a specific implementation using logistic regression. 
    The next step involves configuring a file named config.py in the steps folder.
    """
    @abstractmethod
    def train(self,X_train:pd.DataFrame,y_train:pd.Series):
        """
        Trains the model on given data
        """
        pass
    

class LogisticReg(Model):
    """
    Implementing the Logistic Regression model.
    """
    def train(self, X_train: pd.DataFrame, y_train: pd.Series):
        """
        Training the model
        
        Args:
            X_train: pd.DataFrame,
            y_train: pd.Series
        """
        logistic_reg = LogisticRegression()
        logistic_reg.fit(X_train, y_train)
        return logistic_reg