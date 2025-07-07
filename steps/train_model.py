import logging
 
import pandas as pd
from src.train_model import LogisticReg
from zenml import step
from .config import ModelName

#Define a step called train_model
@step(enable_cache=False)
def train_model(X_train:pd.DataFrame,y_train:pd.Series,config:ModelName):
    """
    Trains the data based on the configured model
        
    """
    try:
        model = None
        if config.model_name == "logistic regression":
            model = LogisticReg()
        else:
            raise ValueError("Model name is not supported")
        
        trained_model = model.train(X_train=X_train,y_train=y_train)
        return trained_model
    
    except Exception as e:
        logging.error("Error in step training model",e)
        raise e