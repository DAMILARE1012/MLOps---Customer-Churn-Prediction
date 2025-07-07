# from zenml import BaseParameters
from pydantic import BaseModel

"""
This file is used for used for configuring
and specifying various parameters related to 
your machine learning models and training process
"""

class ModelName(BaseModel):
    """
    Model configurations
    """
    model_name: str = "logistic regression"