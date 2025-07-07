from zenml import pipeline
from steps.ingest_data import ingest_df
from steps.data_cleaning import cleaning_data
import logging
from steps.train_model import train_model
from steps.evaluation import evaluate_model
from steps.config import ModelName
#Define a ZenML pipeline called training_pipeline.

@pipeline(enable_cache=False)
def train_pipeline(data_path:str):
    '''
    Data pipeline for training the model.

    Args:
        data_path (str): The path to the data to be ingested.
    '''
    df = ingest_df(data_path=data_path)
    X_train, X_test, y_train, y_test = cleaning_data(df=df)
    
    # Train the model
    model = train_model(X_train=X_train, y_train=y_train, config=ModelName(model_name="logistic regression"))
    
    # Evaluate the model
    accuracy_score, precision_score = evaluate_model(model=model,X_test=X_test,y_test=y_test)
    

    
