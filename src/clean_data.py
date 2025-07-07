import pandas as pd
import numpy as np
import logging
from sklearn.model_selection import train_test_split
from abc import abstractmethod, ABC
from typing import Union
from sklearn.preprocessing import LabelEncoder

class DataStrategy(ABC):
    @abstractmethod
    def handle_data(self, df:pd.DataFrame) -> Union[pd.DataFrame,pd.Series]:
        pass
        
    
# Data Preprocessing strategy
class DataPreprocessing(DataStrategy):
    """
        1. DataPreprocessing: This class is responsible 
        for removing unwanted columns and handling missing values (NA values) in the dataset.
    """
    def handle_data(self, df: pd.DataFrame) -> Union[pd.DataFrame, pd.Series]:
        try:
            df['TotalCharges'] = df['TotalCharges'].replace(' ', 0).astype(float)
            df.drop('customerID', axis=1, inplace=True)
            df['Churn'] = df['Churn'].replace({'Yes': 1, 'No': 0}).astype(int)
            service = ['PhoneService', 'MultipleLines', 'InternetService', 'OnlineSecurity',
                       'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV',
                       'StreamingMovies']
            for col in service:
                df[col] = df[col].replace({'No phone service': 'No', 'No internet service': 'No'})
            logging.info(f"Length of df: {len(df.columns)}")
            return df
        except Exception as e:
            logging.error("Error in Preprocessing", e)
            raise e

# Feature Encoding Strategy
class LabelEncoding(DataStrategy):
    """
        2. LabelEncoding: The LabelEncoding class is designed to encode categorical variables 
        into a numerical format that machine learning algorithms can work with effectively. 
        It transforms text-based categories into numeric values.
    """
    def handle_data(self, df: pd.DataFrame) -> Union[pd.DataFrame, pd.Series]:
        try:
            df_cat = ['gender', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines',
                      'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
                      'TechSupport', 'StreamingTV',  'StreamingMovies', 'Contract',
                      'PaperlessBilling', 'PaymentMethod']
            lencod = LabelEncoder()
            for col in df_cat:
                df[col] = lencod.fit_transform(df[col])
            logging.info(df.head())
            return df
        except Exception as e:
            logging.error(e)
            raise e
            
# Data splitting Strategy
class DataDivideStrategy(DataStrategy):
    """
        3. DataDivideStrategy: This class separates the dataset into 
           independent variables(X) and dependent variables (y). 
           Then, it splits the data into training and testing sets.
    """
    def handle_data(self, df:pd.DataFrame) -> Union[pd.DataFrame, pd.Series]:
        try:
            X = df.drop('Churn', axis=1)
            y = df['Churn']
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
            return X_train, X_test, y_train, y_test
        except Exception as e:
            logging.error("Error in DataDividing", e)
            raise e