import joblib
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from src.clean_data import DataPreprocessing, LabelEncoding
import os

def save_model_for_deployment():
    """Save the trained model locally for deployment"""
    
    # Load and prepare data
    df = pd.read_csv('./data/WA_Fn-UseC_-Telco-Customer-Churn.csv')
    
    # Preprocess data
    data_preprocessing = DataPreprocessing()
    data = data_preprocessing.handle_data(df)
    
    # Encode features
    label_encode = LabelEncoding()
    df_encoded = label_encode.handle_data(data)
    
    # Prepare features and target
    X = df_encoded.drop(['Churn'], axis=1)
    y = df_encoded['Churn']
    
    # Train model
    model = LogisticRegression(random_state=42, max_iter=1000)
    model.fit(X, y)
    
    # Save model and feature names
    os.makedirs('./saved_model', exist_ok=True)
    joblib.dump(model, './saved_model/churn_model.pkl')
    joblib.dump(X.columns.tolist(), './saved_model/feature_names.pkl')
    
    print("✅ Model saved successfully!")
    print(f"Model saved to: ./saved_model/churn_model.pkl")
    print(f"Feature names saved to: ./saved_model/feature_names.pkl")
    print(f"Number of features: {len(X.columns)}")
    
    # Test the saved model
    test_prediction = model.predict(X.head(1))
    print(f"Test prediction: {test_prediction[0]}")
    
    return model, X.columns.tolist()

if __name__ == "__main__":
    save_model_for_deployment() 