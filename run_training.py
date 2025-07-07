#!/usr/bin/env python3
"""
Main script to run the customer churn prediction training pipeline.
Run this from the project root directory.
"""

from pipelines.training_pipeline import train_pipeline

if __name__ == "__main__":
    data_path = "data/WA_Fn-UseC_-Telco-Customer-Churn.csv"
    df = train_pipeline(data_path)
    print("Pipeline completed successfully!")
    print(f"DataFrame shape: {df.shape}")
    print("\nFirst few rows:")
    print(df.head()) 