from pipelines.training_pipeline import train_pipeline

if __name__ == "__main__":
    data_path = "../data/WA_Fn-UseC_-Telco-Customer-Churn.csv"
    train_pipeline(data_path)