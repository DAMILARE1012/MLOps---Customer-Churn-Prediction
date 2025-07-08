from pipelines.training_pipeline import train_pipeline
from zenml.client import Client

if __name__ == "__main__":
    print(Client().active_stack.experiment_tracker.get_tracking_uri())
    data_path = "../data/WA_Fn-UseC_-Telco-Customer-Churn.csv"
    train_pipeline(data_path)