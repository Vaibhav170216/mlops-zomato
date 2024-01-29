from pipelines.training_pipeline import train_pipeline
from zenml.integrations.mlflow.mlflow_utils import get_tracking_uri
from zenml.client import Client
import mlflow


if __name__ == "__main__":
    train_pipeline(data_path = "C:/Users/vaib3/mlops/data/zomato.csv")
    print(f"Now run mlflow ui --backend-store-uri '{get_tracking_uri()}'\n"
         "To inspect your experiment runs within the mlflow UI.")
