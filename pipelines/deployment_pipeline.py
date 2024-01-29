import numpy as np
import pandas as pd
import json
from zenml import pipeline, step
from zenml.config import DockerSettings
from zenml.constants import DEFAULT_SERVICE_START_STOP_TIMEOUT
from zenml.integrations.constants import MLFLOW
from zenml.integrations.mlflow.model_deployers.mlflow_model_deployer import MLFlowModelDeployer
from zenml.integrations.mlflow.services import MLFlowDeploymentService
from zenml.integrations.mlflow.steps import mlflow_model_deployer_step
from zenml.steps import BaseParameters, Output
from .utils import get_data_for_test

from steps.ingesting_data import ingest_data
from steps.cleaning_data import clean_data
from steps.model_training import model_train
from steps.evaluate import evaluate_model

docker_settings = DockerSettings(required_integrations = [MLFLOW])

class DeploymentTriggerConfig(BaseParameters):
    min_accuracy: float = 0.0

@step(enable_cache = False)
def dynamic_importer() -> str:
    data = get_data_for_test()
    return data

@step
def deployment_trigger(config: DeploymentTriggerConfig, accuracy: float):
    return accuracy >= config.min_accuracy

class MLFLowDeploymentLoaderStepParameters(BaseParameters):
    pipeline_name: str
    step_name: str
    running: bool = True

@step(enable_cache = False)
def prediction_service_loader(
    pipeline_name: str,
    pipeline_step_name: str,
    running: bool = True,
    model_name: str = "model",
) -> MLFlowDeploymentService:
    
    mlflow_model_deployer_component = MLFlowModelDeployer.get_active_model_deployer()

    existing_services = mlflow_model_deployer_component.find_model_server(
        pipeline_name = pipeline_name,
        pipeline_step_name = pipeline_step_name,
        model_name = model_name,
        running = running,
    )
    if not existing_services:
        raise RuntimeError(
            f"No MLFlow deployment service is found for the {pipeline_name}"
        )
    return existing_services[0]

@step
def predictor(
    service: MLFlowDeploymentService,
    data: str,
) -> np.ndarray:
    
    service.start(timeout = 10)
    data.pop("columns")
    data.pop("index")
    columns_for_df = [
        "rate",
        "num of ratings",
    ]
    df = pd.DataFrame(data["data"], columns = columns_for_df)
    json_list = json.loads(json.dumps(list(df.T.to_dict().values())))
    data = np.array(json_list)
    prediction = service.predicT(data)
    return prediction

@pipeline(enable_cache = False, settings = {"docker": docker_settings})
def continuous_deployment_pipeline(
    data_path: str,
    min_accuracy: float = 0.0,
    workers: int = 1,
    timeout: int = DEFAULT_SERVICE_START_STOP_TIMEOUT
):
    df = ingest_data(data_path = data_path)
    X_train, X_test, y_train, y_test = clean_data(df)
    model = model_train(X_train, y_train)
    r2, rmse = evaluate_model(model, X_test, y_test)
    deployment_decision = deployment_trigger(r2)
    mlflow_model_deployer_step(
        model = model,
        deploy_decision = deployment_decision,
        workers = workers,
        timeout = timeout,
    )

@pipeline(enable_cache = False, settings = {"docker": docker_settings})
def inference_pipeline(pipeline_name: str, pipeline_step_name: str):
    data = dynamic_importer()
    service = prediction_service_loader(
        pipeline_name = pipeline_name,
        pipeline_step_name = pipeline_step_name,
        running = False,
    )
    prediction = predictor(service = service, data = data)
    return prediction