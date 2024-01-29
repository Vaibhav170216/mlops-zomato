from pipelines.deployment_pipeline import continuous_deployment_pipeline, inference_pipeline
import click
from typing import cast
from rich import print
from zenml.integrations.mlflow.mlflow_utils import get_tracking_uri
from zenml.integrations.mlflow.model_deployers.mlflow_model_deployer import MLFlowModelDeployer
from zenml.integrations.mlflow.services import MLFlowDeploymentService


DEPLOY = "deploy"
PREDICT = "predict"
DEPLOY_AND_PREDICT = "deploy_and_predict"

@click.command()
@click.option(
    "--config",
    "-c",
    type = click.Choice([DEPLOY, PREDICT, DEPLOY_AND_PREDICT]),
    default = DEPLOY_AND_PREDICT,
    help = "optionally you can choose to run only the deployment"
    "pipeline to train and deploy a model, or to only run the"
    "prediction against the deployed model. By default both will"
    "be run.",
)

@click.option(
    "--min-accuracy",
    default = 0.0,
    help = "Minimum accuracy required to deploy the model.",
)

def run_deployment(config: str, min_accuracy: float):
    mlflow_model_deployer_component = MLFlowModelDeployer.get_active_model_deployer()
    deploy = config == DEPLOY or config == DEPLOY_AND_PREDICT
    predict = config == PREDICT or config == DEPLOY_AND_PREDICT
    if deploy:
        continuous_deployment_pipeline(
            data_path = "C:/Users/vaib3/mlops/data/zomato.csv",
            min_accuracy = min_accuracy,
            workers = 1,
            timeout = 60,
            )
    if predict:
        inference_pipeline(
            pipeline_name = "continuous_deployment_pipeline",
            pipeline_step_name = "mlflow_mdoel_deployer_step",
        )

    print(
        "You can run:\n"
        f"mlflow ui --backend-store-uri {get_tracking_uri()}"
    )   

    existing_services = mlflow_model_deployer_component.find_model_server(
        pipeline_name = "continuous_deployment_pipeline",
        pipeline_step_name = "mlflow_model_deployer_step",
        mode_name = "model",
    )

    if existing_services:
        service = cast(MLFlowDeploymentService, existing_services[0])
        service.start(timeout=60)
        if service.is_running:
            print(
                f"The MLFlow prediction server is running locally as a daemon"
                f"process service at: {service.prediction_url}"
            )
        elif service.is_failed:
            print(
                f"The MLFlow prediction server is in a failed state"
                f"Last state: '{service.status.state.value}'\n"
                f"Last error: '{service.status.error}'"
            )
    else:
        print(
            "No MLFlow predcition server is currently running. The deployment pipeline must be run to train a model and deploy it."
        )

if __name__ == "__main__":
    run_deployment()