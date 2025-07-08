
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import click  # For handling command-line arguments
import logging  
from typing import cast
from rich import print  # For console output formatting

# Import pipelines for deployment and inference
from pipelines.deployment_pipeline import ( 
continuous_deployment_pipeline, inference_pipeline
)
# Import MLflow utilities and components
from zenml.integrations.mlflow.mlflow_utils import get_tracking_uri
from zenml.integrations.mlflow.model_deployers.mlflow_model_deployer import ( 
MLFlowModelDeployer
)
from zenml.integrations.mlflow.services.mlflow_deployment import MLFlowDeploymentService

# Define constants for different configurations: DEPLOY, PREDICT, DEPLOY_AND_PREDICT
DEPLOY = "deploy"
PREDICT = "predict"
DEPLOY_AND_PREDICT = "deploy_and_predict"

# Define a main function that uses Click to handle command-line arguments
@click.command()
@click.option(
    "--config",
    "-c",
    type=click.Choice([DEPLOY, PREDICT, DEPLOY_AND_PREDICT]),
    default=DEPLOY_AND_PREDICT,
    help="Optionally you can choose to only run the deployment "
    "pipeline to train and deploy a model (`deploy`), or to "
    "only run a prediction against the deployed model "
    "(`predict`). By default both will be run "
    "(`deploy_and_predict`).",
)
@click.option(
    "--min-accuracy",
    default=0.92,
    help="Minimum accuracy required to deploy the model",
)
def run_main(config:str, min_accuracy:float ):
    # Get the active MLFlow model deployer component
    mlflow_model_deployer_component = MLFlowModelDeployer.get_active_model_deployer()
    
    # Determine if the user wants to deploy a model (deploy), make predictions (predict), or both (deploy_and_predict)
    deploy = config == DEPLOY or config == DEPLOY_AND_PREDICT
    predict = config == PREDICT or config == DEPLOY_AND_PREDICT
    
    # If deploying a model is requested:
    if deploy:
        continuous_deployment_pipeline(
            data_path='./data/WA_Fn-UseC_-Telco-Customer-Churn.csv',
            min_accuracy=min_accuracy,
            workers=3,
            timeout=60
        )
    
    # If making predictions is requested:
    if predict:
        # Initialize an inference pipeline run
        inference_pipeline(
            pipeline_name="continuous_deployment_pipeline",
            pipeline_step_name="mlflow_model_deployer_step",
        )
    
    # Print instructions for viewing experiment runs in the MLflow UI
    print(
        "You can run:\n "
        f"[italic green]    mlflow ui --backend-store-uri '{get_tracking_uri()}"
        "[/italic green]\n ...to inspect your experiment runs within the MLflow"
        " UI.\nYou can find your runs tracked within the "
        "`mlflow_example_pipeline` experiment. There you'll also be able to "
        "compare two or more runs.\n\n"
    )
    
    # Fetch existing services with the same pipeline name, step name, and model name
    existing_services = mlflow_model_deployer_component.find_model_server(
        pipeline_name = "continuous_deployment_pipeline",
        pipeline_step_name = "mlflow_model_deployer_step",
    )
    
    # Check the status of the prediction server:
    if existing_services:
        service = cast(MLFlowDeploymentService, existing_services[0])
        if service.is_running:
            print(
                f"The MLflow prediciton server is running locally as a daemon"
                f"process service and accepts inference requests at: \n"
                f"     {service.prediction_url}\n"
                f"To stop the service, run"
                f"[italic green] zenml model-deployer models delete"
                f"{str(service.uuid)}'[/italic green]."
            )
        elif service.is_failed:
            print(
                f"The MLflow prediciton server is in a failed state: \n"
                f" Last state: '{service.status.state.value}'\n"
                f" Last error: '{service.status.last_error}'"
            )
    else:
        print(
            "No MLflow prediction server is currently running. The deployment"
            "pipeline must run first to train a model and deploy it. Execute"
            "the same command with the '--deploy' argument to deploy a model."
        )
        
# Entry point: If this script is executed directly, run the main function
if __name__ == "__main__":
    run_main()