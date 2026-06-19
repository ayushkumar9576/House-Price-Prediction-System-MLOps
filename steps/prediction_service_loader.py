"""
PURPOSE:
    Looks up the MLflow deployment service that was started by the deployment pipeline.
    It queries ZenML's MLFlowModelDeployer to find a running prediction server matching
    the given pipeline_name and step_name. Raises RuntimeError if no server is found.
"""

import logging
from zenml import step
from zenml.integrations.mlflow.model_deployers import MLFlowModelDeployer

logger = logging.getLogger(__name__)
    
@step(enable_cache=False)
def prediction_service_loader(pipeline_name: str, step_name: str):
    logger.info("Getting the prediction service started by the deployment pipeline")
    model_deployer = MLFlowModelDeployer.get_active_model_deployer()

    existing_services = model_deployer.find_model_server(pipeline_name= pipeline_name, pipeline_step_name= step_name)

    if not existing_services:
        raise RuntimeError(
            f"No MLflow prediction service deployed by the "
            f"{step_name} step in the {pipeline_name} "
            f"pipeline is currently "
            f"running."
        )

    return existing_services[0]
