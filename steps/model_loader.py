"""
PURPOSE:
    Retrieves a previously trained sklearn Pipeline from ZenML's Model Control Plane.
    The model is looked up by name and the "production" version tag.
"""

import logging
from sklearn.pipeline import Pipeline
from zenml import step, Model

logger = logging.getLogger(__name__)

@step
def model_loader(model_name: str)-> Pipeline:
    logger.info("Loading the current production model pipeline")

    model = Model(name= model_name, version="production")

    model_pipeline: Pipeline = model.load_artifact("sklearn_pipeline")

    return model_pipeline