"""
PURPOSE:
    Builds a scikit-learn Pipeline (preprocessing + LinearRegression), trains it
    on X_train/y_train, and logs all parameters, metrics, and artifacts to MLflow.
"""

import logging
from typing import Annotated
import mlflow
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from zenml import step, ArtifactConfig, Model
from zenml.client import Client
from zenml.enums import ArtifactType

logger = logging.getLogger(__name__)

@step(
    enable_cache=False,
    model=Model(
        name="price_predictor",
        version=None,
        license="Apache 2.0",
        description="Price Prediction Model For House.",
    ),
)
def model_building_step(X_train: pd.DataFrame, y_train: pd.Series) -> Annotated[Pipeline, ArtifactConfig(name="sklearn_pipeline", artifact_type=ArtifactType.MODEL)]:
    experiment = Client().active_stack.experiment_tracker
    if experiment is None:
        raise RuntimeError("No experiment tracker configured on the active ZenML stack")
    if not isinstance(X_train, pd.DataFrame):
        raise TypeError("X_train must be a pandas DataFrame")
    if not isinstance(y_train, pd.Series):
        raise TypeError("y_train must be a pandas Series")
    
    categorial_cols = X_train.select_dtypes(include=["object", "category"]).columns
    numerical_cols = X_train.select_dtypes(exclude=["object", "category"]).columns

    logger.info(f"Categorial columns: {categorial_cols.tolist()}")
    logger.info(f"Numerical columns: {numerical_cols.tolist()}")

    numerical_transform = SimpleImputer(strategy="mean")
    categorial_transform = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numerical_transform, numerical_cols),
            ("cat", categorial_transform, categorial_cols),
        ]
    )

    pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("model", LinearRegression())])

    started_run = False
    if not mlflow.active_run():
        started_run = True
        mlflow.start_run()
    
    try:
        mlflow.sklearn.autolog()

        logger.info("Building and training the linear regression model")
        pipeline.fit(X_train, y_train)
        logger.info("Model Training Completion")

        if len(categorial_cols) > 0:
            onehot_encoder = (pipeline.named_steps["preprocessor"].transformers_[1][1].named_steps["onehot"])
            expected_columns = numerical_cols.tolist() + list(
                onehot_encoder.get_feature_names_out(categorial_cols)
            )
        else:
            expected_columns = numerical_cols.tolist()
        logger.info(f"Model expects the following columns: {expected_columns}")

    except Exception as e:
        logger.error(f"Error during model training: {e}")
        raise
    
    finally:
        if started_run:
            mlflow.end_run()

    return pipeline
