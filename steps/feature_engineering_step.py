"""
PURPOSE:
    Applies a feature transformation (log, scaling, or encoding) to specified columns.
    Delegates to src/feature_engineering.py (Strategy pattern).
"""

import logging
import pandas as pd
from zenml import step
from src.feature_engineering import FeatureEngineering, LogTransformation, MinMaxScaling, OneHotEncoding, StandardScaling

logger = logging.getLogger(__name__)

@step
def feature_engineering_step(df: pd.DataFrame, strategy: str = "log", features: list[str] | None = None)-> pd.DataFrame:
    logger.info("Performing feature engineering using FeatureEngineering and selected strategy")

    if features is None:
        features = []

    if strategy == "log":
        engineer = FeatureEngineering(LogTransformation(features))
    elif strategy == "standard_scaling":
        engineer = FeatureEngineering(StandardScaling(features))
    elif strategy == "minmax_scaling":
        engineer = FeatureEngineering(MinMaxScaling(features))
    elif strategy == "onehot_encoding":
        engineer = FeatureEngineering(OneHotEncoding(features))
    else:
        raise ValueError(f"Unsupported feature engineering strategy: {strategy}")

    transformed_df = engineer.fit_transform(df)
    logger.info("feature engineering completed successfully")
    return transformed_df