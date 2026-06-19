"""
PURPOSE:
    Removes rows with extreme values from the dataset using Z-score filtering.
    Only numeric columns are analyzed. Delegates to src/outlier_detection.py.
"""

import logging
import pandas as pd
from zenml import step
from src.outlier_detection import OutlierDetector, ZScoreOutlierDetection

logger = logging.getLogger(__name__)

@step
def outlier_detection_step(df: pd.DataFrame, method: str = "remove",threshold: float = 3.0)-> pd.DataFrame:
    logger.info("Detecting and removing outlier using OutlierDetector")

    if df is None:
        logger.info("Recieved a Nonetype Dataframe")
        raise ValueError("Input df must be a non-null pandas DataFrame")
    
    if not isinstance(df, pd.DataFrame):
        logger.info(f"Expected pandas dataframe, got {type(df)} instead.")
        raise ValueError("Input df must be a pandas DataFrame")

    outlier_detection = OutlierDetector(ZScoreOutlierDetection(threshold))
    df_cleaned = outlier_detection.handle_outlier(df,method)
    return df_cleaned