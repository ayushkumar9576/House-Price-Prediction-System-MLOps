"""
PURPOSE:
    Splits the cleaned DataFrame into training features (X_train), testing features (X_test),
    training target (y_train), and testing target (y_test) using an 80/20 random split.
    Delegates to src/data_splitter.py (Strategy pattern).
"""
import logging
from typing import Tuple
import pandas as pd
from src.data_splitter import DataSplitter, SimpleTrainTestSplit
from zenml import step

logger = logging.getLogger(__name__)

@step
def data_splitter_step(df: pd.DataFrame, target_columns: str)-> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    logger.info("Spliting the data into training and testing set using the DataSplitter and selected strategy")
    
    splitter = DataSplitter(strategy= SimpleTrainTestSplit())
    X_train, X_test, y_train, y_test = splitter.split(df, target_columns)

    logger.info("data splitting completed successfully")
    return X_train, X_test, y_train, y_test
