"""
AVAILABLE STRATEGIES:
    1. LinearRegressionStrategy — Creates a Pipeline with StandardScaler + LinearRegression.
       (More strategies like RandomForest, XGBoost, etc. can be added as new subclasses.)
"""

import logging
from abc import ABC, abstractmethod
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.base import RegressorMixin

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

class ModelBuildingStrategy(ABC):
    @abstractmethod
    def build_and_train_model(self, X_train: pd.DataFrame, y_train: pd.Series)-> RegressorMixin:
        pass

class LinearRegressionStrategy(ModelBuildingStrategy):
    def build_and_train_model(self, X_train: pd.DataFrame, y_train: pd.Series)-> Pipeline:
        if not isinstance(X_train,pd.DataFrame):
            raise TypeError("X_train data must be a pandas DataFrame.")
        if not isinstance(y_train,pd.Series):
            raise TypeError("y_train data must be a pandas Series.")
        
        if X_train.empty:
            raise ValueError("X_train must not be empty")
        if y_train.empty:
            raise ValueError("y_train must not be empty")
        
        if len(X_train) != len(y_train):
            raise ValueError(
                "X_train and y_train must contain the same number of samples"
            )
        if X_train.shape[1] == 0:
            raise ValueError("X_train must contain at least one feature")

        logging.info("Building a linear regression model")
        pipeline = Pipeline(
            [
                ("scaler",StandardScaler()),
                ("model",LinearRegression()),
            ]
        )
        logging.info("Training a linear regression model.")
        pipeline.fit(X_train,y_train)
        logging.info("Model Training Completed.")
        return pipeline
    
class ModelBuilder:
    def __init__(self, strategy: ModelBuildingStrategy):
        self.strategy = strategy
    
    @property
    def strategy(self) -> ModelBuildingStrategy:
        return self._strategy

    @strategy.setter
    def strategy(self, strategy: ModelBuildingStrategy) -> None:
        if not isinstance(strategy, ModelBuildingStrategy):
            raise TypeError(f"Expected a ModelBuildingStrategy, got {type(strategy)}")
        logging.info(f"Strategy updated to: {strategy.__class__.__name__}")
        self._strategy = strategy

    def set_strategy(self, strategy: ModelBuildingStrategy)-> None:
        self.strategy = strategy
    
    def build_model(self, X_train: pd.DataFrame, y_train: pd.Series)-> RegressorMixin:
        logging.info("Building and training the model using the selected strategy.")
        return self._strategy.build_and_train_model(X_train,y_train)