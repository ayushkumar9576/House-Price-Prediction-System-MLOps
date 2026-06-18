"""
AVAILABLE STRATEGIES:
    1. LinearRegressionStrategy — Creates a Pipeline with StandardScaler + LinearRegression.
       (More strategies like RandomForest, XGBoost, etc. can be added as new subclasses.)
"""

import logging
from abc import ABC, abstractmethod
from typing import Any
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
        if not isinstance(X_train, pd.DataFrame):
            raise TypeError("X_train data must be a panda DataFrame.")
        if not isinstance(y_train,pd.Series):
            raise TypeError("y_train data must be a panda Series")
        
        logging.info("Building a linear regression model.")
        pipeline = Pipeline(
            [
                ("scaler",StandardScaler()), # Feature Scaling
                ("model",LinearRegression()), #Linear Regression Model
            ]
        )
        logging.info("Training a linear regression model")
        pipeline.fit(X_train,y_train)
        logging.info("Model Training Completed")
        return pipeline
    
class ModelBuilder:
    def __init__(self, strategy: ModelBuildingStrategy):
        self._strategy = strategy
    
    def set_strategy(self, strategy: ModelBuildingStrategy):
        self._strategy = strategy
    
    def build_model(self, X_train: pd.DataFrame, y_train: pd.Series)-> RegressorMixin:
        logging.info("Building and training the model using the selected strategy")
        return self._strategy.build_and_train_model(X_train,y_train)