import logging
from abc import ABC, abstractmethod
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.base import RegressorMixin

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

class ModelEvalutingStrategy(ABC):
    @abstractmethod
    def evaluate_model(self, model: RegressorMixin, X_test: pd.DataFrame, y_test: pd.Series)-> dict:
        pass

class RegressionModelEvaluationStrategy(ModelEvalutingStrategy):
    def evaluate_model(self, model: RegressorMixin, X_test: pd.DataFrame, y_test: pd.Series)-> dict:
        logging.info("Predicting using the trained model")
        y_pred = model.predict(X_test)

        logging.info("Calculating evaluating matrix")
        mse = mean_squared_error(y_test,y_pred)
        r2 = r2_score(y_test,y_pred)

        metrices = {"Mean Squared Error":mse, "R-Squared":r2}

        logging.info("Evaluating metrics calculated.")
        return metrices


class ModelEvaluator:
    def __init__(self, strategy: ModelEvalutingStrategy):
        self._strategy = strategy
    
    def set_strategy(self, strategy: ModelEvalutingStrategy):
        logging.info("Applying strategy for model evaluation.")
        self._strategy = strategy
    
    def evaluate(self,model: RegressorMixin, X_test: pd.DataFrame, y_test: pd.Series)->dict:
        logging.info("Evaluating the model with specified strategy.")
        return self._strategy.evaluate_model(model,X_test,y_test)
        