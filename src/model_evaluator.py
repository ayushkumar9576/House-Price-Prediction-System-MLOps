import logging
from abc import ABC, abstractmethod
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
        if not isinstance(X_test, pd.DataFrame):
            raise TypeError("X_test must be a pandas DataFrame.")
        if not isinstance(y_test, pd.Series):
            raise TypeError("y_test must be a pandas Series.")
        if X_test.empty:
            raise ValueError("X_test must not be empty.")
        if y_test.empty:
            raise ValueError("y_test must not be empty.")
        if len(X_test) != len(y_test):
            raise ValueError("X_test and y_test must contain the same number of samples.")
        
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
        self.strategy = strategy

    @property
    def strategy(self) -> ModelEvalutingStrategy:                   # FIX 1
        return self._strategy

    @strategy.setter
    def strategy(self, strategy: ModelEvalutingStrategy) -> None:   # FIX 3
        if not isinstance(strategy, ModelEvalutingStrategy):
            raise TypeError(f"Expected a ModelEvaluatingStrategy, got {type(strategy)}")
        logging.info(f"Strategy updated to: {strategy.__class__.__name__}")
        self._strategy = strategy


    def set_strategy(self, strategy: ModelEvalutingStrategy)-> None:
        logging.info("Applying strategy for model evaluation.")
        self.strategy = strategy
    
    def evaluate(self,model: RegressorMixin, X_test: pd.DataFrame, y_test: pd.Series)->dict:
        logging.info("Evaluating the model with specified strategy.")
        return self._strategy.evaluate_model(model,X_test,y_test)
        