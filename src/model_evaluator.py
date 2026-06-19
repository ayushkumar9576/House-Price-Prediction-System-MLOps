import logging
from abc import ABC, abstractmethod
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.base import RegressorMixin

logger = logging.getLogger(__name__)

class ModelEvalutingStrategy(ABC):
    @abstractmethod
    def evaluate_model(self, model: RegressorMixin, X_test, y_test: pd.Series)-> dict:
        pass

class RegressionModelEvaluationStrategy(ModelEvalutingStrategy):
    def evaluate_model(self, model: RegressorMixin, X_test, y_test: pd.Series)-> dict:
        
        logger.info("Predicting using the trained model")
        y_pred = model.predict(X_test)

        logger.info("Calculating evaluating matrix")
        mse = mean_squared_error(y_test,y_pred)
        r2 = r2_score(y_test,y_pred)

        metrices = {"Mean Squared Error":mse, "R-Squared":r2}

        logger.info("Evaluating metrics calculated.")
        return metrices


class ModelEvaluator:
    def __init__(self, strategy: ModelEvalutingStrategy):
        self.strategy = strategy

    @property
    def strategy(self) -> ModelEvalutingStrategy:
        return self._strategy

    @strategy.setter
    def strategy(self, strategy: ModelEvalutingStrategy) -> None:
        if not isinstance(strategy, ModelEvalutingStrategy):
            raise TypeError(f"Expected a ModelEvaluatingStrategy, got {type(strategy)}")
        logger.info(f"Strategy updated to: {strategy.__class__.__name__}")
        self._strategy = strategy


    def set_strategy(self, strategy: ModelEvalutingStrategy)-> None:
        logger.info("Applying strategy for model evaluation.")
        self.strategy = strategy
    
    def evaluate(self,model: RegressorMixin, X_test, y_test: pd.Series)->dict:
        logger.info("Evaluating the model with specified strategy.")
        return self._strategy.evaluate_model(model,X_test,y_test)
        