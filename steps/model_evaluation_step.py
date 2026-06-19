"""
PURPOSE:
    Runs the trained model's preprocessor on test data, then evaluates predictions
    using MSE and R² metrics. Delegates metric computation to src/model_evaluator.py.
"""

import logging
from typing import Tuple
import pandas as pd
from sklearn.pipeline import Pipeline
from zenml import step
from src.model_evaluator import ModelEvaluator, RegressionModelEvaluationStrategy

logger = logging.getLogger(__name__)

@step(enable_cache=False)
def model_evaluator_step(trained_model: Pipeline, X_test: pd.DataFrame, y_test: pd.Series)-> Tuple[dict, float|None]:
    logger.info("Evaluating the trained model using ModelEvaluator and RegressionModelEvaluationStrategy")

    if not isinstance(X_test, pd.DataFrame):
        raise TypeError("X_test must be a pandas DataFrame.")
    if not isinstance(y_test, pd.Series):
        raise TypeError("y_test must be a pandas Series.")
    
    logger.info("Applying the same preproscessing to the test data")

    X_test_processed = trained_model.named_steps["preprocessor"].transform(X_test)

    evaluator = ModelEvaluator(strategy= RegressionModelEvaluationStrategy())
    evaluation_metrix = evaluator.evaluate(trained_model.named_steps["model"], X_test_processed, y_test)

    if not isinstance(evaluation_metrix,dict):
        raise ValueError("Evaluation metrics must be returned as a dictionary.")
    
    mse = evaluation_metrix.get("Mean Squared Error", None)

    return evaluation_metrix, mse

