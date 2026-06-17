"""
STRATEGIES:
    1. DropMissingValuesStrategy   — Drops rows or columns that have too many NaNs.
    2. FillMissingValuesStrategy   — Fills NaNs using: mean, median, mode, or a constant value.
"""
import logging
from abc import ABC, abstractmethod

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

class MissingValueHandlingStrategy(ABC):
    @abstractmethod
    def handle(self,df: pd.DataFrame)-> pd.DataFrame:
        pass


class DropMissingValuesStrategy(MissingValueHandlingStrategy):
    def __init__(self, axis = 0, thresh = None):
        self.axis = axis
        self.thresh = thresh
    
    def handle(self, df: pd.DataFrame)->pd.DataFrame:
        logging.info(f"Dropping missing value with axis = {self.axis} and thresh = {self.thresh}")
        
        cleaned = df.dropna(    axis=self.axis, thresh=self.thresh)
        
        logging.info("Dropped Missing values")
        return cleaned
    

class FillMissingValueStrategy(MissingValueHandlingStrategy):
    def __init__(self, method = "mean", fill_value = None):
        self.method = method
        self.fill_value = fill_value
    
    def handle(self, df: pd.DataFrame)->pd.DataFrame:
        logging.info(f"Filling missing Values Using method {self.method}")

        cleaned = df.copy()
        
        if self.method=="mean":
            numeric_col = cleaned.select_dtypes(include="number").columns
            cleaned[numeric_col] = cleaned[numeric_col].fillna(df[numeric_col].mean())
        elif self.method=="median":
            numeric_col = cleaned.select_dtypes(include="number").columns
            cleaned[numeric_col] = cleaned[numeric_col].fillna(df[numeric_col].median())
        elif self.method=="mode":
            for c in cleaned.columns:
                mode_series = df[c].mode()
                if not mode_series.empty:
                    cleaned[c] = cleaned[c].fillna(df[c].mode().iloc[0])
        elif self.method=="constant":
            cleaned = cleaned.fillna(self.fill_value)
        else:
            logging.info(f"Unknown method {self.method}. No missing values handled")

        logging.info("Missing values Handled")
        return cleaned


class MissingValueHandler:
    def __init__(self, strategy: MissingValueHandlingStrategy)->None:
        self._strategy = strategy

    def set_strategy(self, strategy: MissingValueHandlingStrategy)->None:
        logging.info("Changing the Handling method.")
        
        self._strategy = strategy
    
    def handle_missing_value(self, df: pd.DataFrame)->pd.DataFrame:
        logging.info("Executing missing value handling strategy")
        
        return self._strategy.handle(df)