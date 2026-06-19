"""
AVAILABLE STRATEGIES:
    1. ZScoreOutlierDetection  — Flags rows where any value's Z-score exceeds a threshold (default: 3).
    2. IQROutlierDetection     — Flags rows outside 1.5× the interquartile range (Q1–Q3).
"""

import logging
from abc import ABC, abstractmethod
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

logger = logging.getLogger(__name__)

class OutlierDetectionStrategy(ABC):
    @abstractmethod
    def detect_outliers(self, df: pd.DataFrame)-> pd.DataFrame:
        pass

class ZScoreOutlierDetection(OutlierDetectionStrategy):
    def __init__(self, threshold: float = 3.0):
        if threshold <= 0:
            raise ValueError("Threshold must be positive.")
        self.threshold = threshold
    

    def detect_outliers(self, df: pd.DataFrame)-> pd.DataFrame:
        logger.info("Detecting outlier using z-score method.")
        numeric_df = df.select_dtypes(include=[np.number])

        if numeric_df.empty:
            raise ValueError("No numeric columns found.")
        
        std = numeric_df.std().replace(0, np.nan)
        z_score = np.abs((numeric_df - numeric_df.mean()) / std)

        outlier = z_score > self.threshold
        logger.info(f"Outlier detected using z-score threshold: {self.threshold}")
        return outlier


class IQROutlierDetection(OutlierDetectionStrategy):
    def detect_outliers(self, df: pd.DataFrame)-> pd.DataFrame:
        logger.info("Detecting outlier using IQR method")
        numeric_df = df.select_dtypes(include=[np.number])

        if numeric_df.empty:
            raise ValueError("No numeric columns found.")

        Q1 = numeric_df.quantile(0.25)
        Q3 = numeric_df.quantile(0.75)
        IQR = Q3-Q1
        outlier = (numeric_df<(Q1-1.5*IQR)) | (numeric_df>(Q3+1.5*IQR))
        logger.info("Outlier detected using the IQR method.")
        return outlier
    

class OutlierDetector:
    def __init__(self, strategy: OutlierDetectionStrategy):
        self.strategy = strategy
    
    @property
    def strategy(self) -> OutlierDetectionStrategy:
        return self._strategy

    @strategy.setter
    def strategy(self, strategy: OutlierDetectionStrategy):
        if not isinstance(strategy, OutlierDetectionStrategy):
            raise TypeError(f"Expected an OutlierDetectionStrategy, got {type(strategy)}")
        logger.info(f"Strategy updated to: {strategy.__class__.__name__}")
        self._strategy = strategy

    def set_strategy(self, strategy: OutlierDetectionStrategy)-> None:
        self.strategy = strategy
    
    def detect_outliers(self, df: pd.DataFrame)-> pd.DataFrame:
        logger.info("Executing outlier detection strategy")
        return self._strategy.detect_outliers(df)
    
    def handle_outlier(self, df: pd.DataFrame, method: str = "remove")-> pd.DataFrame:
        outlier =  self.detect_outliers(df)
        method = method.lower()
        if method == "remove":
            logger.info("Removing outlier from the dataset.")

            df_cleaned = df[(~outlier).all(axis=1)].copy()
            removed = len(df) - len(df_cleaned)

            logger.info(f"Removed {removed} outlier row(s). {len(df_cleaned)} row(s) remaining.")
        elif method == "cap":
            logger.info("Capping outliers in the dataset.")

            numeric_cols = df.select_dtypes(include=[np.number]).columns
            df_cleaned = df.copy()
            df_cleaned[numeric_cols] = df_cleaned[numeric_cols].clip(
                lower=df_cleaned[numeric_cols].quantile(0.01),
                upper=df_cleaned[numeric_cols].quantile(0.99),
                axis=0,  # align Series index (columns) with DataFrame columns
            )

            logger.info("Outliers capped at 1st and 99th percentiles.")
        else:
            logger.warning(f"Unknown method '{method}'. No outlier handling performed.")
            return df
        
        logger.info("Outlier handling completed")
        return df_cleaned
    
    def visualize_outliers(self, df: pd.DataFrame, features: list[str])-> None:
        missing = [f for f in features if f not in df.columns]
        if missing:
            raise ValueError(f"Features not found in DataFrame: {missing}")

        logger.info(f"Visualizing outliers for features: {features}")
        for f in features:
            if not pd.api.types.is_numeric_dtype(df[f]):
                raise ValueError(f"{f} is not a numeric column.")

            plt.figure(figsize=(10,6))
            sns.boxplot(x=df[f])
            plt.title(f"Boxplot of {f}")
            plt.show()
        logger.info("Outlier visualization completed.")

