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

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

class OutlierDetectionStrategy(ABC):
    @abstractmethod
    def detect_outliers(self, df: pd.DataFrame)->pd.DataFrame:
        pass

class ZScoreOutlierDetection(OutlierDetectionStrategy):
    def __init__(self, threshold = 3):
        self.threshold = threshold
    
    def detect_outliers(self, df: pd.DataFrame)-> pd.DataFrame:
        logging.info("Detecting outlier using z-score method.")
        z_score = np.abs((df-df.mean())/df.std())
        outlier = z_score > self.threshold
        logging.info(f"Outlier detected using z-score threshold: {self.threshold}")
        return outlier

class IQROutlierDetection(OutlierDetectionStrategy):
    def detect_outliers(self, df: pd.DataFrame)-> pd.DataFrame:
        logging.info("Detecting outlier using IQR method")
        Q1 = df.quantile(0.25)
        Q3 = df.quantile(0.75)
        IQR = Q3-Q1
        outlier = (df<(Q1-1.5*IQR)) | (df>(Q3+1.5*IQR))
        logging.info("Outlier detected using the IQR method.")
        return outlier
    
class OutlierDetector:
    def __init__(self, strategy: OutlierDetectionStrategy):
        self._strategy = strategy
    
    def set_strategy(self, strategy: OutlierDetectionStrategy):
        logging.info("Applying startegy for outlier detection")
        self._strategy = strategy
    
    def detect_outliers(self, df: pd.DataFrame)-> pd.DataFrame:
        logging.info("Executing outlier detection strategy")
        return self._strategy.detect_outliers(df)
    
    def handle_outlier(self, df: pd.DataFrame, method="remove", **kwargs)-> pd.DataFrame:
        outlier =  self.detect_outliers(df)
        if method == "remove":
            logging.info("Removing outlier from the dataset.")
            df_cleaned = df[(~outlier).all(axis = 1)]
        elif method == "cap":
            logging.info("Capping outliers in the dataset.")
            df_cleaned = df.clip(lower=df.quantile(0.01), upper=df.quantile(0.99), axis=1)
        else:
            logging.warning(f"Unknown method '{method}'. No outlier handling performed.")
            return df
        
        logging.info("Outlier handling completed")
        return df_cleaned
    
    def visalize_outlier(self, df: pd.DataFrame, features: list):
        logging.info(f"Visualizing outliers for features: {features}")
        for f in features:
            plt.figure(figsize=(10,6))
            sns.boxplot(x=df[f])
            plt.title(f"Boxplot of {f}")
            plt.show()
        logging.info("Outlier visualization completed.")

