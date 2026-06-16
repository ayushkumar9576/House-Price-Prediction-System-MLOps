import logging
from abc import ABC, abstractmethod
import pandas as pd
from sklearn.model_selection import train_test_split

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

class DataSplittingTechnique(ABC):
    @abstractmethod
    def split_data(self, df:pd.DataFrame, target_col: str)-> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        pass

class SimpleTrainTestSplit(DataSplittingTechnique):
    def __init__(self, test_size=0.2, random_state=42):
        self.test_size = test_size
        self.random_state = random_state
    
    def split_data(self, df: pd.DataFrame, target_col: str)-> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        if target_col not in df.columns:
            raise ValueError(f"Target column '{target_col}' not found.")
        logging.info("Performing Train-Test Splitting on the data.")
        X = df.drop(columns=[target_col])
        y = df[target_col]
        X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=self.test_size, random_state=self.random_state)
        logging.info("Train-Test Split Completed.")
        return X_train, X_test, y_train, y_test

class DataSplitter:
    def __init__(self, strategy: DataSplittingTechnique):
        self.strategy = strategy
    
    def set_strategy(self, strategy: DataSplittingTechnique):
        logging.info("Changing the Splitting strategy.")
        self.strategy = strategy

    def split(self, df: pd.DataFrame, target_col: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        logging.info("Splitting data using the selected strategy.")
        return self.strategy.split_data(df,target_col)
        