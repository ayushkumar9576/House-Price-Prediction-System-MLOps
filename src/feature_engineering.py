"""
AVAILABLE STRATEGIES:
    1. LogTransformation   — Applies log(1+x) to reduce skewness in heavy-tailed features.
    2. StandardScaling     — Z-score normalization: centers data at 0, scales to unit variance.
    3. MinMaxScaling       — Scales features to a fixed range, default [0, 1].
    4. OneHotEncoding      — Converts categorical columns into binary indicator columns.
"""

import logging
from abc import ABC,abstractmethod
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, StandardScaler

logger = logging.getLogger(__name__)

class FeatureEngineeringStrategy(ABC):
    @abstractmethod
    def fit(self, df: pd.DataFrame)->None:
        pass
    
    @abstractmethod
    def transform(self,df: pd.DataFrame)->pd.DataFrame:
        pass
    
    def fit_transform(self,df: pd.DataFrame)-> pd.DataFrame:
        self.fit(df)
        return self.transform(df)
    
    @staticmethod
    def _validate_features(df: pd.DataFrame, features)->None:
        missing = set(features) - set(df.columns)
        if missing:
                raise KeyError(f"Features not found in dataframe: {sorted(missing)}")
    
class LogTransformation(FeatureEngineeringStrategy):
    def __init__(self,features):
        self.features = list(features)
        self._is_fitted = False
    
    def fit(self, df: pd.DataFrame)->None:
        self._validate_features(df,self.features)

        for feature in self.features:
            if(df[feature] <= -1).any():
                raise ValueError(f"Feature '{feature}' contains values <= -1. log1p cannot be applied.")
        self._is_fitted = True
    def transform(self, df: pd.DataFrame)->pd.DataFrame:
        if not self._is_fitted:
            raise RuntimeError("Log transformation has not been fitted yet")
        self._validate_features(df, self.features)
        
        logger.info(f"Applying log transformation to features: {self.features}")
        df_transformed = df.copy()
        for feature in self.features:
            df_transformed[feature] = np.log1p(df_transformed[feature])
        logger.info("Log transformation completed.")
        return df_transformed



class StandardScaling(FeatureEngineeringStrategy):
    def __init__(self, features):
        self.features = list(features)
        self.scaler = StandardScaler()
        self._is_fitted = False

    def fit(self, df: pd.DataFrame)->None:
        self._validate_features(df,self.features)
        
        logger.info(f"Fitting StandardScaler on features: {self.features}")
        self.scaler.fit(df[self.features])
        self._is_fitted = True

    def transform(self, df: pd.DataFrame)-> pd.DataFrame:
        if not self._is_fitted:
            raise RuntimeError("StandardScaler has not been fitted yet.")
        self._validate_features(df,self.features)
        
        logger.info(f"Applying standard scaling to features: {self.features}")
        df_transformed = df.copy()
        df_transformed[self.features] = self.scaler.transform(df[self.features])
        logger.info("Standard scaling completed.")
        return df_transformed

class MinMaxScaling(FeatureEngineeringStrategy):
    def __init__(self, features,feature_range = (0,1)):
        self.features = list(features)
        self.scaler = MinMaxScaler(feature_range=feature_range)
        self._is_fitted = False
    
    def fit(self, df: pd.DataFrame)->None:
        self._validate_features(df,self.features)
        
        logger.info(f"Fitting MinMaxScaler on features: {self.features}")
        self.scaler.fit(df[self.features])
        self._is_fitted = True

    def transform(self, df: pd.DataFrame)->pd.DataFrame:
        if not self._is_fitted:
            raise RuntimeError("MinMaxScaler has not been fitted yet.")
        self._validate_features(df,self.features)
        
        logger.info(f"Applying Min-Max scaling to features: {self.features} with range {self.scaler.feature_range}")
        df_transformed = df.copy()
        df_transformed[self.features] = self.scaler.transform(df[self.features])
        logger.info("Min-Max scaling completed.")
        return df_transformed
    
class OneHotEncoding(FeatureEngineeringStrategy):
    def __init__(self, features):
        self.features = list(features)
        self.encoder = OneHotEncoder(sparse_output=False, drop="first",handle_unknown="ignore")
        self._is_fitted = False

    def fit(self, df: pd.DataFrame)->None:
        self._validate_features(df,self.features)
        
        logger.info(f"Fitting OneHotEncoder on features: {self.features}")
        self.encoder.fit(df[self.features])
        self._is_fitted = True
    
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self._is_fitted:
            raise RuntimeError("OneHotEncoder has not been fitted")
        self._validate_features(df,self.features)
        
        logger.info(f"Applying one-hot encoding to features: {self.features}")
        encoded_array = self.encoder.transform(df[self.features])
        encoded_df = pd.DataFrame(encoded_array,columns=self.encoder.get_feature_names_out(self.features),index=df.index,)
        df_transformed = df.drop(columns=self.features)
        result = pd.concat([df_transformed, encoded_df],axis=1)
        logger.info("One-hot encoding completed.")

        return result

class FeatureEngineering:
    def __init__(self, strategy: FeatureEngineeringStrategy):
        self._strategy = strategy

    def set_strategy(self, strategy: FeatureEngineeringStrategy)->None:
        logger.info("Changing the Feature Engineering strategy.")
        self._strategy = strategy
    
    def fit(self, df: pd.DataFrame)->None:
        self._strategy.fit(df)
    
    def transform(self, df: pd.DataFrame)->pd.DataFrame:
        return self._strategy.transform(df)

    def fit_transform(self, df: pd.DataFrame)->pd.DataFrame:
        return self._strategy.fit_transform(df)
