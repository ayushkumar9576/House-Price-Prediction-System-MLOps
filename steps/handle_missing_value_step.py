"""
PURPOSE:
    Fills or drops missing (NaN) values in the DataFrame.
    Delegates to src/handle_missing_value.py (Strategy pattern).
"""

from typing import Any
import logging
import pandas as pd
from zenml import step
from src.handle_missing_value import MissingValueHandler, FillMissingValueStrategy, DropMissingValuesStrategy

logger = logging.getLogger(__name__)

@step
def handle_missing_value_step(df: pd.DataFrame, strategy: str= "mean", axis: int = 0, thresh: int | None = None, fill_value: Any = None)-> pd.DataFrame:
    logger.info("Handling missing value with MissingValueHandler and selected strategy")

    if strategy == "drop":
        handler = MissingValueHandler(DropMissingValuesStrategy(axis= axis, thresh= thresh))
    elif strategy in ["mean", "median", "mode", "constant"]:
        handler = MissingValueHandler(FillMissingValueStrategy(method= strategy, fill_value= fill_value))
    else:
        raise ValueError(f"Unsupported missing value handling strategy: {strategy}")
    
    cleaned_df = handler.handle_missing_value(df)
    logger.info("missing value handling completed successfully")
    return cleaned_df