"""
PURPOSE:
    This is a ZenML pipeline step that loads the Ames Housing dataset from a ZIP file.
    It delegates the actual extraction work to src/ingest_data.py (Factory pattern).
"""

import logging
import pandas as pd
from zenml import step
from src.ingest_data import DataIngestorFactory

logger = logging.getLogger(__name__)

@step
def data_ingestor_step(file_path: str)-> pd.DataFrame:
    logger.info("Ingestion data from a ZIP file using the DataIngestor")
    logger.info("starting data ingestion from : %s",file_path)

    file_extension = ".zip"
    data_ingestor = DataIngestorFactory.get_data_ingestor(file_extension)

    df = data_ingestor.ingest(file_path)

    logger.info("Data ingestor completed. Loaded %d row and %d cols.",len(df), len(df.columns))
    return df