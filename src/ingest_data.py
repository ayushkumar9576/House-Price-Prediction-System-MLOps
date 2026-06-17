import os
import zipfile
from abc import ABC, abstractmethod
import pandas as pd 

class DataIngestor(ABC):
    @abstractmethod
    def ingest(self, file_path: str)-> pd.DataFrame:
        pass

class ZipDataIngestor(DataIngestor):
    def ingest(self, file_path: str)-> pd.DataFrame:
        if not file_path.endswith(".zip"):
            raise ValueError("The provided file is not a zip file")

        with zipfile.ZipFile(file_path,"r") as ref:
            ref.extractall("extracted_data")
        
        extracted_file = os.listdir("extracted_data")
        csv_file = [f for f in extracted_file if f.endswith(".csv")]

        if len(csv_file)==0:
            raise FileNotFoundError("No CSV file found.")
        if len(csv_file)>1:
            raise ValueError("Multiple CSV file found, need only 1")
        
        csv_file_path = os.path.join("extracted_data",csv_file[0])
        df = pd.read_csv(csv_file_path)

        return df
    
class DataIngestorFactory:
    @staticmethod
    def get_data_ingestor(file_extension: str)->DataIngestor:
        if file_extension.lower() == ".zip":
            return ZipDataIngestor()
        else:
            raise ValueError(f"No ingestor available for file type: {file_extension}")