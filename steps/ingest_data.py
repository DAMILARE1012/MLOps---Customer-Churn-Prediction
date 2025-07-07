import pandas as pd
import numpy as np
import logging
from zenml import step

class IngestData:
    """
    Ingesting data to the workflow
    """
    def __init__(self, data_path: str)  -> None:
        """
        Args:
            data_path: path to the data file
        """
        self.data_path = data_path

    def get_data(self):
        """
        Get data from the data path
        """
        try:
            df = pd.read_csv(self.data_path)
            logging.info(f"Data from {self.data_path} successfully read with pandas")
            return df
        except Exception as e:
            logging.error(f"Error loading data from {self.data_path}: {e}")
            raise e
    
@step(enable_cache=False)
def ingest_df(data_path: str) -> pd.DataFrame:
    """
    Ingest data from the data path
    """
    try:
        ingest_data = IngestData(data_path)
        logging.info(f"Ingesting data completed from {data_path}")
        df = ingest_data.get_data()
        return df
    except Exception as e:
        logging.error(f"Error while ingesting data from {data_path}: {e}")
        raise e
    
