import logging
import pandas as pd
from zenml import step
from pathlib import Path

@step(enable_cache=False)
def data_ingestion_step(DATA_PATH) -> pd.DataFrame:
    """
    Ingests the raw data from a CSV file and returns it as a pandas DataFrame.

    Returns:
        pd.DataFrame: The ingested data.
    """
    try:
        logging.info(f"Starting data ingestion from: {DATA_PATH}")
        data = pd.read_csv(DATA_PATH)
        logging.info(f"Data ingestion completed. Shape: {data.shape}")
        return data
    except FileNotFoundError:
        logging.error(f"CSV file not found at: {DATA_PATH}")
        raise
    except Exception as e:
        logging.error(f"Unexpected error in data ingestion: {e}")
        raise