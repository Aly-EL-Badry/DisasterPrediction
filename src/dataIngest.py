import logging
import pandas as pd


class dataIngest:
    def __init__(self, file_path: str):
        self.file_path = file_path
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    def load_data(self) -> pd.DataFrame:
        try:
            data = pd.read_csv(self.file_path)
            logging.info(f"Data loaded successfully from {self.file_path}")
            return data
        except Exception as e:
            logging.error(f"Error loading data from {self.file_path}: {e}")
            raise