import logging
import pandas as pd


class dataIngest:
    def __init__(self, file_path: str):
        """
        Initializes the dataIngest class with the given file path.

        Args:
            file_path (str): The path to the file from which to load the data.

        Returns:
            None

        Raises:
            None
        """
        self.file_path = file_path
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    def load_data(self) -> pd.DataFrame:
        """
        Loads the data from the given file path.

        Returns:
            pd.DataFrame: The loaded data.

        Raises:
            Exception: If an error occurs while loading the data.
        """
        try:
            data = pd.read_csv(self.file_path)
            logging.info(f"Data loaded successfully from {self.file_path}")
            return data
        except Exception as e:
            logging.error(f"Error loading data from {self.file_path}: {e}")
            raise