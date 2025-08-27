from .base import DataStrategy
import logging

class DropColumnsStrategy(DataStrategy):
    """
        Strategy to drop specified columns from the DataFrame.
    """
    def __init__(self, columns_to_drop):
        """
        Initializes the DropColumnsStrategy with the given columns to drop.

        Args:
            columns_to_drop (list[str]): The list of columns to drop from the DataFrame.
        """

        self.columns_to_drop = columns_to_drop

    def handle_data(self, data):
        try: 
            logging.info(f"Dropping columns: {self.columns_to_drop}")
            return data.drop(columns=self.columns_to_drop)
        except Exception as e:
            logging.error(f"Error dropping columns {self.columns_to_drop}: {e}")
            raise

class DropDuplicatesStrategy(DataStrategy):
    def handle_data(self, data):
        """
        Drops duplicate rows from the given DataFrame.

        Args:
            data (pd.DataFrame): The DataFrame to drop duplicates from.

        Returns:
            pd.DataFrame: The DataFrame with duplicate rows dropped.

        Raises:
            Exception: If an error occurs while dropping duplicates.
        """
        try:
            logging.info("Dropping duplicate rows...")
            return data.drop_duplicates()
        except Exception as e:
            logging.error(f"Error dropping duplicates: {e}")
            raise
