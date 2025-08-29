from abc import ABC, abstractmethod
from sklearn.preprocessing import StandardScaler
import logging

class ScalingStrategy():
    @abstractmethod
    def fit_transform(self, data):
        pass

    @abstractmethod
    def transform(self, data):
        pass
    
class StandardScalerStrategy(ScalingStrategy):
    def __init__(self):
        """
        Initializes the StandardScalerStrategy with the given column to scale.

        Args:
            column (str): The column of the DataFrame to scale.
        """
        self.scaler = StandardScaler()
        logging.info(f"StandardScaler initialized for data")

    def fit_transform(self, data):
        """
        Fits the scaler to the data and transforms the specified column.
        The scaler is fit to the data and then the specified column is transformed using the scaler.
        The transformed data is returned as a new DataFrame.
        """
        
        try:
            data = self.scaler.fit_transform(data)
            logging.info(f"StandardScaler fit and transformed on column: {data}")
            return data
        except Exception as e:
            logging.error(f"Error in fit_transform: {e}")
            raise

    def transform(self, data):
        """
        Transforms the specified column using the scaler.

        Args:
            data (pd.DataFrame): The DataFrame containing the column to transform.

        Returns:
            pd.DataFrame: The transformed DataFrame.
        """
        try:
            data = self.scaler.transform(data)
            logging.info(f"StandardScaler transformed on column: {data}")
            return data
        except Exception as e:
            logging.error(f"Error in transform: {e}")
            raise