from abc import ABC, abstractmethod
from sklearn.preprocessing import StandardScaler
import logging

class ScalingStrategy():
    def __init__(self, column):
        self.columnToScale = column
    
    @abstractmethod
    def fit_transform(self, data):
        pass

    @abstractmethod
    def transform(self, data):
        pass
    
class StandardScalerStrategy(ScalingStrategy):
    def __init__(self, column):
        super().__init__(column)
        self.scaler = StandardScaler()
        logging.info(f"StandardScaler initialized for column: {self.columnToScale}")

    def fit_transform(self, data):
        try:
            data[self.columnToScale] = self.scaler.fit_transform(data[[self.columnToScale]])
            logging.info(f"StandardScaler fit and transformed on column: {self.columnToScale}")
            return data
        except Exception as e:
            logging.error(f"Error in fit_transform: {e}")
            raise

    def transform(self, data):
        try:
            data[self.columnToScale] = self.scaler.transform(data[[self.columnToScale]])
            logging.info(f"StandardScaler transformed on column: {self.columnToScale}")
            return data
        except Exception as e:
            logging.error(f"Error in transform: {e}")
            raise