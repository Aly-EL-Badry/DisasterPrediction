from .base import DataStrategy
import logging

class TransformationStrategy(DataStrategy):
    """
        A strategy to apply transformations to the data.
    """
    def __init__(self, transform_function, column):
        self.transform_function = transform_function
        self.columnToTransform = column

    def handle_data(self, data):
        try:
            return self.transform_function(data[self.columnToTransform])
        except Exception as e:
            logging.error(f"Error in TransformationStrategy: {e}")
            raise