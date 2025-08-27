from .base import DataStrategy
import logging

class TransformationStrategy(DataStrategy):
    def __init__(self, transform_function, column):
        """
        Initializes the TransformationStrategy with the given transformation function and column to transform.

        Args:
            transform_function (function): The function to apply to the column.
            column (str): The column of the DataFrame to transform.
        """
        self.transform_function = transform_function
        self.columnToTransform = column

    def handle_data(self, data):
        """
        Applies the given transformation function to the specified column.

        Args:
            data (pd.DataFrame): The DataFrame containing the column to transform.

        Returns:
            pd.Series: The transformed Series.

        Raises:
            Exception: If an error occurs while transforming the column.
        """

        try:
            return self.transform_function(data[self.columnToTransform])
        except Exception as e:
            logging.error(f"Error in TransformationStrategy: {e}")
            raise