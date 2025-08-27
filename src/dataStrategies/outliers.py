from .base import DataStrategy
import logging

class OutliersHandlingStrategy(DataStrategy):
    def __init__(self, column):
        """
        Initializes the OutliersHandlingStrategy with the given column to clean.

        Args:
            column (str): The column of the DataFrame to clean from outliers.
        """

        super().__init__()
        self.columnToClean = column
        

class removingOutliersStrategy(OutliersHandlingStrategy):
    def handle_data(self, data):
        """
        Handles outliers by removing data points outside of the IQR range.

        Removes all data points that are outside of the range defined by the lower and upper bounds
        calculated using the Interquartile Range (IQR).

        Args:
            data (pd.DataFrame): The DataFrame containing the column to clean.

        Returns:
            pd.DataFrame: The cleaned DataFrame with outliers removed.
        """
        
        try:

            logging.info("Handling outliers...")
            Q1 = data[self.columnToClean].quantile(0.25)
            Q3 = data[self.columnToClean].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            data = data[(data[self.columnToClean] >= lower_bound) & (data[self.columnToClean] <= upper_bound)]
            return data
        
        except Exception as e:
            logging.error(f"Error handling outliers: {e}")
            raise

class cappingOutliersStrategy(OutliersHandlingStrategy):
    def handle_data(self, data):
        """
        Handles outliers by capping data points outside of the IQR range.

        Caps all data points that are outside of the range defined by the lower and upper bounds
        calculated using the Interquartile Range (IQR).

        Args:
            data (pd.DataFrame): The DataFrame containing the column to clean.

        Returns:
            pd.DataFrame: The cleaned DataFrame with outliers capped.
        """
        
        try:
            logging.info("Handling outliers...")

            data = data.copy()

            Q1 = data[self.columnToClean].quantile(0.25)
            Q3 = data[self.columnToClean].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR

            data.loc[:, self.columnToClean] = data[self.columnToClean].apply(
                lambda x: lower_bound if x < lower_bound else upper_bound if x > upper_bound else x
            )

            return data

        except Exception as e:
            logging.error(f"Error handling outliers: {e}")
            raise
