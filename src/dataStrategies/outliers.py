from .base import DataStrategy
import logging

class OutliersHandlingStrategy(DataStrategy):
    def __init__(self, column):
        super().__init__()
        self.columnToClean = column
        

class removingOutliersStrategy(OutliersHandlingStrategy):
    def handle_data(self, data):
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
    """
    A strategy to cap outliers in the dataset using the IQR method.
    """

    def handle_data(self, data):
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
