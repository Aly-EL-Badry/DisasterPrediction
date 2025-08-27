from abc import ABC, abstractmethod

class DataStrategy(ABC):
    @abstractmethod
    def handle_data(self, data):
        """
        Handles the given data according to the strategy's purpose.

        Args:
            data (pd.DataFrame): The DataFrame to handle.

        Returns:
            pd.DataFrame: The handled DataFrame.

        Raises:
            Exception: If an error occurs while handling the data.
        """
        pass

