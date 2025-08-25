from abc import ABC, abstractmethod

class DataStrategy(ABC):
    """
    Abstract base class for data handling strategies.
    
    """
    @abstractmethod
    def handle_data(self, data):
        pass

