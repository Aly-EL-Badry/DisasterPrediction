from abc import ABC, abstractmethod
import pandas as pd
from typing import Any
import pickle

class Model(ABC):
    """
    Abstract base class for all ML models.
    Defines a standard interface for training, predicting, evaluating, saving, and loading models.
    """
    def __init__(self):
        """
        Initialize the model object.

        Sets the internal model object to None.
        """
        self.model = None

    @abstractmethod
    def train(self, X_train: pd.DataFrame, y_train: pd.Series) -> None:
        """
        Train the model on the given data.
        """
        pass


    @abstractmethod
    def predict(self, X: pd.DataFrame) -> Any:
        """
        Make predictions using the trained model.
        """
        pass

    def save(self, path: str) -> None:
        """
        Save the trained model to a file.
        """
        if self.model is None:
            raise ValueError("No model to save. Train a model first.")
        pickle.dump(self.model, open(path, "wb"))
        print(f"Model saved at {path}")