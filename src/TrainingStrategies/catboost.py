from .base import Model
from catboost import CatBoostClassifier

class CatboostModel(Model):
    def __init__(self, **kwargs):
        """
        Initialize the catboost model.
        You can pass hyperparameters via kwargs.
        """
        self.model = CatBoostClassifier(random_state=42, **kwargs)

    def train(self, X_train, y_train):
        """
        Train the model on the provided training data.
        """
        self.model.fit(X_train, y_train)

    def predict(self, X):
        """
        Predict class labels for the given input data.
        """
        return self.model.predict(X)