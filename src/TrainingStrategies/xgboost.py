from .base import Model
from xgboost import XGBClassifier

class XGBoostModel(Model):
    def __init__(self, **kwargs):
        """
        Initialize the XGBoost model.
        You can pass hyperparameters via kwargs.
        """
        self.model = XGBClassifier(random_state=42, verbosity=0, **kwargs)

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

