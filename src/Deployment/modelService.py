import pickle
import pandas as pd

class ModelService:
    def __init__(self, model_path: str):
        self.model = self._load_model(model_path)

    def _load_model(self, model_path: str):
        with open(model_path, "rb") as f:
            model = pickle.load(f)
        return model

    def predict(self, data: pd.DataFrame):

        return self.model.predict(data)
