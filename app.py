
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Dict, Any
import pandas as pd
from src.Deployment.modelService import ModelService
from src.Deployment.FeedbackStore import save_feedback
from pydantic import BaseModel
from typing import List, Dict, Any
import pandas as pd
import numpy as np
import logging

from src.dataStrategies.cleaning import DropColumnsStrategy, DropDuplicatesStrategy
from src.dataStrategies.outliers import cappingOutliersStrategy, removingOutliersStrategy
from src.dataStrategies.Transformation import TransformationStrategy
from src.dataStrategies.Scalling import StandardScalerStrategy

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

app = FastAPI(title="ML Model API with Feedback", version="1.0")
service = ModelService("Artifacts/ctb-model.pkl")

class FeedbackRequest(BaseModel):
    features: Dict[str, Any]
    prediction: float
    actual: float


class PredictionRequest(BaseModel):
    features: List[Dict[str, Any]]

# -------------------------------
# Preprocessing Helpers
# -------------------------------
def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """Apply same preprocessing pipeline used in training."""
    try:
        logger.info("Starting preprocessing pipeline...")

        drop_cols = ["date"]  
        df = DropColumnsStrategy(columns_to_drop=drop_cols).handle_data(df)
        df = DropDuplicatesStrategy().handle_data(df)

        remove_cols = [] 
        cap_cols = []     
        for col in remove_cols:
            df = removingOutliersStrategy(col).handle_data(df)
        for col in cap_cols:
            df = cappingOutliersStrategy(col).handle_data(df)

        if "temp_max" not in df.columns or "temp_min" not in df.columns:
            raise ValueError("Missing required columns: temp_max or temp_min")
        df["temp_diff"] = df["temp_max"] - df["temp_min"]
        df["temp_avg"] = (df["temp_max"] + df["temp_min"]) / 2

        transformations = []  # Add your transformations here
        for transform in transformations:
            column = transform["column"]
            method = transform["method"]
            func = getattr(np, method)
            df[column] = TransformationStrategy(func, column).handle_data(df)

        columns_to_scale = ["precipitation", "temp_max", "temp_min", "wind", "temp_diff", "temp_avg"]
        for col in columns_to_scale:
            df = StandardScalerStrategy(col).fit_transform(df)

        logger.info("Preprocessing pipeline completed.")
        return df
    except Exception as e:
        logger.error(f"Error in preprocessing: {e}")
        raise

# -------------------------------
# Prediction Endpoint
# -------------------------------
@app.post("/predict")
def predict(request: PredictionRequest):
    try:
        df = pd.DataFrame(request.features)

        required_cols = ["date", "precipitation", "temp_max", "temp_min", "wind"]
        for col in required_cols:
            if col not in df.columns:
                return {"error": f"Missing required column: {col}"}

        df = preprocess_data(df)

        preds = service.predict(df)
        return {"predictions": preds.tolist()}
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        return {"error": str(e)}


@app.post("/feedback")
def feedback(request: FeedbackRequest):
    save_feedback(request.features, request.prediction, request.actual)
    return {"message": "Feedback saved successfully"}

@app.get("/")
def root():
    return {"message": "API is running"}
