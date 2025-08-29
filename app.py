
from fastapi import FastAPI
import pickle
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any
import pandas as pd
from src.Deployment.modelService import ModelService
from src.Deployment.FeedbackStore import save_feedback
from pydantic import BaseModel
from typing import List, Dict, Any
import numpy as np
import logging
    

logging.basicConfig(level=logging.INFO)

app = FastAPI(title="ML Model API with Feedback", version="1.0")
service = ModelService("Artifacts/ctb-model.pkl")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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
        logging.info("Starting preprocessing pipeline...")
        logging.info(f"Original data shape: {df}")
        df.drop(columns=["date"], inplace=True)

        if "temp_max" not in df.columns or "temp_min" not in df.columns:
            raise ValueError("Missing required columns: temp_max or temp_min")

        df["temp_diff"] = df["temp_max"] - df["temp_min"]
        df["temp_avg"] = (df["temp_max"] + df["temp_min"]) / 2
        logging.info(f"Preprocessing data shape: {df}")
        df["precipitation"] = np.log1p(df["precipitation"])
        df["wind"] = np.sqrt(df["wind"])
        logging.info(f"Preprocessing data shape: {df}")
        columns_to_scale = ["precipitation","temp_max", "temp_min","wind","temp_avg", "temp_diff"]
        
        
        with open("Artifacts/scaler.pkl", "rb") as f:
            scaler = pickle.load(f)

        df[columns_to_scale] = scaler.transform(df[columns_to_scale])
        
        logging.info(f"Preprocessed data shape: {df}")

        logging.info("Preprocessing completed.")
        return df
    except Exception as e:
        logging.error(f"Error in preprocessing: {e}")
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
        logging.error(f"Prediction error: {e}")
        return {"error": str(e)}


@app.post("/feedback")
def feedback(request: FeedbackRequest):
    save_feedback(request.features, request.prediction, request.actual)
    return {"message": "Feedback saved successfully"}

@app.get("/")
def root():
    return {"message": "API is running"}
