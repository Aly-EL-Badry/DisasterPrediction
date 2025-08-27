# app.py
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import pandas as pd
from src.Deployment.modelService import ModelService
from src.Deployment.FeedbackStore import save_feedback

service = ModelService("Artifacts/ctb-model.pkl")

class PredictionRequest(BaseModel):
    features: List[Dict[str, Any]]

class FeedbackRequest(BaseModel):
    features: Dict[str, Any]
    prediction: float
    actual: float

app = FastAPI(title="ML Model API with Feedback", version="1.0")

@app.post("/predict")
def predict(request: PredictionRequest):
    df = pd.DataFrame(request.features)

    required_cols = ["date", "precipitation", "temp_max", "temp_min", "wind"]
    for col in required_cols:
        if col not in df.columns:
            return {"error": f"Missing required column: {col}"}

    df["temp_diff"] = df["temp_max"] - df["temp_min"]
    df["temp_avg"] = (df["temp_max"] + df["temp_min"]) / 2

    if "date" in df.columns:
        df = df.drop(columns=["date"])

    preds = service.predict(df)

    return {"predictions": preds.tolist()}

@app.post("/feedback")
def feedback(request: FeedbackRequest):
    save_feedback(request.features, request.prediction, request.actual)
    return {"message": "Feedback saved successfully"}

@app.get("/")
def root():
    return {"message": "API is running"}
