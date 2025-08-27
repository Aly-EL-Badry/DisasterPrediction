import pandas as pd
import os

FEEDBACK_FILE = "data/feedback.csv"

def save_feedback(features: dict, prediction: float, actual: float):
    feedback = pd.DataFrame([{
        **features,
        "prediction": prediction,
        "actual": actual
    }])
    
    if not os.path.exists(FEEDBACK_FILE):
        feedback.to_csv(FEEDBACK_FILE, index=False)
    else:
        feedback.to_csv(FEEDBACK_FILE, mode="a", header=False, index=False)
