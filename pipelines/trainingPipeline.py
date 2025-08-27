from zenml import pipeline
from steps.dataIngestion import data_ingestion_step
from steps.training import split_step, trainingCatBoost, trainingXGBoost, modelEvaluation, decodeData, smoteData
from pathlib import Path
import yaml
import os

with open("config/config.yaml", "r") as f:
    config = yaml.safe_load(f)

data_path = os.path.join(config['project']['root'], config['project']['output_path'])
ctbPath = os.path.join(config['project']['root'], config['project']['model_path'], "ctb-model.pkl")
xgbPath = os.path.join(config['project']['root'], config['project']['model_path'], "xgb-model.pkl")
@pipeline
def training_pipeline():
    data = data_ingestion_step(DATA_PATH=data_path)
    balanceData = smoteData(data)
    X_train, X_test, y_train, y_test = split_step(balanceData)
    cat_model = trainingCatBoost(X_train, y_train, ctbPath)
    y_train_decoded = decodeData(y_train)
    y_test_decoded = decodeData(y_test)
    xg_model = trainingXGBoost(X_train, y_train_decoded, xgbPath)
    cat_report, cat_cm = modelEvaluation(cat_model, X_test, y_test)
    xg_report, xg_cm = modelEvaluation(xg_model, X_test, y_test_decoded)