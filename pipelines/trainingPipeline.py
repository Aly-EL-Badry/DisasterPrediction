from zenml import pipeline
from steps.dataIngestion import data_ingestion_step
from steps.training import split_step, trainingCatBoost, trainingXGBoost, modelEvaluation, decodeData, smoteData
import logging
import yaml
import os
import mlflow

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Load config
with open("config/config.yaml", "r") as f:
    config = yaml.safe_load(f)

data_path = os.path.join(config['project']['root'], config['project']['output_path'])
ctbPath = os.path.join(config['project']['root'], config['project']['model_path'], "ctb-model.pkl")
xgbPath = os.path.join(config['project']['root'], config['project']['model_path'], "xgb-model.pkl")

@pipeline
def training_pipeline():

    """
    Defines a pipeline for training two models (CatBoost and XGBoost) on a dataset and logging the results to MLflow.

    The pipeline consists of the following steps:

    1. Data ingestion: loads the dataset from a specified path
    2. SMOTE: applies SMOTE oversampling to the dataset
    3. Split: splits the dataset into training and test sets
    4. CatBoost: trains a CatBoost model on the training set
    5. XGBoost: trains an XGBoost model on the training set
    6. Evaluation: evaluates both models on the test set and logs the results to MLflow

    The pipeline takes no arguments and returns no values.

    """
    
    with mlflow.start_run(run_name="training_pipeline_run", nested=True) as run:

        data = data_ingestion_step(DATA_PATH=data_path)
        mlflow.log_artifact(data_path, artifact_path="dataset")

        balanceData = smoteData(data)

        X_train, X_test, y_train, y_test = split_step(balanceData)

        cat_model = trainingCatBoost(X_train, y_train, ctbPath)
        mlflow.log_artifact(ctbPath, artifact_path="models/catboost")
        modelUrl = f"runs:/{run.info.run_id}/models/catboost"
        mlflow.register_model(modelUrl, "CatboostModel")

        y_train_decoded = decodeData(y_train)
        y_test_decoded = decodeData(y_test)
        xg_model = trainingXGBoost(X_train, y_train_decoded, xgbPath)
        mlflow.log_artifact(xgbPath, artifact_path="models/xgboost")
        modelUrl = f"runs:/{run.info.run_id}/models/xgboost"
        mlflow.register_model(modelUrl, "XGBoostModel")

        cat_report, cat_cm = modelEvaluation(cat_model, X_test, y_test, run_id='CatBoost_Training')

        
        xg_report, xg_cm = modelEvaluation(xg_model, X_test, y_test_decoded, run_id='XGBoost_Training')
        
        logger.info(f"MLflow run completed")
