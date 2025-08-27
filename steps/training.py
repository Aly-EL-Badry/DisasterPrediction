import pandas as pd
from zenml import step
from sklearn.model_selection import train_test_split
from typing import Tuple, Any
import numpy as np
from imblearn.combine import SMOTEENN
from src.TrainingStrategies.catboost import CatboostModel
from src.TrainingStrategies.xgboost import XGBoostModel
from src.modelEvaluate.classificationReport import classificationReportEvaluation
from src.modelEvaluate.confusionMatrix import confusionMatrixEvaluation
from src.dataDecode import labelEncodingStrategy
import logging
import mlflow
import os 


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

@step
def split_step(
    data: pd.DataFrame,
    target: str = "weather",
    test_size: float = 0.2,
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Splits a given dataset into training and test sets using train_test_split.
    
    Args:
        data (pd.DataFrame): The dataset to split.
        target (str, optional): The target column name. Defaults to "weather".
        test_size (float, optional): The proportion of the dataset to include in the test split.
            Defaults to 0.2.
        random_state (int, optional): Controls the shuffling applied to the data before applying the split.
            Defaults to 42.
    
    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]: A tuple containing the training data, test data, 
        training labels, and test labels.
    """
    try:
        logger.info(f"Splitting data with test_size={test_size}")
        X = data.drop(columns=[target])
        y = data[target]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        logger.info(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")
        return X_train, X_test, y_train, y_test
    except Exception as e:
        logger.error(f"Error in data splitting: {e}")
        raise



@step
def trainingCatBoost(x_train : pd.DataFrame, x_test:pd.DataFrame, path : str) -> CatboostModel:
    """
    Step to train a CatBoost model on given data and save it to the given path.

    Args:
        x_train: The training data
        x_test: The test data
        path: The path to save the model

    Returns:
        The trained CatBoost model
    """
    try:
        model = CatboostModel()

        model.train(x_train, x_test)

        mlflow.log_params(model.model.get_params())
        mlflow.catboost.log_model(model.model, "models/catboost")


    
        model.save(path=path)
        return model
    except Exception as e:
        logger.error(f"Error in training model: {e}")
        raise


@step
def trainingXGBoost(x_train : pd.DataFrame, x_test:pd.DataFrame, path : str) -> XGBoostModel:
    """
    Step to train a XGBoost model on given data and save it to the given path.

    Args:
        x_train: The training data
        x_test: The test data
        path: The path to save the model

    Returns:
        The trained XGBoost model
    """
    try:
        model = XGBoostModel()
        model.train(x_train, x_test)

        mlflow.log_param("random_state", model.model.get_params()["random_state"])
        mlflow.xgboost.log_model(model.model, "models/xgboost")
        model.save(path=path)
        return model
    except Exception as e:
        logger.error(f"Error in training model: {e}")
        raise


@step
def modelEvaluation(model : Any, x_test:pd.DataFrame, y_test:pd.Series, run_id) -> Tuple[str, np.ndarray]:
    """
    Step to evaluate a model on given test data and save the classification report and confusion matrix to the given path.

    Args:
        model: The model to evaluate
        x_test: The test data
        y_test: The test labels
        run_id: The MLflow run ID to log the artifacts

    Returns:
        A tuple of the classification report as a string and the confusion matrix as a numpy array
    """
    try:
        y_pred = model.predict(x_test)

        cr = classificationReportEvaluation().evaluate(y_test, y_pred)
        cm = confusionMatrixEvaluation().evaluate(y_test, y_pred)

        os.makedirs("logs/classificationReports", exist_ok=True)
        os.makedirs("logs/confusionMatrix", exist_ok=True)

        with open(f"logs/classificationReports/cr_{run_id}.txt", "w") as f:
            f.write(cr)

        with open(f"logs/confusionMatrix/cm_{run_id}.txt", "w") as f:
            f.write(str(cm))

        mlflow.log_artifact(f"logs/classificationReports/cr_{run_id}.txt")
        mlflow.log_artifact(f"logs/confusionMatrix/cm_{run_id}.txt")

        return cr, cm

    except Exception as e:
        logger.error(f"Error in model evaluation: {e}")
        raise

@step
def smoteData(data: pd.DataFrame, target: str = "weather") -> pd.DataFrame:
    """
    Step to balance data using SMOTEENN (combination of SMOTE and ENN).

    Args:
        data (pd.DataFrame): Input dataset with features + target.
        target (str): Target column name.

    Returns:
        pd.DataFrame: Balanced dataset after resampling.
    """
    try:
        logger.info("Applying SMOTEENN for data balancing...")

        X = data.drop(columns=[target])
        y = data[target]

        smote_enn = SMOTEENN(
            sampling_strategy="auto",  
            random_state=42,
            n_jobs=-1
        )
        X_res, y_res = smote_enn.fit_resample(X, y)

        balanced_data = pd.DataFrame(X_res, columns=X.columns)
        balanced_data[target] = y_res

        logger.info(f"Data balanced: Original size={len(data)}, New size={len(balanced_data)}")
        return balanced_data

    except Exception as e:
        logger.error(f"Error in SMOTEENN step: {e}")
        raise

@step
def decodeData(Data : pd.DataFrame) -> pd.DataFrame:
    """
    Step to decode categorical columns in the DataFrame using LabelEncoder.

    Args:
        Data (pd.DataFrame): Input DataFrame with features.

    Returns:
        pd.DataFrame: Decoded DataFrame.
    """

    try:
        return labelEncodingStrategy().encode(Data)
    except Exception as e:
        logger.error(f"Error in data decoding: {e}")
        raise