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
    Step to split dataset into train/test sets.
    Encodes categorical target labels to numeric values.
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
    try:
        model = CatboostModel()
        model.train(x_train, x_test)
        model.save(path=path)
        return model
    except Exception as e:
        logger.error(f"Error in training model: {e}")
        raise
@step
def trainingXGBoost(x_train : pd.DataFrame, x_test:pd.DataFrame, path : str) -> XGBoostModel:
    try:
        model = XGBoostModel()
        model.train(x_train, x_test)
        model.save(path=path)
        return model
    except Exception as e:
        logger.error(f"Error in training model: {e}")
        raise


@step
def modelEvaluation(model : Any, x_test:pd.DataFrame, y_test:pd.Series) -> Tuple[str, np.ndarray]:
    try:
        y_predict = model.predict(x_test)
        return classificationReportEvaluation().evaluate(y_test, y_predict), confusionMatrixEvaluation().evaluate(y_test, y_predict)
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
    try:
        return labelEncodingStrategy().encode(Data)
    except Exception as e:
        logger.error(f"Error in data decoding: {e}")
        raise