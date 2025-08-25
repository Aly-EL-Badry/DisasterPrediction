import logging
from typing import List
import pandas as pd
from typing import Dict, Any
from sklearn.model_selection import train_test_split
import numpy as np
from zenml import step

from src.dataStrategies.cleaning import DropColumnsStrategy, DropDuplicatesStrategy
from src.dataStrategies.outliers import cappingOutliersStrategy, removingOutliersStrategy
from src.dataStrategies.Transformation import TransformationStrategy
from src.dataStrategies.Scalling import StandardScalerStrategy

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# -------------------------------
# Helper Functions
# -------------------------------
def add_temp_avg(data: pd.DataFrame) -> pd.DataFrame:
    if "temp_max" not in data.columns or "temp_min" not in data.columns:
        raise ValueError("Missing columns: temp_max or temp_min")
    data["temp_avg"] = (data["temp_max"] + data["temp_min"]) / 2
    return data

def add_temp_diff(data: pd.DataFrame) -> pd.DataFrame:
    if "temp_max" not in data.columns or "temp_min" not in data.columns:
        raise ValueError("Missing columns: temp_max or temp_min")
    data["temp_diff"] = data["temp_max"] - data["temp_min"]
    return data

# -------------------------------
# ZenML Steps
# -------------------------------

@step
def data_cleaning_step(data: pd.DataFrame, drop_cols: List[str]) -> pd.DataFrame:
    """Step to drop useless columns and duplicates."""
    try:
        logger.info(f"Starting data cleaning. Dropping: {drop_cols}")
        data = DropColumnsStrategy(columns_to_drop=drop_cols).handle_data(data)
        data = DropDuplicatesStrategy().handle_data(data)
        logger.info(f"Data cleaning completed. Shape: {data.shape}")
        return data 
    except Exception as e:
        logger.error(f"Error in data cleaning step: {e}")
        raise


@step
def outlier_handling_step(data: pd.DataFrame, remove_cols: List[str], cap_cols: List[str]) -> pd.DataFrame:
    """Step to remove and cap outliers."""
    try:
        logger.info(f"Starting outlier handling. Remove: {remove_cols}, Cap: {cap_cols}")
        for col in remove_cols:
            data = removingOutliersStrategy(col).handle_data(data)
        for col in cap_cols:
            data = cappingOutliersStrategy(col).handle_data(data)
        logger.info("Outlier handling completed.")
        return data
    except Exception as e:
        logger.error(f"Error in outlier handling step: {e}")
        raise

@step
def feature_engineering_step(data: pd.DataFrame, transformations: list) -> pd.DataFrame:
    """Step to add new features and transformations from config."""
    try:
        logger.info(f"Starting feature engineering with transformations: {transformations}")
        data = add_temp_avg(data)
        data = add_temp_diff(data)

        for transform in transformations:
            column = transform["column"]
            method = transform["method"]
            func = getattr(np, method) 
            data[column] = TransformationStrategy(func, column).handle_data(data)

        logger.info("Feature engineering completed.")
        return data
    except Exception as e:
        logger.error(f"Error in feature engineering: {e}")
        raise

@step
def scaling_step(data: pd.DataFrame, columns_to_scale: List[str]) -> pd.DataFrame:
    """Step to scale specified columns."""
    try:
        logger.info(f"Starting scaling step for columns: {columns_to_scale}")
        for col in columns_to_scale:
            data = StandardScalerStrategy(col).fit_transform(data)
        logger.info("Scaling completed.")
        return data
    except Exception as e:
        logger.error(f"Error in scaling step: {e}")
        raise

@step
def split_step(
    data: pd.DataFrame,
    target: str = "weather",
    test_size: float = 0.2,
    random_state: int = 42
) -> Dict[str, Any]:
    """
    Step to split dataset into train/test sets.
    """
    try:
        logger.info(f"Splitting data with test_size={test_size}")
        X = data.drop(columns=[target])
        y = data[target]
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        logger.info(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")
        return {
            "X_train": X_train,
            "X_test": X_test,
            "y_train": y_train,
            "y_test": y_test
        }
    except Exception as e:
        logger.error(f"Error in data splitting: {e}")
        raise
