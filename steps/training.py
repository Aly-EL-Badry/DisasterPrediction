import pandas as pd
from zenml import step
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

@step
def split_step(
    data: pd.DataFrame,
    target: str = "weather",
    test_size: float = 0.2,
    random_state: int = 42
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
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
