import pandas as pd
from sklearn.preprocessing import LabelEncoder

class labelEncodingStrategy:
    def encode(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Encode categorical columns in the DataFrame using LabelEncoder.

        Args:
            data (pd.DataFrame): The DataFrame to encode.

        Returns:
            pd.DataFrame: The encoded DataFrame.
        """
        le = LabelEncoder()
        for col in data.columns:
            if data[col].dtype == 'object' or pd.api.types.is_categorical_dtype(data[col]):
                data[col] = le.fit_transform(data[col].astype(str))
        return data
