from typing import List, Tuple

import pandas as pd
from sklearn.model_selection import train_test_split


class DataSplitter:
    """A class used to split data into training and testing parts"""

    def __init__(self, df: pd.DataFrame, features: List[str], target: str, test_size: float = 0.2):
        """
        Args:
        df: pandas DataFrame, the entire dataset.
        features: list of str, the column names to be used as features.
        target: str, the column name to be used as target.
        test_size: float, proportion of the dataset to include in the test split.
        """
        self.df = df
        self.features = features
        self.target = target
        self.test_size = test_size

    def split(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """Splits data into training and testing parts.
        
        Returns:
        Tuple of pandas DataFrame and Series: (X_train, X_test, y_train, y_test)
        """ 
        # drop month_year and id columns 
        X = self.df[self.features]
        y = self.df[self.target]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=self.test_size, shuffle=False)
        return X_train, X_test, y_train, y_test