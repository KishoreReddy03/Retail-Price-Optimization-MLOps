from abc import ABC, abstractmethod
from typing import List

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder

from steps.src.data_loader import DataLoader


class CategoricalEncoder:
    """
    This class applies encoding to categorical variables. 
    
    Parameters
    ----------
    method: str, default="onehot"
        The method to encode the categorical variables. Can be "onehot" or "ordinal".
    
    categories: 'auto' or a list of lists, default='auto'
        Categories for the encoders. Must match the number of columns. If 'auto', categories are determined from data.
    """
    def __init__(self, method="onehot", categories='auto'):
        self.method = method
        self.categories = categories
        self.encoders = {}
        
    def fit(self, df, columns):
        """
        This function fits the encoding method to the provided data.
        
        Parameters
        ----------
        df: pandas DataFrame
            The input data to fit.
            
        columns: list of str
            The names of the columns to encode.
        """
        for col in columns:
            if self.method == "onehot":
                self.encoders[col] = OneHotEncoder(sparse_output=False, categories=self.categories)

            elif self.method == "ordinal":
                self.encoders[col] = OrdinalEncoder(categories=self.categories)
            else:
                raise ValueError(f"Invalid method: {self.method}")
            # Encoders expect 2D input data
            self.encoders[col].fit(df[[col]])
    def transform(self, df, columns):
        """
        This function applies the encoding to the provided data.
        
        Parameters
        ----------
        df: pandas DataFrame
            The input data to transform.
            
        columns: list of str
            The names of the columns to encode.
        """
        df_encoded = df.copy()
        for col in columns:
            # Encoders expect 2D input data
            transformed = self.encoders[col].transform(df[[col]])
            if self.method == "onehot":
                # OneHotEncoder returns a 2D array, we need to convert it to a DataFrame
                transformed = pd.DataFrame(transformed, columns=self.encoders[col].get_feature_names_out([col]))
                df_encoded = pd.concat([df_encoded.drop(columns=[col]), transformed], axis=1)
            else:
                df_encoded[col] = transformed
        return df_encoded
    def fit_transform(self, df, columns):
        """
        This function fits the encoding method to the provided data and then transforms the data.
        
        Parameters
        ----------
        df: pandas DataFrame
            The input data to fit and transform.
            
        columns: list of str
            The names of the columns to encode.
        """
        self.fit(df, columns)
        return self.transform(df, columns)
class OutlierHandler:
    """
    A class used to handle outliers in a pandas DataFrame.

    ...

    Attributes
    ----------
    multiplier : float
        The multiplier for the IQR. Observations outside of the range [Q1 - multiplier*IQR, Q3 + multiplier*IQR] are considered outliers.

    Methods
    -------
    fit(df: pd.DataFrame, columns: List[str])
        Compute the median and IQR for each specified column in the DataFrame.
    transform(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame
        Replace outliers in the specified columns with the respective column's median.
    fit_transform(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame
        Fit the OutlierHandler to the DataFrame and then transform the DataFrame.
    """

    def __init__(self, multiplier: float = 1.5):
        """
        Constructs all the necessary attributes for the OutlierHandler object.

        Parameters
        ----------
            multiplier : float
                The multiplier for the IQR. Observations outside of the range [Q1 - multiplier*IQR, Q3 + multiplier*IQR] are considered outliers.
        """
        self.multiplier = multiplier
        self.medians = {}
        self.iqr_bounds = {}
        self.outliers = pd.DataFrame()

    def fit(self, df: pd.DataFrame, columns: List[str]):
        """
        Compute the median and IQR for each specified column in the DataFrame.

        Parameters
        ----------
            df : pd.DataFrame
                The DataFrame to compute the median and IQR on.
            columns : List[str]
                The columns of the DataFrame to compute the median and IQR on.
        """
        for col in columns:
            self.medians[col] = df[col].median()
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            self.iqr_bounds[col] = (Q1 - self.multiplier * IQR, Q3 + self.multiplier * IQR)

    def transform(self, df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
        """
        Replace outliers in the specified columns with the respective column's median.

        Parameters
        ----------
            df : pd.DataFrame
                The DataFrame to replace outliers in.
            columns : List[str]
                The columns of the DataFrame to replace outliers in.

        Returns
        -------
            df : pd.DataFrame
                The DataFrame with outliers replaced.
        """
        for col in columns:
            outliers = df[(df[col] < self.iqr_bounds[col][0]) | (df[col] > self.iqr_bounds[col][1])]
            self.outliers = pd.concat([self.outliers, outliers])
            df[col] = np.where((df[col] < self.iqr_bounds[col][0]) | (df[col] > self.iqr_bounds[col][1]), self.medians[col], df[col])
        return df

    def fit_transform(self, df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
        """
        Fit the OutlierHandler to the DataFrame and then transform the DataFrame.

        Parameters
        ----------
            df : pd.DataFrame
                The DataFrame to fit the OutlierHandler on and replace outliers in.
            columns : List[str]
                The columns of the DataFrame to fit the OutlierHandler on and replace outliers in.

        Returns
        -------
            df : pd.DataFrame
                The DataFrame with outliers replaced.
        """
        self.fit(df, columns)
        return self.transform(df, columns)