import time

import pandas as pd
from sqlalchemy import create_engine, exc


class DataLoader:
    """
    DataLoader class encapsulates the details of connecting to a database 
    and loading data into a pandas DataFrame.
    """

    def __init__(self, db_uri: str):
        """
        Initializes DataLoader class with a database URI and creates an SQLAlchemy engine.
        
        Args:
            db_uri (str): The database URI.
        """
        self.db_uri = db_uri
        self.engine = create_engine(self.db_uri)
        self.data = None
    def load_data(self, table_name: str) -> pd.DataFrame:
        """
        Loads data from the specified table into a DataFrame, which is stored 
        as an instance variable self.data.

        Args:
            table_name (str): The name of the table from which to load data.

        Returns:
            pd.DataFrame: The loaded data.

        Raises:
            ValueError: If the table does not exist or the query execution fails.
        """
        query = "SELECT * FROM " + table_name
        try:
            self.data = pd.read_sql_query(query, self.engine)
            return self.data
        except exc.SQLAlchemyError as e:
            raise ValueError(f"Failed to execute query: {e}")
    def get_data(self) -> pd.DataFrame:
        """
        Returns the loaded data. If no data has been loaded, it raises a ValueError.
        
        Returns:
            pd.DataFrame: The loaded data.

        Raises:
            ValueError: If data has not been loaded yet.
        """
        if self.data is not None:
            return self.data
        else:
            raise ValueError("Data has not been loaded yet. Please call 'load_data' first.")