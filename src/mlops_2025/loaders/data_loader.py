"""Data loader for CSV files."""

import os
from typing import Any
import pandas as pd
from mlops_2025.loaders.baseloader import BaseLoader


class DataLoader(BaseLoader):
    """Loader for CSV data files.
    
    Loads CSV files and returns pandas DataFrames.
    """

    def load(self, path: str, **kwargs) -> pd.DataFrame:
        """Load a CSV file as a pandas DataFrame.
        
        Args:
            path: Path to the CSV file
            **kwargs: Additional arguments passed to pd.read_csv()
                (e.g., sep, encoding, index_col, etc.)
            
        Returns:
            pandas.DataFrame: The loaded data
            
        Raises:
            FileNotFoundError: If the CSV file doesn't exist
            ValueError: If the file cannot be read as CSV
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"CSV file not found: {path}")
        
        try:
            return pd.read_csv(path, **kwargs)
        except Exception as e:
            raise ValueError(f"Failed to load CSV from {path}: {str(e)}") from e
