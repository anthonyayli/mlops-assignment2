"""Data saver for CSV files."""

import os
from typing import Any
import pandas as pd
from mlops_2025.savers.basesaver import BaseSaver


class DataSaver(BaseSaver):
    """Saver for CSV data files.
    
    Saves pandas DataFrames to CSV files.
    """

    def save(self, data: pd.DataFrame, path: str, **kwargs) -> None:
        """Save a pandas DataFrame or Series to a CSV file.
        
        Args:
            data: pandas.DataFrame or pandas.Series to save
            path: Path to the CSV file where data should be saved
            **kwargs: Additional arguments passed to pd.DataFrame.to_csv()
                (e.g., index, sep, encoding, etc.)
            
        Raises:
            ValueError: If data is not a pandas DataFrame or Series
            IOError: If the file cannot be written
        """
        # Convert Series to DataFrame if needed
        if isinstance(data, pd.Series):
            data = pd.DataFrame(data)
        elif not isinstance(data, pd.DataFrame):
            raise ValueError(f"Data must be a pandas DataFrame or Series, got {type(data)}")
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(path), exist_ok=True) if os.path.dirname(path) else None
        
        try:
            # Default to index=False if not specified
            if 'index' not in kwargs:
                kwargs['index'] = False
            data.to_csv(path, **kwargs)
            print(f"DataFrame saved to: {path}")
        except Exception as e:
            raise IOError(f"Failed to save CSV to {path}: {str(e)}") from e


