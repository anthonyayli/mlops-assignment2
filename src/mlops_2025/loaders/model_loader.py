"""Model loader for pickle files."""

import os
from typing import Any
import pickle
from mlops_2025.loaders.baseloader import BaseLoader


class ModelLoader(BaseLoader):
    """Loader for machine learning models (pickle files).
    
    Loads trained model objects saved as pickle files.
    """

    def load(self, path: str, **kwargs) -> Any:
        """Load a model from a pickle file.
        
        Args:
            path: Path to the pickle file containing the model
            **kwargs: Additional arguments (currently unused)
            
        Returns:
            The loaded model object
            
        Raises:
            FileNotFoundError: If the pickle file doesn't exist
            ValueError: If the file cannot be unpickled
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"Model file not found: {path}")
        
        try:
            with open(path, 'rb') as f:
                model = pickle.load(f)
            print(f"Model loaded from: {path}")
            return model
        except Exception as e:
            raise ValueError(f"Failed to load model from {path}: {str(e)}") from e
