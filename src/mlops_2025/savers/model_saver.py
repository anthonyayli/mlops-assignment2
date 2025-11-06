"""Model saver for pickle files."""

import os
from typing import Any
import pickle
from mlops_2025.savers.basesaver import BaseSaver


class ModelSaver(BaseSaver):
    """Saver for machine learning models (pickle files).
    
    Saves trained model objects as pickle files.
    """

    def save(self, data: Any, path: str, **kwargs) -> None:
        """Save a model to a pickle file.
        
        Args:
            data: Model object to save
            path: Path to the pickle file where model should be saved
            **kwargs: Additional arguments (currently unused)
            
        Raises:
            IOError: If the file cannot be written
        """
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(path), exist_ok=True) if os.path.dirname(path) else None
        
        try:
            with open(path, 'wb') as f:
                pickle.dump(data, f)
            print(f"Model saved to: {path}")
        except Exception as e:
            raise IOError(f"Failed to save model to {path}: {str(e)}") from e


