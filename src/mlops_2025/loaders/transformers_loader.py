"""Transformers loader for pickle files."""

import os
from typing import Any, Dict
import pickle
from mlops_2025.loaders.baseloader import BaseLoader


class TransformersLoader(BaseLoader):
    """Loader for transformer objects (pickle files).
    
    Loads transformer objects saved as pickle files.
    Can load a single transformer or multiple transformers from a directory.
    """

    def load(self, path: str, **kwargs) -> Any:
        """Load a transformer from a pickle file.
        
        Args:
            path: Path to the pickle file containing the transformer
            **kwargs: Additional arguments (currently unused)
            
        Returns:
            The loaded transformer object
            
        Raises:
            FileNotFoundError: If the pickle file doesn't exist
            ValueError: If the file cannot be unpickled
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"Transformer file not found: {path}")
        
        try:
            with open(path, 'rb') as f:
                transformer = pickle.load(f)
            return transformer
        except Exception as e:
            raise ValueError(f"Failed to load transformer from {path}: {str(e)}") from e
    
    def load_all(self, directory: str) -> Dict[str, Any]:
        """Load all transformers from a directory.
        
        Args:
            directory: Path to the directory containing transformer pickle files
            
        Returns:
            Dictionary mapping transformer names to transformer objects
            
        Raises:
            FileNotFoundError: If the directory doesn't exist
        """
        if not os.path.exists(directory):
            raise FileNotFoundError(f"Transformers directory not found: {directory}")
        
        transformers = {}
        for filename in os.listdir(directory):
            if filename.endswith('.pkl'):
                transformer_name = filename.replace('.pkl', '')
                transformer_path = os.path.join(directory, filename)
                transformers[transformer_name] = self.load(transformer_path)
        
        return transformers
