"""Transformers saver for pickle files."""

import os
from typing import Any, Dict
import pickle
from mlops_2025.savers.basesaver import BaseSaver


class TransformersSaver(BaseSaver):
    """Saver for transformer objects (pickle files).
    
    Saves transformer objects as pickle files.
    Can save a single transformer or multiple transformers to a directory.
    """

    def save(self, data: Any, path: str, **kwargs) -> None:
        """Save a transformer to a pickle file.
        
        Args:
            data: Transformer object to save
            path: Path to the pickle file where transformer should be saved
            **kwargs: Additional arguments (currently unused)
            
        Raises:
            IOError: If the file cannot be written
        """
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(path), exist_ok=True) if os.path.dirname(path) else None
        
        try:
            with open(path, 'wb') as f:
                pickle.dump(data, f)
        except Exception as e:
            raise IOError(f"Failed to save transformer to {path}: {str(e)}") from e
    
    def save_all(self, transformers: Dict[str, Any], directory: str, **kwargs) -> None:
        """Save multiple transformers to a directory.
        
        Args:
            transformers: Dictionary mapping transformer names to transformer objects
            directory: Directory path where transformers should be saved
            **kwargs: Additional arguments passed to save() for each transformer
            
        Raises:
            ValueError: If transformers is not a dictionary
            IOError: If the directory cannot be created or files cannot be written
        """
        if not isinstance(transformers, dict):
            raise ValueError(f"Transformers must be a dictionary, got {type(transformers)}")
        
        os.makedirs(directory, exist_ok=True)
        
        for name, transformer in transformers.items():
            # Ensure .pkl extension
            filename = name if name.endswith('.pkl') else f"{name}.pkl"
            transformer_path = os.path.join(directory, filename)
            self.save(transformer, transformer_path, **kwargs)
        
        print(f"Transformers saved to {directory}/")


