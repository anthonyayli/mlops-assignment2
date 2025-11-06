"""Base abstract class for preprocessors."""

from abc import ABC, abstractmethod
from typing import Any


class BasePreprocessor(ABC):
    """Abstract base class for all preprocessors.
    
    All preprocessor classes must implement the `preprocess` method.
    """

    @abstractmethod
    def preprocess(self, *args, **kwargs) -> Any:
        """Preprocess data.
        
        This method should handle loading, preprocessing, and saving the data.
        
        Args:
            *args: Variable length argument list
            **kwargs: Arbitrary keyword arguments
            
        Returns:
            The preprocessed data (type depends on the specific preprocessor)
            
        Raises:
            ValueError: If the input data is invalid
            IOError: If the data cannot be saved
        """
        pass

