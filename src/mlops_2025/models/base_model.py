"""Base abstract class for models."""

from abc import ABC, abstractmethod
from typing import Any


class BaseModel(ABC):
    """Abstract base class for all models.
    
    All model classes must implement the `train` method.
    """

    @abstractmethod
    def train(self, *args, **kwargs) -> Any:
        """Train the model.
        
        This method should handle loading data, training the model, and saving it.
        
        Args:
            *args: Variable length argument list
            **kwargs: Arbitrary keyword arguments
            
        Returns:
            The trained model (type depends on the specific model)
            
        Raises:
            ValueError: If the input data is invalid
            IOError: If the model cannot be saved
        """
        pass

