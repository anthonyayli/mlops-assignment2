"""Base abstract class for savers."""

from abc import ABC, abstractmethod
from typing import Any


class BaseSaver(ABC):
    """Abstract base class for all savers.
    
    All saver classes must implement the `save` method.
    """

    @abstractmethod
    def save(self, data: Any, path: str, **kwargs) -> None:
        """Save data to the specified path.
        
        Args:
            data: The data to save (type depends on the specific saver)
            path: Path where the data should be saved
            **kwargs: Additional keyword arguments for saving
            
        Raises:
            ValueError: If the data cannot be saved or is invalid
            IOError: If there's an error writing to the file
        """
        pass


