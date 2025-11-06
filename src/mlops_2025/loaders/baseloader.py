"""Base abstract class for loaders."""

from abc import ABC, abstractmethod
from typing import Any


class BaseLoader(ABC):
    """Abstract base class for all loaders.
    
    All loader classes must implement the `load` method.
    """

    @abstractmethod
    def load(self, path: str, **kwargs) -> Any:
        """Load data from the specified path.
        
        Args:
            path: Path to the resource to load
            **kwargs: Additional keyword arguments for loading
            
        Returns:
            The loaded resource (type depends on the specific loader)
            
        Raises:
            FileNotFoundError: If the file at the specified path doesn't exist
            ValueError: If the file cannot be loaded or is invalid
        """
        pass


