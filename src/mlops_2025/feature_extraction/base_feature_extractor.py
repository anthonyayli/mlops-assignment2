"""Base abstract class for feature extractors."""

from abc import ABC, abstractmethod
from typing import Any


class BaseFeatureExtractor(ABC):
    """Abstract base class for all feature extractors.
    
    All feature extractor classes must implement the `feature_computer` method.
    """

    @abstractmethod
    def feature_computer(self, *args, **kwargs) -> Any:
        """Compute features from input data.
        
        This method should handle loading, feature engineering, and saving the results.
        
        Args:
            *args: Variable length argument list
            **kwargs: Arbitrary keyword arguments
            
        Returns:
            The computed features (type depends on the specific feature extractor)
            
        Raises:
            ValueError: If the input data is invalid
            IOError: If the data cannot be saved
        """
        pass

