"""XGBoost model."""

import argparse
import pandas as pd
from mlops_2025.models.base_model import BaseModel
from mlops_2025.loaders.data_loader import DataLoader
from mlops_2025.savers.model_saver import ModelSaver

try:
    from xgboost import XGBClassifier
except ImportError:
    XGBClassifier = None


class XGBoostModel(BaseModel):
    """XGBoost model for classification.
    
    Uses DataLoader for loading and ModelSaver for saving.
    Loads already-transformed features (no transformers applied during training).
    """

    def __init__(self, random_state=42, **kwargs):
        """Initialize the XGBoost model.
        
        Args:
            random_state: Random state for reproducibility
            **kwargs: Additional arguments passed to XGBClassifier
        """
        if XGBClassifier is None:
            raise ImportError("xgboost is not installed. Install it with: pip install xgboost")
        
        self.loader = DataLoader()
        self.saver = ModelSaver()
        self.random_state = random_state
        kwargs['random_state'] = random_state
        self.model = XGBClassifier(**kwargs)

    def train(
        self,
        train_features: str,
        train_labels: str,
        model_output: str
    ):
        """Train the model and save it.
        
        This is the abstract method implementation.
        Loads already-transformed features (matching scripts/train.py logic).
        
        Args:
            train_features: Path to training features CSV (already transformed)
            train_labels: Path to training labels CSV
            model_output: Path to save trained model
            
        Returns:
            XGBClassifier: The trained model
        """
        # Load already-transformed features using DataLoader
        X = self.loader.load(train_features)
        y = self.loader.load(train_labels)
        
        if isinstance(y, pd.DataFrame):
            y = y.iloc[:, 0]
        
        X_train = X.values
        y_train = y.values if hasattr(y, 'values') else y
        
        # Train model on already-transformed features
        self.model.fit(X_train, y_train)
        
        # Save model
        self.saver.save(self.model, model_output)
        
        return self.model


def build_parser() -> argparse.ArgumentParser:
    """Build argument parser for command-line interface.
    
    Returns:
        argparse.ArgumentParser: Configured argument parser
    """
    parser = argparse.ArgumentParser(description="Train XGBoost model on Titanic dataset")
    parser.add_argument(
        "--train-features",
        required=True,
        help="Path to training features CSV (already transformed)"
    )
    parser.add_argument(
        "--train-labels",
        required=True,
        help="Path to training labels CSV"
    )
    parser.add_argument(
        "--model-output",
        default="models/xgboost.pkl",
        help="Path to save trained model"
    )
    return parser


def main():
    """Main function to run the XGBoost model from command line."""
    args = build_parser().parse_args()
    
    model = XGBoostModel()
    model.train(
        train_features=args.train_features,
        train_labels=args.train_labels,
        model_output=args.model_output
    )


if __name__ == "__main__":
    main()

