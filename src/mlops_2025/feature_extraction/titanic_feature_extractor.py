"""Titanic dataset feature extractor."""

import argparse
import warnings
import pandas as pd
from sklearn.model_selection import train_test_split
from mlops_2025.feature_extraction.base_feature_extractor import BaseFeatureExtractor
from mlops_2025.feature_extraction.num_cat_transformer import NumCatTransformer
from mlops_2025.feature_extraction.bins_transformer import BinsTransformer
from mlops_2025.loaders.data_loader import DataLoader
from mlops_2025.savers.data_saver import DataSaver
from mlops_2025.savers.transformers_saver import TransformersSaver

warnings.filterwarnings("ignore")


class TitanicFeatureExtractor(BaseFeatureExtractor):
    """Feature extractor for Titanic dataset.
    
    Handles feature engineering, transformation, and saving of features and labels.
    Uses DataLoader for loading and DataSaver/TransformersSaver for saving.
    Matches the logic from scripts/feature_extraction.py
    """

    def __init__(self):
        """Initialize the Titanic feature extractor with loaders and savers."""
        self.loader = DataLoader()
        self.data_saver = DataSaver()
        self.transformers_saver = TransformersSaver()
        self.num_cat_transformer = NumCatTransformer()
        self.bins_transformer = BinsTransformer()

    def feature_computer(
        self,
        input_frame: str,
        output_train_features: str,
        output_train_labels: str,
        output_test_features: str,
        output_test_labels: str,
        transformers_output: str
    ) -> tuple:
        """Compute features from input data and save results.
        
        This is the abstract method implementation that handles loading, 
        feature engineering, transformation, and saving.
        Matches the logic from scripts/feature_extraction.py
        
        Args:
            input_frame: Path to input dataframe CSV file
            output_train_features: Path to save training features (transformed)
            output_train_labels: Path to save training labels
            output_test_features: Path to save test features (raw, from split)
            output_test_labels: Path to save test labels
            transformers_output: Directory to save fitted transformers
            
        Returns:
            tuple: (X_train_df, X_test_raw, y_train, y_test) as pandas DataFrames
            
        Raises:
            FileNotFoundError: If input file doesn't exist
            ValueError: If data cannot be processed
        """
        # Load data using DataLoader
        data = self.loader.load(input_frame)
        
        # Prepare training data (filter for rows with Survived)
        train_data = data.dropna(subset=['Survived'])
        train_data['Survived'] = train_data['Survived'].astype('int64')
        train_data = train_data.drop("PassengerId", axis=1)
        X = train_data.drop("Survived", axis=1)
        Y = train_data["Survived"]
        
        # Split data (80/20)
        X_train, X_test, y_train, y_test = train_test_split(
            X, Y, test_size=0.2, random_state=42
        )
        
        # Fit and transform training data
        # First apply num_cat_transformer
        X_train_transformed = self.num_cat_transformer.fit_transform(X_train)
        # Then apply bins_transformer on the transformed data
        X_train_final = self.bins_transformer.fit_transform(X_train_transformed)
        
        # Save transformers
        transformers_dict = {
            'num_cat_transformer': self.num_cat_transformer.transformer,
            'bins_transformer': self.bins_transformer.transformer
        }
        self.transformers_saver.save_all(transformers_dict, transformers_output)
        
        # Convert to DataFrame
        X_train_df = pd.DataFrame(X_train_final)
        
        # Save results using DataSaver
        self.data_saver.save(X_train_df, output_train_features)
        self.data_saver.save(y_train, output_train_labels)
        self.data_saver.save(X_test, output_test_features)
        self.data_saver.save(y_test, output_test_labels)
        
        return X_train_df, X_test, y_train, y_test


def build_parser() -> argparse.ArgumentParser:
    """Build argument parser for command-line interface.
    
    Returns:
        argparse.ArgumentParser: Configured argument parser
    """
    parser = argparse.ArgumentParser(description="Feature engineering for Titanic dataset")
    parser.add_argument(
        "--input-frame",
        required=True,
        help="Path to dataframe CSV file"
    )
    parser.add_argument(
        "--output-train-features",
        required=True,
        help="Path to save training features (transformed)"
    )
    parser.add_argument(
        "--output-train-labels",
        required=True,
        help="Path to save training labels"
    )
    parser.add_argument(
        "--output-test-features",
        required=True,
        help="Path to save test features (raw, from split)"
    )
    parser.add_argument(
        "--output-test-labels",
        required=True,
        help="Path to save test labels"
    )
    parser.add_argument(
        "--transformers-output",
        required=True,
        help="Directory to save fitted transformers"
    )
    return parser


def main():
    """Main function to run the feature extractor from command line."""
    args = build_parser().parse_args()
    
    extractor = TitanicFeatureExtractor()
    extractor.feature_computer(
        input_frame=args.input_frame,
        output_train_features=args.output_train_features,
        output_train_labels=args.output_train_labels,
        output_test_features=args.output_test_features,
        output_test_labels=args.output_test_labels,
        transformers_output=args.transformers_output
    )


if __name__ == "__main__":
    main()

