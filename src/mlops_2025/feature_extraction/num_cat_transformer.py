"""Numerical and categorical transformer."""

import argparse
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, MinMaxScaler
from mlops_2025.feature_extraction.base_feature_extractor import BaseFeatureExtractor
from mlops_2025.loaders.data_loader import DataLoader
from mlops_2025.savers.transformers_saver import TransformersSaver


class NumCatTransformer(BaseFeatureExtractor):
    """Transformer for numerical and categorical features.
    
    Applies MinMaxScaler to numerical columns and OneHotEncoder/OrdinalEncoder
    to categorical columns.
    """

    def __init__(self):
        """Initialize the num_cat transformer."""
        self.loader = DataLoader()
        self.transformers_saver = TransformersSaver()
        self.transformer = ColumnTransformer([
            ('scaling', MinMaxScaler(), [0, 2]),
            ('onehotencolding1', OneHotEncoder(), [1, 3]),
            ('ordinal', OrdinalEncoder(), [4]),
            ('onehotencolding2', OneHotEncoder(), [5, 6])
        ], remainder='passthrough')

    def fit(self, X):
        """Fit the transformer on training data.
        
        Args:
            X: Training data to fit on
            
        Returns:
            self: Returns self for method chaining
        """
        self.transformer.fit(X)
        return self

    def transform(self, X):
        """Transform data using the fitted transformer.
        
        Args:
            X: Data to transform
            
        Returns:
            Transformed data
        """
        return self.transformer.transform(X)

    def fit_transform(self, X):
        """Fit the transformer and transform the data.
        
        Args:
            X: Training data to fit and transform
            
        Returns:
            Transformed data
        """
        return self.transformer.fit_transform(X)

    def feature_computer(
        self,
        input_frame: str,
        transformers_output: str
    ) -> ColumnTransformer:
        """Compute and save the num_cat transformer.
        
        This is the abstract method implementation.
        
        Args:
            input_frame: Path to input dataframe CSV file
            transformers_output: Directory to save fitted transformer
            
        Returns:
            ColumnTransformer: The fitted transformer
        """
        # Load data using DataLoader
        data = self.loader.load(input_frame)
        
        # Prepare training data
        train_data = data.dropna(subset=['Survived'])
        train_data['Survived'] = train_data['Survived'].astype('int64')
        train_data = train_data.drop("PassengerId", axis=1)
        X = train_data.drop("Survived", axis=1)
        
        # Fit transformer
        self.fit(X)
        
        # Save transformer
        transformers_dict = {
            'num_cat_transformer': self.transformer
        }
        self.transformers_saver.save_all(transformers_dict, transformers_output)
        
        return self.transformer


def build_parser() -> argparse.ArgumentParser:
    """Build argument parser for command-line interface.
    
    Returns:
        argparse.ArgumentParser: Configured argument parser
    """
    parser = argparse.ArgumentParser(description="NumCat transformer for Titanic dataset")
    parser.add_argument(
        "--input-frame",
        required=True,
        help="Path to dataframe CSV file"
    )
    parser.add_argument(
        "--transformers-output",
        required=True,
        help="Directory to save fitted transformer"
    )
    return parser


def main():
    """Main function to run the num_cat transformer from command line."""
    args = build_parser().parse_args()
    
    transformer = NumCatTransformer()
    transformer.feature_computer(
        input_frame=args.input_frame,
        transformers_output=args.transformers_output
    )


if __name__ == "__main__":
    main()

