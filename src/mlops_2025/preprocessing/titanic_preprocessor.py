"""Titanic dataset preprocessor."""

import argparse
import warnings
import pandas as pd
from mlops_2025.preprocessing.base_preprocessor import BasePreprocessor
from mlops_2025.loaders.data_loader import DataLoader
from mlops_2025.savers.data_saver import DataSaver

warnings.filterwarnings("ignore")


class TitanicPreprocessor(BasePreprocessor):
    """Preprocessor for Titanic dataset.
    
    Handles data loading, cleaning, and preprocessing steps for Titanic survival prediction.
    Uses DataLoader for loading and DataSaver for saving.
    """

    def __init__(self):
        """Initialize the Titanic preprocessor with loader and saver."""
        self.loader = DataLoader()
        self.saver = DataSaver()

    @staticmethod
    def family_size(number):
        """Categorize family size.
        
        Args:
            number: Family size number
            
        Returns:
            str: Family size category ('Alone', 'Small', or 'Large')
        """
        if number == 1:
            return "Alone"
        elif number > 1 and number < 5:
            return "Small"
        else:
            return "Large"

    def preprocess(
        self, 
        train_path: str, 
        test_path: str, 
        output_path: str
    ) -> pd.DataFrame:
        """Preprocess Titanic training and test datasets and save the result.
        
        This is the abstract method implementation that handles loading, preprocessing, and saving.
        
        Args:
            train_path: Path to training CSV file
            test_path: Path to test CSV file
            output_path: Path to save preprocessed data
            
        Returns:
            pandas.DataFrame: Preprocessed and combined dataframe
            
        Raises:
            FileNotFoundError: If input files don't exist
            ValueError: If data cannot be processed
        """
        # Load data using DataLoader
        train = self.loader.load(train_path)
        test = self.loader.load(test_path)
        
        # Drop Cabin column due to numerous null values
        train.drop(columns=['Cabin'], inplace=True)
        test.drop(columns=['Cabin'], inplace=True)
        
        # Fill missing values
        train['Embarked'].fillna('S', inplace=True)
        test['Fare'].fillna(test['Fare'].mean(), inplace=True)
        
        # Create unified dataframe for easier manipulation
        df = pd.concat([train, test], sort=True).reset_index(drop=True)
        
        # Fill missing Age values using group median
        df['Age'] = df.groupby(['Sex', 'Pclass'])['Age'].transform(
            lambda x: x.fillna(x.median())
        )
        
        # Extract Title from Name
        df['Title'] = df['Name'].str.split(", ", expand=True)[1].str.split(".", expand=True)[0]
        df['Age'] = df['Age'].astype('int64')
        
        # Replace rare titles
        df['Title'] = df['Title'].replace([
            'Lady', 'the Countess', 'Capt', 'Col', 'Don', 'Dr', 
            'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'
        ], 'Rare')
        df['Title'] = df['Title'].replace('Mlle', 'Miss')
        df['Title'] = df['Title'].replace('Ms', 'Miss')
        df['Title'] = df['Title'].replace('Mme', 'Mrs')
        
        # Create Family_size feature
        df['Family_size'] = df['SibSp'] + df['Parch'] + 1
        df.drop(columns=['Name', 'Parch', 'SibSp', 'Ticket'], inplace=True)
        df['Family_size'] = df['Family_size'].apply(self.family_size)
        
        # Save using DataSaver
        self.saver.save(df, output_path)
        
        return df



def build_parser() -> argparse.ArgumentParser:
    """Build argument parser for command-line interface.
    
    Returns:
        argparse.ArgumentParser: Configured argument parser
    """
    parser = argparse.ArgumentParser(description="Preprocess Titanic dataset")
    parser.add_argument(
        "--input-train", 
        required=True, 
        help="Path to training data CSV file"
    )
    parser.add_argument(
        "--input-test", 
        required=True, 
        help="Path to test data CSV file"
    )
    parser.add_argument(
        "--output", 
        required=True, 
        help="Path to save processed data"
    )
    return parser


def main():
    """Main function to run the preprocessor from command line."""
    args = build_parser().parse_args()
    
    preprocessor = TitanicPreprocessor()
    preprocessor.preprocess(
        train_path=args.input_train,
        test_path=args.input_test,
        output_path=args.output
    )


if __name__ == "__main__":
    main()

