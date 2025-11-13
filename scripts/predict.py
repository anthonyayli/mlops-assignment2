#!/usr/bin/env python3
import argparse
import pandas as pd
from mlops_2025.loaders import DataLoader, ModelLoader, TransformersLoader
from mlops_2025.savers import DataSaver


def extract_test_data_from_processed(processed_data_path: str, data_loader: DataLoader) -> pd.DataFrame:
    """Extract preprocessed test data from processed_data.csv.
    
    The processed_data.csv contains both train and test rows.
    Test rows are identified by Survived being NaN.
    
    Args:
        processed_data_path: Path to processed_data.csv file
        data_loader: DataLoader instance to load the CSV file
        
    Returns:
        DataFrame containing only the preprocessed test rows
    """
    print(f"Loading preprocessed data from {processed_data_path}...")
    df_processed = data_loader.load(processed_data_path)
    print(f"Loaded {len(df_processed)} total rows (train + test).")
    
    # Extract test rows (where Survived is NaN)
    test_data = df_processed[df_processed['Survived'].isna()].copy()
    print(f"Extracted {len(test_data)} preprocessed test rows.")
    
    if len(test_data) == 0:
        raise ValueError("No test data found in processed_data.csv (no rows with Survived=NaN)")
    
    return test_data


def apply_transformers(df: pd.DataFrame, transformers_dir: str, transformers_loader: TransformersLoader) -> pd.DataFrame:
    """Apply transformers to the data.
    
    Args:
        df: DataFrame to transform
        transformers_dir: Directory containing transformer pickle files
        transformers_loader: TransformersLoader instance to load transformers
        
    Returns:
        Transformed DataFrame
    """
    print(f"Loading transformers from {transformers_dir}...")
    
    # Load all transformers from directory
    transformers = transformers_loader.load_all(transformers_dir)
    
    num_cat_transformer = transformers.get('num_cat_transformer')
    bins_transformer = transformers.get('bins_transformer')
    
    if num_cat_transformer is None:
        raise ValueError(f"num_cat_transformer.pkl not found in {transformers_dir}")
    if bins_transformer is None:
        raise ValueError(f"bins_transformer.pkl not found in {transformers_dir}")
    
    print("Applying transformations...")
    
    df_transformed = num_cat_transformer.transform(df)
    df_final = bins_transformer.transform(df_transformed)
    
    print("Transformations applied successfully.")
    return df_final


def make_predictions(model, features: pd.DataFrame) -> pd.Series:
    print("Making predictions...")
    predictions = model.predict(features)
    print(f"Generated {len(predictions)} predictions.")
    return predictions


def save_predictions(predictions: pd.Series, passenger_ids: pd.Series, output_path: str, data_saver: DataSaver):
    """Save predictions to CSV file.
    
    Args:
        predictions: Series of predictions
        passenger_ids: Series of passenger IDs
        output_path: Path to save the predictions CSV
        data_saver: DataSaver instance to save the CSV file
    """
    print(f"Saving predictions to {output_path}...")
    
    results_df = pd.DataFrame({
        'PassengerId': passenger_ids,
        'Survived': predictions
    })
    
    data_saver.save(results_df, output_path)
    
    print(f"Predictions saved successfully!")
    print(f"Results summary:")
    print(f"  Total predictions: {len(predictions)}")
    print(f"  Predicted survivors: {predictions.sum()}")
    print(f"  Predicted deaths: {len(predictions) - predictions.sum()}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Make predictions on preprocessed test data")
    p.add_argument("--model", required=True, help="Path to trained model file (.pkl)")
    p.add_argument("--processed-data", required=True, help="Path to processed_data.csv (from preprocessing step)")
    p.add_argument("--transformers-dir", required=True, help="Directory containing saved transformers")
    p.add_argument("--output", required=True, help="Path to save predictions CSV file")
    return p


def main():
    args = build_parser().parse_args()
    
    # Initialize loaders and savers
    data_loader = DataLoader()
    model_loader = ModelLoader()
    transformers_loader = TransformersLoader()
    data_saver = DataSaver()
    
    print("=" * 50)
    print("TITANIC SURVIVAL PREDICTION - INFERENCE")
    print("=" * 50)
    
    # Extract preprocessed test data (rows where Survived is NaN)
    test_data = extract_test_data_from_processed(args.processed_data, data_loader)
    
    # Extract PassengerId before dropping it
    passenger_ids = test_data['PassengerId'].copy()
    
    # Prepare features: drop PassengerId and Survived columns
    df_features = test_data.drop(['PassengerId', 'Survived'], axis=1)
    
    print(f"Using {len(df_features)} preprocessed test samples for prediction.")
    print("Note: Data is already preprocessed, only applying transformers...")
    
    # Apply transformers (no preprocessing needed - already done)
    df_transformed = apply_transformers(df_features, args.transformers_dir, transformers_loader)
    
    # Load model and make predictions
    model = model_loader.load(args.model)
    print(f"Model loaded successfully: {type(model).__name__}")
    predictions = make_predictions(model, df_transformed)
    
    # Save predictions
    save_predictions(predictions, passenger_ids, args.output, data_saver)
    
    print("=" * 50)
    print("INFERENCE COMPLETED SUCCESSFULLY!")
    print("=" * 50)


if __name__ == "__main__":
    main()