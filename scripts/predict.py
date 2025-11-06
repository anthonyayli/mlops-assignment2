#!/usr/bin/env python3
import argparse
import pandas as pd
import pickle
import os
from pathlib import Path


def load_csv(file_path: str, **kwargs) -> pd.DataFrame:
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"CSV file not found: {file_path}")
    
    return pd.read_csv(file_path, **kwargs)


def save_csv(df: pd.DataFrame, file_path: str, **kwargs) -> None:
    df.to_csv(file_path, index=False, **kwargs)
    print(f"DataFrame saved to: {file_path}")


def load_model(model_path: str):
    print(f"Loading model from {model_path}...")
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    print(f"Model loaded successfully: {type(model).__name__}")
    return model


def load_transformer(transformer_path: str):
    if not os.path.exists(transformer_path):
        raise FileNotFoundError(f"Transformer file not found: {transformer_path}")
    
    with open(transformer_path, 'rb') as f:
        transformer = pickle.load(f)
    
    return transformer


def preprocess_inference_data(df: pd.DataFrame) -> pd.DataFrame:
    print("Applying preprocessing to inference data...")
    
    df_processed = df.copy()
    
    df_processed['Age'].fillna(df_processed['Age'].median(), inplace=True)
    df_processed['Embarked'].fillna('S', inplace=True)
    df_processed['Fare'].fillna(df_processed['Fare'].median(), inplace=True)
    
    df_processed['FamilySize'] = df_processed['SibSp'] + df_processed['Parch'] + 1
    df_processed['IsAlone'] = (df_processed['FamilySize'] == 1).astype(int)
    
    df_processed['Title'] = df_processed['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)
    df_processed['Title'] = df_processed['Title'].replace(['Lady', 'Countess','Capt', 'Col',
        'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
    df_processed['Title'] = df_processed['Title'].replace('Mlle', 'Miss')
    df_processed['Title'] = df_processed['Title'].replace('Ms', 'Miss')
    df_processed['Title'] = df_processed['Title'].replace('Mme', 'Mrs')
    
    df_processed['Title'].fillna('Mr', inplace=True)
    
    df_processed['AgeGroup'] = pd.cut(df_processed['Age'], bins=[0, 12, 18, 35, 60, 100], 
                                     labels=['Child', 'Teen', 'Young Adult', 'Adult', 'Senior'])
    
    df_processed['FareGroup'] = pd.qcut(df_processed['Fare'], q=4, labels=['Low', 'Medium', 'High', 'Very High'])
    
    print("Preprocessing completed.")
    return df_processed


def apply_transformers(df: pd.DataFrame, transformers_dir: str) -> pd.DataFrame:
    print(f"Loading transformers from {transformers_dir}...")
    
    num_cat_transformer = load_transformer(os.path.join(transformers_dir, 'num_cat_transformer.pkl'))
    bins_transformer = load_transformer(os.path.join(transformers_dir, 'bins_transformer.pkl'))
    
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


def save_predictions(predictions: pd.Series, passenger_ids: pd.Series, output_path: str):
    print(f"Saving predictions to {output_path}...")
    
    results_df = pd.DataFrame({
        'PassengerId': passenger_ids,
        'Survived': predictions
    })
    
    save_csv(results_df, output_path)
    
    print(f"Predictions saved successfully!")
    print(f"Results summary:")
    print(f"  Total predictions: {len(predictions)}")
    print(f"  Predicted survivors: {predictions.sum()}")
    print(f"  Predicted deaths: {len(predictions) - predictions.sum()}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Make predictions on new Titanic data")
    p.add_argument("--model", required=True, help="Path to trained model file (.pkl)")
    p.add_argument("--input-data", required=True, help="Path to inference data CSV file")
    p.add_argument("--transformers-dir", required=True, help="Directory containing saved transformers")
    p.add_argument("--output", required=True, help="Path to save predictions CSV file")
    return p


def main():
    args = build_parser().parse_args()
    
    print("=" * 50)
    print("TITANIC SURVIVAL PREDICTION - INFERENCE")
    print("=" * 50)
    
    print(f"Loading inference data from {args.input_data}...")
    df_raw = load_csv(args.input_data)
    print(f"Loaded {len(df_raw)} samples for inference.")
    
    passenger_ids = df_raw['PassengerId'].copy()
    
    df_features = df_raw.drop('PassengerId', axis=1)
    
    df_preprocessed = preprocess_inference_data(df_features)
    
    df_transformed = apply_transformers(df_preprocessed, args.transformers_dir)
    
    model = load_model(args.model)
    
    predictions = make_predictions(model, df_transformed)
    
    save_predictions(predictions, passenger_ids, args.output)
    
    print("=" * 50)
    print("INFERENCE COMPLETED SUCCESSFULLY!")
    print("=" * 50)


if __name__ == "__main__":
    main()