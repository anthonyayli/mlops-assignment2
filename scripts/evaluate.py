#!/usr/bin/env python3
import argparse
import pandas as pd
import json
import os
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from mlops_2025.loaders import DataLoader, ModelLoader, TransformersLoader


def save_metrics(metrics: dict, output_path: str) -> None:
    """Save metrics dictionary to a JSON file.
    
    Args:
        metrics: Dictionary containing metric values
        output_path: Path to save the JSON file
        
    Raises:
        IOError: If the file cannot be written
    """
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True) if os.path.dirname(output_path) else None
    
    try:
        with open(output_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        print(f"\nMetrics saved to {output_path}")
    except Exception as e:
        raise IOError(f"Failed to save metrics to {output_path}: {str(e)}") from e


def process_test_data_with_transformers(
    test_features_path: str, 
    transformers_dir: str, 
    data_loader: DataLoader,
    transformers_loader: TransformersLoader
):
    """Process test data by applying transformers.
    
    Args:
        test_features_path: Path to test features CSV file
        transformers_dir: Directory containing transformer pickle files
        data_loader: DataLoader instance to load CSV
        transformers_loader: TransformersLoader instance to load transformers
        
    Returns:
        Transformed test features
    """
    X_test_raw = data_loader.load(test_features_path)
    
    # Load all transformers from directory
    transformers = transformers_loader.load_all(transformers_dir)
    
    num_cat_transformer = transformers.get('num_cat_transformer')
    bins_transformer = transformers.get('bins_transformer')
    
    if num_cat_transformer is None:
        raise ValueError(f"num_cat_transformer.pkl not found in {transformers_dir}")
    if bins_transformer is None:
        raise ValueError(f"bins_transformer.pkl not found in {transformers_dir}")
    
    X_test_transformed = num_cat_transformer.transform(X_test_raw)
    X_test_final = bins_transformer.transform(X_test_transformed)
    
    return X_test_final


def load_data(
    features_path: str, 
    labels_path: str, 
    data_loader: DataLoader,
    transformers_dir: str = None,
    transformers_loader: TransformersLoader = None
):
    """Load and optionally transform test data.
    
    Args:
        features_path: Path to test features CSV file
        labels_path: Path to test labels CSV file
        data_loader: DataLoader instance to load CSV files
        transformers_dir: Optional directory containing transformers
        transformers_loader: Optional TransformersLoader instance
        
    Returns:
        Tuple of (X_test, y_test) as numpy arrays
    """
    if transformers_dir and transformers_loader:
        X = process_test_data_with_transformers(
            features_path, transformers_dir, data_loader, transformers_loader
        )
        if X is None:
            raise ValueError("Failed to process test data")
    else:
        X = data_loader.load(features_path)
        X = X.values
    
    y = data_loader.load(labels_path)
    if isinstance(y, pd.DataFrame):
        y = y.iloc[:, 0]
    y = y.values if hasattr(y, 'values') else y
    return X, y


def compute_metrics(model, X_test, y_test):
    y_pred = model.predict(X_test)
    
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1_score': f1_score(y_test, y_pred)
    }
    return metrics


def print_metrics(metrics):
    print("Model Evaluation Metrics:")
    print("-" * 30)
    for metric_name, value in metrics.items():
        print(f"{metric_name.capitalize()}: {value:.4f}")


def build_parser():
    p = argparse.ArgumentParser(description="Evaluate trained model")
    p.add_argument("--model", required=True, help="Path to trained model pickle file")
    p.add_argument("--test-features", required=True, help="Path to raw test features CSV")
    p.add_argument("--test-labels", required=True, help="Path to test labels CSV")
    p.add_argument("--transformers-dir", default="transformers", help="Directory containing saved transformers")
    p.add_argument("--metrics-output", help="Optional path to save metrics as JSON")
    return p


def main():
    args = build_parser().parse_args()
    
    # Initialize loaders
    data_loader = DataLoader()
    model_loader = ModelLoader()
    transformers_loader = TransformersLoader() if args.transformers_dir else None
    
    # Load model
    model = model_loader.load(args.model)
    
    # Load and transform test data
    X_test, y_test = load_data(
        args.test_features, 
        args.test_labels, 
        data_loader,
        args.transformers_dir,
        transformers_loader
    )
    
    # Compute metrics
    metrics = compute_metrics(model, X_test, y_test)
    print_metrics(metrics)
    
    # Save metrics if output path provided
    if args.metrics_output:
        save_metrics(metrics, args.metrics_output)


if __name__ == "__main__":
    main()