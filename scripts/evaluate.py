#!/usr/bin/env python3
import argparse
import pandas as pd
import pickle
import json
import os
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def load_csv(file_path: str, **kwargs) -> pd.DataFrame:
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"CSV file not found: {file_path}")
    
    return pd.read_csv(file_path, **kwargs)


def load_model(model_path: str):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    
    print(f"Model loaded from: {model_path}")
    return model


def load_transformer(transformer_path: str):
    if not os.path.exists(transformer_path):
        raise FileNotFoundError(f"Transformer file not found: {transformer_path}")
    
    with open(transformer_path, 'rb') as f:
        transformer = pickle.load(f)
    
    return transformer


def save_metrics(metrics, output_path):
    with open(output_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"\nMetrics saved to {output_path}")


def process_test_data_with_transformers(test_features_path, transformers_dir):
    X_test_raw = load_csv(test_features_path)
    
    num_cat_transformer = load_transformer(os.path.join(transformers_dir, 'num_cat_transformer.pkl'))
    bins_transformer = load_transformer(os.path.join(transformers_dir, 'bins_transformer.pkl'))
    
    X_test_transformed = num_cat_transformer.transform(X_test_raw)
    X_test_final = bins_transformer.transform(X_test_transformed)
    
    return X_test_final


def load_data(features_path, labels_path, transformers_dir=None):
    if transformers_dir:
        X = process_test_data_with_transformers(features_path, transformers_dir)
        if X is None:
            raise ValueError("Failed to process test data")
    else:
        X = load_csv(features_path)
        X = X.values
    
    y = load_csv(labels_path)
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
    
    model = load_model(args.model)
    X_test, y_test = load_data(args.test_features, args.test_labels, args.transformers_dir)
    
    metrics = compute_metrics(model, X_test, y_test)
    print_metrics(metrics)
    
    if args.metrics_output:
        save_metrics(metrics, args.metrics_output)


if __name__ == "__main__":
    main()