#!/usr/bin/env python3
import argparse
import pandas as pd
import pickle
import os
from sklearn.linear_model import LogisticRegression

def load_csv(file_path: str, **kwargs) -> pd.DataFrame:
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"CSV file not found: {file_path}")
    
    return pd.read_csv(file_path, **kwargs)



def save_model(model, model_path: str) -> None:
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    
    print(f"Model saved to: {model_path}")


def load_data(features_path, labels_path):
    X = load_csv(features_path)
    y = load_csv(labels_path)
    if isinstance(y, pd.DataFrame):
        y = y.iloc[:, 0]
    X = X.values
    y = y.values if hasattr(y, 'values') else y
    return X, y


def train_model(X_train, y_train):
    classifier = LogisticRegression(max_iter=1000, random_state=42)
    classifier.fit(X_train, y_train)
    return classifier


def build_parser():
    p = argparse.ArgumentParser(description="Train model on Titanic dataset")
    p.add_argument("--train-features", required=True, help="Path to training features CSV")
    p.add_argument("--train-labels", required=True, help="Path to training labels CSV")
    p.add_argument("--model-output", default="models/model.pkl", help="Path to save trained model")
    return p


def main():
    args = build_parser().parse_args()
    
    X_train, y_train = load_data(args.train_features, args.train_labels)
    model = train_model(X_train, y_train)
    save_model(model, args.model_output)


if __name__ == "__main__":
    main()