#!/usr/bin/env python3
import argparse
import pandas as pd
import pickle
import os
import warnings
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import KBinsDiscretizer

warnings.filterwarnings("ignore")


def load_csv(file_path: str, **kwargs) -> pd.DataFrame:
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"CSV file not found: {file_path}")
    
    return pd.read_csv(file_path, **kwargs)


def save_csv(df: pd.DataFrame, file_path: str, **kwargs) -> None:
    df.to_csv(file_path, index=False, **kwargs)
    print(f"DataFrame saved to: {file_path}")


def get_cat_tranformation():
    num_cat_tranformation = ColumnTransformer([
        ('scaling', MinMaxScaler(), [0, 2]),
        ('onehotencolding1', OneHotEncoder(), [1, 3]),
        ('ordinal', OrdinalEncoder(), [4]),
        ('onehotencolding2', OneHotEncoder(), [5, 6])
    ], remainder='passthrough')
    return num_cat_tranformation


def get_bins():
    bins = ColumnTransformer([
        ('Kbins', KBinsDiscretizer(n_bins=15, encode='ordinal', strategy='quantile'), [0, 2]),
    ], remainder='passthrough')
    return bins


def save_transformers(num_cat_transformation, bins, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    
    with open(os.path.join(output_dir, 'num_cat_transformer.pkl'), 'wb') as f:
        pickle.dump(num_cat_transformation, f)
    
    with open(os.path.join(output_dir, 'bins_transformer.pkl'), 'wb') as f:
        pickle.dump(bins, f)
    
    print(f"Transformers saved to {output_dir}/")


def engineering_data(df_path, transformers_output_dir):
    data = load_csv(df_path)
    train_data = data.dropna(subset=['Survived'])
    train_data['Survived'] = train_data['Survived'].astype('int64')
    train_data = train_data.drop("PassengerId", axis=1)
    X = train_data.drop("Survived", axis=1)
    Y = train_data["Survived"]
    
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
    
    num_cat_transformation = get_cat_tranformation()
    bins = get_bins()
    
    X_train_transformed = num_cat_transformation.fit_transform(X_train)
    X_train_final = bins.fit_transform(X_train_transformed)
    
    save_transformers(num_cat_transformation, bins, transformers_output_dir)
    
    X_train_df = pd.DataFrame(X_train_final)
    
    return X_train_df, X_test, y_train, y_test


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Feature engineering for Titanic dataset")
    p.add_argument("--input-frame", required=True, help="Path to dataframe CSV file")
    p.add_argument("--output-train-features", required=True, help="Path to save training features")
    p.add_argument("--output-train-labels", required=True, help="Path to save training labels")
    p.add_argument("--output-test-features", required=True, help="Path to save test features")
    p.add_argument("--output-test-labels", required=True, help="Path to save test labels")
    p.add_argument("--transformers-output", required=True, help="Directory to save fitted transformers")
    return p


def main():
    args = build_parser().parse_args()
    
    X_train_df, X_test_raw, y_train, y_test = engineering_data(args.input_frame, args.transformers_output)
    
    save_csv(X_train_df, args.output_train_features)
    save_csv(y_train, args.output_train_labels)
    
    save_csv(X_test_raw, args.output_test_features)
    save_csv(y_test, args.output_test_labels)


if __name__ == "__main__":
    main()