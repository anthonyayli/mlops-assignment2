#!/usr/bin/env python3
import argparse
import pandas as pd
import numpy as np
import os
import warnings

warnings.filterwarnings("ignore")


def load_csv(file_path: str, **kwargs) -> pd.DataFrame:
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"CSV file not found: {file_path}")
    
    return pd.read_csv(file_path, **kwargs)


def save_csv(df: pd.DataFrame, file_path: str, **kwargs) -> None:
    df.to_csv(file_path, index=False, **kwargs)
    print(f"DataFrame saved to: {file_path}")


def family_size(number):
    if number == 1:
        return "Alone"
    elif number > 1 and number < 5:
        return "Small"
    else:
        return "Large"


def preprocess_data(train_path: str, test_path: str) -> pd.DataFrame:
    train = load_csv(train_path)
    test = load_csv(test_path)
    
    train.drop(columns=['Cabin'], inplace=True)
    test.drop(columns=['Cabin'], inplace=True)
    
    train['Embarked'].fillna('S', inplace=True)
    test['Fare'].fillna(test['Fare'].mean(), inplace=True)
    
    df = pd.concat([train, test], sort=True).reset_index(drop=True)
    
    df['Age'] = df.groupby(['Sex', 'Pclass'])['Age'].transform(lambda x: x.fillna(x.median()))
    
    df['Title'] = df['Name'].str.split(", ", expand=True)[1].str.split(".", expand=True)[0]
    df['Age'] = df['Age'].astype('int64')
    
    df['Title'] = df['Title'].replace(['Lady', 'the Countess', 'Capt', 'Col', 'Don', 'Dr', 
                                       'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
    df['Title'] = df['Title'].replace('Mlle', 'Miss')
    df['Title'] = df['Title'].replace('Ms', 'Miss')
    df['Title'] = df['Title'].replace('Mme', 'Mrs')
    
    df['Family_size'] = df['SibSp'] + df['Parch'] + 1
    df.drop(columns=['Name', 'Parch', 'SibSp', 'Ticket'], inplace=True)
    df['Family_size'] = df['Family_size'].apply(family_size)
    
    return df




def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Preprocess Titanic dataset")
    p.add_argument("--input-train", required=True, help="Path to training data CSV file")
    p.add_argument("--input-test", required=True, help="Path to test data CSV file")
    p.add_argument("--output", required=True, help="Path to save processed data")
    return p


def main():
    args = build_parser().parse_args()
    
    df_processed = preprocess_data(args.input_train, args.input_test)
    
    save_csv(df_processed, args.output)


if __name__ == "__main__":
    main()