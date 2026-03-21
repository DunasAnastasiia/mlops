import argparse
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import yaml
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer


def load_params():
    with open('params.yaml', 'r') as f:
        params = yaml.safe_load(f)
    return params['prepare']


def load_data(data_path):
    print(f"Loading data from {data_path}...")
    df = pd.read_csv(data_path)
    print(f"Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")
    return df


def preprocess_data(df, target_column='RainTomorrow'):
    print("\nPreprocessing data...")

    data = df.copy()

    data = data.dropna(subset=[target_column])
    print(f"After dropping missing target: {data.shape[0]} rows")

    y = data[target_column].map({'Yes': 1, 'No': 0})
    X = data.drop(columns=[target_column])

    if 'Date' in X.columns:
        X = X.drop(columns=['Date'])

    numerical_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = X.select_dtypes(include=['object']).columns.tolist()

    print(f"Numerical features: {len(numerical_cols)}")
    print(f"Categorical features: {len(categorical_cols)}")

    if numerical_cols:
        imputer_num = SimpleImputer(strategy='median')
        X[numerical_cols] = imputer_num.fit_transform(X[numerical_cols])

    if categorical_cols:
        imputer_cat = SimpleImputer(strategy='most_frequent')
        X[categorical_cols] = imputer_cat.fit_transform(X[categorical_cols])

        for col in categorical_cols:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled = pd.DataFrame(X_scaled, columns=X.columns, index=X.index)

    print("Preprocessing completed")
    return X_scaled, y


import sys
from omegaconf import OmegaConf
from hydra import compose, initialize

def main():
    with initialize(config_path="../conf", version_base="1.2"):
        overrides = sys.argv[1:]
        cfg = compose(config_name="config", overrides=overrides)
        
        print(f"Configuration:\n{OmegaConf.to_yaml(cfg)}")

        test_size = cfg.prepare.test_size
        random_state = cfg.prepare.random_state

        print(f"Parameters: test_size={test_size}, random_state={random_state}")

        # Ми можемо додати аргументи для input/output через Hydra, але зараз використаємо дефолтні або з cfg
        input_path = 'data/raw/weatherAUS.csv'
        output_dir_path = 'data/processed'

        df = load_data(input_path)

    X, y = preprocess_data(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )

    print(f"\nTrain set: {X_train.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")

    output_dir = Path(output_dir_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    X_train.to_csv(output_dir / 'train_features.csv', index=False)
    y_train.to_csv(output_dir / 'train_target.csv', index=False, header=True)
    X_test.to_csv(output_dir / 'test_features.csv', index=False)
    y_test.to_csv(output_dir / 'test_target.csv', index=False, header=True)

    print(f"\nProcessed data saved to {output_dir_path}/")


if __name__ == "__main__":
    main()
