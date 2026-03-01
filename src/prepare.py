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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, default='data/raw/weatherAUS.csv')
    parser.add_argument('--output-dir', type=str, default='data/processed')
    args = parser.parse_args()

    params = load_params()
    test_size = params['test_size']
    random_state = params['random_state']

    print(f"Parameters: test_size={test_size}, random_state={random_state}")

    df = load_data(args.input)

    X, y = preprocess_data(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )

    print(f"\nTrain set: {X_train.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    X_train.to_csv(output_dir / 'train_features.csv', index=False)
    y_train.to_csv(output_dir / 'train_target.csv', index=False, header=True)
    X_test.to_csv(output_dir / 'test_features.csv', index=False)
    y_test.to_csv(output_dir / 'test_target.csv', index=False, header=True)

    print(f"\nProcessed data saved to {args.output_dir}/")


if __name__ == "__main__":
    main()
