import warnings
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
import sys
from hydra import compose, initialize

warnings.filterwarnings("ignore")


def load_data(data_path):
    return pd.read_csv(data_path)


def preprocess_data(df, target_column="RainTomorrow"):
    if df.empty:
        raise ValueError("Dataset is empty")
    if target_column not in df.columns:
        raise ValueError(f"Column {target_column} not found in dataset")
    data = df.copy()
    data = data.dropna(subset=[target_column])
    if data.empty:
        raise ValueError("Dataset is empty after dropping missing target values")
    y = data[target_column].map({"Yes": 1, "No": 0})
    X = data.drop(columns=[target_column])
    if "Date" in X.columns:
        X = X.drop(columns=["Date"])
    numerical_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = X.select_dtypes(include=["object"]).columns.tolist()
    if numerical_cols:
        imputer_num = SimpleImputer(strategy="median")
        X[numerical_cols] = imputer_num.fit_transform(X[numerical_cols])
    if categorical_cols:
        imputer_cat = SimpleImputer(strategy="most_frequent")
        X[categorical_cols] = imputer_cat.fit_transform(X[categorical_cols])
        for col in categorical_cols:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled = pd.DataFrame(X_scaled, columns=X.columns, index=X.index)
    return X_scaled, y


def main():
    with initialize(config_path="../conf", version_base="1.2"):
        overrides = sys.argv[1:]
        cfg = compose(config_name="config", overrides=overrides)
        test_size = cfg.prepare.test_size
        random_state = cfg.prepare.random_state
        input_path = "data/raw/weatherAUS.csv"
        output_dir_path = "data/processed"
        df = load_data(input_path)
    X, y = preprocess_data(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    output_dir = Path(output_dir_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    X_train.to_csv(output_dir / "train_features.csv", index=False)
    y_train.to_csv(output_dir / "train_target.csv", index=False, header=True)
    X_test.to_csv(output_dir / "test_features.csv", index=False)
    y_test.to_csv(output_dir / "test_target.csv", index=False, header=True)


if __name__ == "__main__":
    main()
