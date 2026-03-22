import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from hydra import compose, initialize
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

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
    x_data = data.drop(columns=[target_column])
    if "Date" in x_data.columns:
        x_data = x_data.drop(columns=["Date"])
    numerical_cols = x_data.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = x_data.select_dtypes(include=["object"]).columns.tolist()
    if numerical_cols:
        imputer_num = SimpleImputer(strategy="median")
        x_data[numerical_cols] = imputer_num.fit_transform(x_data[numerical_cols])
    if categorical_cols:
        imputer_cat = SimpleImputer(strategy="most_frequent")
        x_data[categorical_cols] = imputer_cat.fit_transform(x_data[categorical_cols])
        for col in categorical_cols:
            le = LabelEncoder()
            x_data[col] = le.fit_transform(x_data[col].astype(str))
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x_data)
    x_scaled = pd.DataFrame(x_scaled, columns=x_data.columns, index=x_data.index)
    return x_scaled, y


def main():
    with initialize(config_path="../conf", version_base="1.2"):
        overrides = sys.argv[1:]
        cfg = compose(config_name="config", overrides=overrides)
        test_size = cfg.prepare.test_size
        random_state = cfg.prepare.random_state
        input_path = "data/raw/weatherAUS.csv"
        output_dir_path = "data/processed"
        df = load_data(input_path)
    x_proc, y_proc = preprocess_data(df)
    x_train, x_test, y_train, y_test = train_test_split(
        x_proc, y_proc, test_size=test_size, random_state=random_state, stratify=y_proc
    )
    output_dir = Path(output_dir_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    x_train.to_csv(output_dir / "train_features.csv", index=False)
    y_train.to_csv(output_dir / "train_target.csv", index=False, header=True)
    x_test.to_csv(output_dir / "test_features.csv", index=False)
    y_test.to_csv(output_dir / "test_target.csv", index=False, header=True)


if __name__ == "__main__":
    main()
