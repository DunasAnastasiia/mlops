from pathlib import Path
import pandas as pd

PROCESSED_DATA_DIR = Path("data/processed")


def test_processed_files_exist():
    assert (PROCESSED_DATA_DIR / "train_features.csv").exists()
    assert (PROCESSED_DATA_DIR / "train_target.csv").exists()
    assert (PROCESSED_DATA_DIR / "test_features.csv").exists()
    assert (PROCESSED_DATA_DIR / "test_target.csv").exists()


def test_data_structure():
    x_train = pd.read_csv(PROCESSED_DATA_DIR / "train_features.csv")
    y_train = pd.read_csv(PROCESSED_DATA_DIR / "train_target.csv")

    assert x_train.shape[0] == y_train.shape[0]

    assert x_train.isnull().sum().sum() == 0
    assert y_train.isnull().sum().sum() == 0


def test_feature_columns():
    x_train = pd.read_csv(PROCESSED_DATA_DIR / "train_features.csv")
    assert "Location" in x_train.columns
    assert "MinTemp" in x_train.columns
    assert "MaxTemp" in x_train.columns
