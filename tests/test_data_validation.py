import pandas as pd
from pathlib import Path
import pytest

PROCESSED_DATA_DIR = Path("data/processed")


def test_processed_files_exist():
    assert (PROCESSED_DATA_DIR / "train_features.csv").exists()
    assert (PROCESSED_DATA_DIR / "train_target.csv").exists()
    assert (PROCESSED_DATA_DIR / "test_features.csv").exists()
    assert (PROCESSED_DATA_DIR / "test_target.csv").exists()


def test_data_structure():
    X_train = pd.read_csv(PROCESSED_DATA_DIR / "train_features.csv")
    y_train = pd.read_csv(PROCESSED_DATA_DIR / "train_target.csv")

    assert X_train.shape[0] == y_train.shape[0]

    assert X_train.isnull().sum().sum() == 0
    assert y_train.isnull().sum().sum() == 0


def test_feature_columns():
    X_train = pd.read_csv(PROCESSED_DATA_DIR / "train_features.csv")
    assert "Location" in X_train.columns
    assert "MinTemp" in X_train.columns
    assert "MaxTemp" in X_train.columns
