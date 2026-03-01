import argparse
import warnings
warnings.filterwarnings('ignore')
import json
from pathlib import Path

import pandas as pd
import numpy as np
import yaml
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import joblib
import mlflow
import mlflow.sklearn


def load_params():
    with open('params.yaml', 'r') as f:
        params = yaml.safe_load(f)
    return params['train']


def load_data(input_dir):
    print(f"Loading preprocessed data from {input_dir}...")
    X_train = pd.read_csv(Path(input_dir) / 'train_features.csv')
    y_train = pd.read_csv(Path(input_dir) / 'train_target.csv').squeeze()
    X_test = pd.read_csv(Path(input_dir) / 'test_features.csv')
    y_test = pd.read_csv(Path(input_dir) / 'test_target.csv').squeeze()

    print(f"Train set: {X_train.shape[0]} samples, {X_train.shape[1]} features")
    print(f"Test set: {X_test.shape[0]} samples")

    return X_train, y_train, X_test, y_test


def train_model(X_train, y_train, model_type, **model_params):
    print(f"\nTraining {model_type} model...")

    if model_type == 'random_forest':
        model = RandomForestClassifier(
            random_state=42,
            **model_params
        )
    elif model_type == 'logistic_regression':
        model = LogisticRegression(
            random_state=42,
            max_iter=1000,
            **model_params
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    model.fit(X_train, y_train)
    print("Training completed")
    return model


def evaluate_model(model, X_test, y_test):
    print("\nEvaluating model...")

    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None

    metrics = {
        'accuracy': float(accuracy_score(y_test, y_pred)),
        'precision': float(precision_score(y_test, y_pred, zero_division=0)),
        'recall': float(recall_score(y_test, y_pred, zero_division=0)),
        'f1_score': float(f1_score(y_test, y_pred, zero_division=0))
    }

    if y_pred_proba is not None:
        metrics['roc_auc'] = float(roc_auc_score(y_test, y_pred_proba))

    print("\nMetrics:")
    for metric_name, value in metrics.items():
        print(f"  {metric_name}: {value:.4f}")

    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-dir', type=str, default='data/processed')
    parser.add_argument('--model-path', type=str, default='models/model.pkl')
    parser.add_argument('--metrics-path', type=str, default='metrics.json')
    args = parser.parse_args()

    params = load_params()
    model_type = params['model_type']

    print(f"Parameters: {params}")

    X_train, y_train, X_test, y_test = load_data(args.input_dir)

    model_params = {}
    if model_type == 'random_forest':
        model_params['n_estimators'] = params.get('n_estimators', 100)
        model_params['min_samples_split'] = params.get('min_samples_split', 2)
        if params.get('max_depth') is not None:
            model_params['max_depth'] = params['max_depth']
    elif model_type == 'logistic_regression':
        model_params['C'] = params.get('C', 1.0)

    mlflow.set_experiment("weather_prediction")

    with mlflow.start_run():
        mlflow.log_param("model_type", model_type)
        for key, value in model_params.items():
            mlflow.log_param(key, value)

        model = train_model(X_train, y_train, model_type, **model_params)

        metrics = evaluate_model(model, X_test, y_test)

        for metric_name, value in metrics.items():
            mlflow.log_metric(metric_name, value)

        Path(args.model_path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, args.model_path)
        print(f"\nModel saved to {args.model_path}")

        with open(args.metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        print(f"Metrics saved to {args.metrics_path}")

        mlflow.sklearn.log_model(model, "model")

        print("\nExperiment logged to MLflow successfully!")
        print(f"Run ID: {mlflow.active_run().info.run_id}")


if __name__ == "__main__":
    main()
