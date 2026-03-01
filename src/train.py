import os
import argparse
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import mlflow
import mlflow.sklearn


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


def train_model(X_train, y_train, model_type='random_forest', **model_params):
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
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, zero_division=0),
        'recall': recall_score(y_test, y_pred, zero_division=0),
        'f1_score': f1_score(y_test, y_pred, zero_division=0)
    }

    if y_pred_proba is not None:
        metrics['roc_auc'] = roc_auc_score(y_test, y_pred_proba)

    print("\nMetrics:")
    for metric_name, value in metrics.items():
        print(f"  {metric_name}: {value:.4f}")

    return metrics


def main():
    parser = argparse.ArgumentParser(description='Train weather prediction model with MLflow tracking')
    parser.add_argument('--data-path', type=str, default='data/raw/weatherAUS.csv',
                        help='Path to the dataset')
    parser.add_argument('--model-type', type=str, default='random_forest',
                        choices=['random_forest', 'logistic_regression'],
                        help='Type of model to train')
    parser.add_argument('--test-size', type=float, default=0.2,
                        help='Test set size (default: 0.2)')
    parser.add_argument('--random-state', type=int, default=42,
                        help='Random state for reproducibility')

    parser.add_argument('--n-estimators', type=int, default=100,
                        help='Number of trees for RandomForest')
    parser.add_argument('--max-depth', type=int, default=None,
                        help='Maximum depth for RandomForest')
    parser.add_argument('--min-samples-split', type=int, default=2,
                        help='Min samples split for RandomForest')
    parser.add_argument('--C', type=float, default=1.0,
                        help='Regularization parameter for LogisticRegression')

    args = parser.parse_args()

    mlflow.set_experiment("weather_prediction")

    with mlflow.start_run():

        mlflow.log_param("model_type", args.model_type)
        mlflow.log_param("test_size", args.test_size)
        mlflow.log_param("random_state", args.random_state)

        df = load_data(args.data_path)
        mlflow.log_param("dataset_rows", df.shape[0])
        mlflow.log_param("dataset_cols", df.shape[1])

        X, y = preprocess_data(df)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=args.test_size,
            random_state=args.random_state,
            stratify=y
        )

        print(f"\nTrain set: {X_train.shape[0]} samples")
        print(f"Test set: {X_test.shape[0]} samples")

        mlflow.log_param("train_samples", X_train.shape[0])
        mlflow.log_param("test_samples", X_test.shape[0])

        model_params = {}
        if args.model_type == 'random_forest':
            model_params['n_estimators'] = args.n_estimators
            model_params['min_samples_split'] = args.min_samples_split
            if args.max_depth is not None:
                model_params['max_depth'] = args.max_depth

            mlflow.log_param("n_estimators", args.n_estimators)
            mlflow.log_param("max_depth", args.max_depth)
            mlflow.log_param("min_samples_split", args.min_samples_split)

        elif args.model_type == 'logistic_regression':
            model_params['C'] = args.C
            mlflow.log_param("C", args.C)

        model = train_model(X_train, y_train, args.model_type, **model_params)

        metrics = evaluate_model(model, X_test, y_test)

        for metric_name, value in metrics.items():
            mlflow.log_metric(metric_name, value)

        mlflow.sklearn.log_model(model, "model")

        print("\nExperiment logged to MLflow successfully!")
        print(f"Run ID: {mlflow.active_run().info.run_id}")


if __name__ == "__main__":
    main()
