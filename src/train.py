import json
import sys
import warnings
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import optuna
import pandas as pd
from hydra import compose, initialize
from omegaconf import OmegaConf
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

warnings.filterwarnings("ignore")


def load_data(input_dir):
    print(f"Loading preprocessed data from {input_dir}...")
    x_train = pd.read_csv(Path(input_dir) / "train_features.csv")
    y_train = pd.read_csv(Path(input_dir) / "train_target.csv").squeeze()
    x_test = pd.read_csv(Path(input_dir) / "test_features.csv")
    y_test = pd.read_csv(Path(input_dir) / "test_target.csv").squeeze()

    print(f"Train set: {x_train.shape[0]} samples, {x_train.shape[1]} features")
    print(f"Test set: {x_test.shape[0]} samples")

    return x_train, y_train, x_test, y_test


def train_model(x_train, y_train, model_type, random_state, **model_params):
    print(f"\nTraining {model_type} model...")

    if model_type == "random_forest":
        model = RandomForestClassifier(random_state=random_state, **model_params)
    elif model_type == "logistic_regression":
        model = LogisticRegression(random_state=random_state, max_iter=1000, **model_params)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    model.fit(x_train, y_train)
    print("Training completed")
    return model


def evaluate_model(model, x_test, y_test):
    print("\nEvaluating model...")

    y_pred = model.predict(x_test)
    y_pred_proba = model.predict_proba(x_test)[:, 1] if hasattr(model, "predict_proba") else None

    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "f1_score": float(f1_score(y_test, y_pred, zero_division=0)),
    }

    if y_pred_proba is not None:
        metrics["roc_auc"] = float(roc_auc_score(y_test, y_pred_proba))

    print("\nMetrics:")
    for metric_name, value in metrics.items():
        print(f"  {metric_name}: {value:.4f}")

    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    plt.title("Confusion Matrix")
    plt.savefig("confusion_matrix.png")
    plt.close()
    print("Confusion matrix saved to confusion_matrix.png")

    return metrics


def objective(trial, cfg, data):
    x_train, y_train, x_test, y_test = data
    model_type = cfg.model.name
    model_params = dict(cfg.model.params)

    if model_type == "random_forest":
        model_params["n_estimators"] = trial.suggest_int("n_estimators", 50, 200, step=50)
        model_params["max_depth"] = trial.suggest_int("max_depth", 5, 20)
    elif model_type == "logistic_regression":
        model_params["C"] = trial.suggest_float("C", 0.01, 10.0, log=True)

    with mlflow.start_run(nested=True):
        mlflow.log_params(model_params)
        model = train_model(x_train, y_train, model_type, cfg.seed, **model_params)
        metrics = evaluate_model(model, x_test, y_test)
        for metric_name, value in metrics.items():
            mlflow.log_metric(metric_name, value)
        return metrics.get("roc_auc", metrics["accuracy"])


def run_hpo(cfg, data):
    x_train, y_train, x_test, y_test = data
    print("\nStarting Hyperparameter Optimization (Optuna)...")
    with mlflow.start_run(run_name="Optuna_Study"):
        study = optuna.create_study(direction="maximize")
        study.optimize(lambda trial: objective(trial, cfg, data), n_trials=20)

        print("\nBest trial:")
        print(f"  Value: {study.best_value:.4f}")
        print(f"  Params: {study.best_params}")

        mlflow.log_params(study.best_params)
        mlflow.log_metric("best_roc_auc", study.best_value)

        print("\nTraining final model with best parameters...")
        best_params = dict(cfg.model.params)
        best_params.update(study.best_params)

        model = train_model(x_train, y_train, cfg.model.name, cfg.seed, **best_params)
        metrics = evaluate_model(model, x_test, y_test)

        for metric_name, value in metrics.items():
            mlflow.log_metric(f"final_{metric_name}", value)

        Path(cfg.model_path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, cfg.model_path)
        mlflow.sklearn.log_model(model, "model")


def run_training(cfg, data):
    x_train, y_train, x_test, y_test = data
    model_type = cfg.model.name
    model_params = OmegaConf.to_container(cfg.model.params, resolve=True)
    random_state = cfg.seed

    with mlflow.start_run():
        mlflow.log_param("model_type", model_type)
        mlflow.log_params(model_params)
        mlflow.log_param("seed", random_state)

        model = train_model(x_train, y_train, model_type, random_state, **model_params)
        metrics = evaluate_model(model, x_test, y_test)

        for metric_name, value in metrics.items():
            mlflow.log_metric(metric_name, value)

        Path(cfg.model_path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, cfg.model_path)

        with open(cfg.metrics_path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)

        mlflow.sklearn.log_model(model, "model")


def main():
    with initialize(config_path="../conf", version_base="1.2"):
        hpo_mode = "--hpo" in sys.argv
        overrides = [arg for arg in sys.argv[1:] if arg != "--hpo"]
        cfg = compose(config_name="config", overrides=overrides)
        print(f"Configuration:\n{OmegaConf.to_yaml(cfg)}")
        data = load_data(cfg.input_dir)
        mlflow.set_experiment("weather_prediction_hpo")
        if hpo_mode:
            run_hpo(cfg, data)
        else:
            run_training(cfg, data)
        print("\nExperiment logged to MLflow successfully!")


if __name__ == "__main__":
    main()
