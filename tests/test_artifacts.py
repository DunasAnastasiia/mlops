from pathlib import Path
import json
import joblib

MODEL_PATH = Path("models/model.pkl")
METRICS_PATH = Path("metrics.json")
CM_PATH = Path("confusion_matrix.png")


def test_artifacts_exist():
    assert MODEL_PATH.exists()
    assert METRICS_PATH.exists()
    assert CM_PATH.exists()


def test_model_loadable():
    model = joblib.load(MODEL_PATH)
    assert model is not None
    assert hasattr(model, "predict")


def test_metrics_structure():
    with open(METRICS_PATH, "r") as f:
        metrics = json.load(f)

    expected_metrics = ["accuracy", "precision", "recall", "f1_score"]
    for metric in expected_metrics:
        assert metric in metrics
        assert isinstance(metrics[metric], (float, int))
