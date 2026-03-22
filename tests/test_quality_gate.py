import json
from pathlib import Path

METRICS_PATH = Path("metrics.json")
F1_THRESHOLD = 0.5


def test_f1_quality_gate():
    with open(METRICS_PATH, "r", encoding="utf-8") as f:
        metrics = json.load(f)

    assert metrics["f1_score"] >= F1_THRESHOLD
