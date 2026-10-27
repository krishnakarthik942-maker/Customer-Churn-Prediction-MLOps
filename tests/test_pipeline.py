
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest

from validation import validate_data


DATA_PATH = Path("data/customers.csv")
MODEL_PATH = Path("artifacts/model.joblib")
METRICS_PATH = Path("artifacts/metrics.json")


def test_data_is_valid():
    df = validate_data(DATA_PATH)

    assert len(df) > 0
    assert "churn" in df.columns


def test_missing_values_are_rejected(tmp_path):
    df = pd.read_csv(DATA_PATH)

    df.loc[0, "tenure"] = np.nan

    bad_file = tmp_path / "bad_customers.csv"
    df.to_csv(bad_file, index=False)

    with pytest.raises(ValueError, match="(?i)missing values"):
        validate_data(bad_file)


def test_model_and_metrics_exist():
    assert MODEL_PATH.exists(), "Trained model file is missing"
    assert METRICS_PATH.exists(), "Metrics file is missing"


def test_model_prediction_shape():
    assert MODEL_PATH.exists(), "Run train.py before testing the model"

    model = joblib.load(MODEL_PATH)

    sample = pd.DataFrame(
        [
            {
                "tenure": 12,
                "monthly_charges": 80.0,
                "support_calls": 3,
                "contract": "Monthly",
                "internet_service": "Fiber",
            }
        ]
    )

    prediction = model.predict(sample)
    probability = model.predict_proba(sample)

    assert prediction.shape == (1,)
    assert prediction[0] in [0, 1]
    assert probability.shape == (1, 2)
    assert np.allclose(probability.sum(axis=1), 1.0)


def test_model_quality():
    assert METRICS_PATH.exists(), "Metrics file is missing"

    with METRICS_PATH.open("r", encoding="utf-8") as file:
        metrics = json.load(file)

    # Support both common metric naming conventions.
    auc = metrics.get("auc", metrics.get("roc_auc"))
    assert auc is not None, "AUC metric is missing from metrics.json"

    minimum_auc = metrics.get("minimum_auc", 0.60)

    assert 0.0 <= auc <= 1.0
    assert auc >= minimum_auc, (
        f"Model AUC {auc:.4f} is below the minimum "
        f"required AUC {minimum_auc:.4f}"
    )
