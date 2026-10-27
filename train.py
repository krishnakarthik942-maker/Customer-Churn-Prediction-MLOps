from pathlib import Path
import argparse
import json
import warnings

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier

warnings.filterwarnings("ignore", category=UserWarning)

# ============================================================
# CONFIGURATION
# ============================================================

ROOT_DIR = Path(__file__).resolve().parent

DATA_PATH = ROOT_DIR / "data" / "customers.csv"
ARTIFACTS_DIR = ROOT_DIR / "artifacts"

MODEL_PATH = ARTIFACTS_DIR / "model.joblib"
METRICS_PATH = ARTIFACTS_DIR / "metrics.json"

# Use file-based tracking to avoid SQLite migration problems.
MLFLOW_DIR = ROOT_DIR / "mlruns"
EXPERIMENT_NAME = "customer-churn-prediction"

TARGET_COLUMN = "churn"

NUMERIC_FEATURES = [
    "tenure",
    "monthly_charges",
    "support_calls",
]

CATEGORICAL_FEATURES = [
    "contract",
    "internet_service",
]

FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

TEST_SIZE = 0.20
RANDOM_STATE = 42
MINIMUM_AUC = 0.60


# ============================================================
# DATA VALIDATION
# ============================================================

def load_and_validate_data(path: Path) -> pd.DataFrame:
    """Load and validate the customer churn dataset."""

    print(f"\n[1/6] Loading dataset: {path}")

    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {path}\n"
            "Expected location: data/customers.csv"
        )

    df = pd.read_csv(path)

    if df.empty:
        raise ValueError("The dataset is empty.")

    df.columns = df.columns.str.strip()

    required_columns = FEATURES + [TARGET_COLUMN]

    missing_columns = [
        col for col in required_columns if col not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}\n"
            f"Available columns: {df.columns.tolist()}"
        )

    # Normalize the target into binary labels.
    target = df[TARGET_COLUMN]

    if target.dtype == object or isinstance(target.dtype, pd.StringDtype):
        normalized = target.astype("string").str.strip().str.lower()

        target_mapping = {
            "yes": 1,
            "no": 0,
            "true": 1,
            "false": 0,
            "1": 1,
            "0": 0,
            "churn": 1,
            "not churn": 0,
            "churned": 1,
            "retained": 0,
        }

        mapped = normalized.map(target_mapping)

        if mapped.isna().any():
            unexpected = normalized[mapped.isna()].unique().tolist()
            raise ValueError(
                f"Unsupported churn labels: {unexpected}"
            )

        df[TARGET_COLUMN] = mapped.astype(int)

    else:
        if target.isna().any():
            raise ValueError(
                "The churn target contains missing values."
            )

        if not set(target.unique()).issubset({0, 1, 0.0, 1.0}):
            raise ValueError(
                "The churn target must contain only 0 and 1."
            )

        df[TARGET_COLUMN] = target.astype(int)

    # Convert numeric values; the pipeline will impute missing features.
    for col in NUMERIC_FEATURES:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Normalize blank categorical values to missing.
    for col in CATEGORICAL_FEATURES:
        df[col] = df[col].replace(r"^\s*$", np.nan, regex=True)

    # Rows without a target cannot be used for supervised training.
    df = df.dropna(subset=[TARGET_COLUMN]).copy()

    if df[TARGET_COLUMN].nunique() != 2:
        raise ValueError(
            "The dataset must contain both churn classes: 0 and 1."
        )

    class_counts = df[TARGET_COLUMN].value_counts()

    if class_counts.min() < 2:
        raise ValueError(
            "Each churn class needs at least two rows."
        )

    print(f"Dataset shape: {df.shape}")
    print(f"Churn distribution:\n{class_counts}")
    print("Data validation passed.")

    return df


# ============================================================
# BUILD MODEL
# ============================================================

def build_model() -> Pipeline:
    """Build preprocessing and Random Forest classifier."""

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="most_frequent"),
            ),
            (
                "encoder",
                OneHotEncoder(handle_unknown="ignore"),
            ),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, NUMERIC_FEATURES),
            (
                "categorical",
                categorical_pipeline,
                CATEGORICAL_FEATURES,
            ),
        ]
    )

    classifier = RandomForestClassifier(
        n_estimators=200,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", classifier),
        ]
    )


# ============================================================
# MAIN TRAINING PIPELINE
# ============================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train and evaluate the customer churn model."
    )

    parser.add_argument(
        "--check-quality-gate",
        action="store_true",
        help="Fail if ROC-AUC is below the configured threshold.",
    )

    args = parser.parse_args()

    print("=" * 60)
    print("CUSTOMER CHURN - MLOPS TRAINING PIPELINE")
    print("=" * 60)

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    MLFLOW_DIR.mkdir(parents=True, exist_ok=True)

    # Step 1: Load and validate data.
    df = load_and_validate_data(DATA_PATH)

    X = df[FEATURES].copy()
    y = df[TARGET_COLUMN].astype(int)

    # Step 2: Split dataset.
    print("\n[2/6] Splitting dataset...")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    print(f"Training rows: {len(X_train)}")
    print(f"Testing rows: {len(X_test)}")

    # Step 3: Configure MLflow.
    print("\n[3/6] Configuring MLflow...")

    mlflow.set_tracking_uri(MLFLOW_DIR.as_uri())
    mlflow.set_experiment(EXPERIMENT_NAME)

    model = build_model()

    # Step 4: Train, evaluate, and log.
    print("\n[4/6] Training Random Forest model...")

    with mlflow.start_run(run_name="random-forest-churn") as run:
        model.fit(X_train, y_train)

        predictions = model.predict(X_test)
        probabilities = model.predict_proba(X_test)[:, 1]

        auc_score = float(
            roc_auc_score(y_test, probabilities)
        )

        metrics = {
            "accuracy": float(
                accuracy_score(y_test, predictions)
            ),
            "precision": float(
                precision_score(
                    y_test,
                    predictions,
                    zero_division=0,
                )
            ),
            "recall": float(
                recall_score(
                    y_test,
                    predictions,
                    zero_division=0,
                )
            ),
            "f1_score": float(
                f1_score(
                    y_test,
                    predictions,
                    zero_division=0,
                )
            ),
            "auc": auc_score,
            "minimum_auc": MINIMUM_AUC,
            "training_rows": int(len(X_train)),
            "testing_rows": int(len(X_test)),
            "random_state": RANDOM_STATE,
        }

        mlflow.log_params(
            {
                "model_type": "RandomForestClassifier",
                "n_estimators": 200,
                "class_weight": "balanced",
                "test_size": TEST_SIZE,
                "random_state": RANDOM_STATE,
                "feature_count": len(FEATURES),
            }
        )

        mlflow.log_metrics(
            {
                key: value
                for key, value in metrics.items()
                if key in {
                    "accuracy",
                    "precision",
                    "recall",
                    "f1_score",
                    "auc",
                }
            }
        )

        # Save the model and metrics locally.
        joblib.dump(model, MODEL_PATH)

        with open(METRICS_PATH, "w", encoding="utf-8") as file:
            json.dump(metrics, file, indent=4)

        # MLflow 2.17.2 uses artifact_path, not name.
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="customer-churn-model",
            input_example=X_train.head(3),
            serialization_format="pickle",
        )

        mlflow.log_artifact(
            str(METRICS_PATH),
            artifact_path="evaluation",
        )

        mlflow.log_artifact(
            str(MODEL_PATH),
            artifact_path="artifacts",
        )

        print(f"MLflow run ID: {run.info.run_id}")

    # Step 5: Display evaluation results.
    print("\n[5/6] Evaluation results")
    print("-" * 45)

    for name, value in metrics.items():
        print(f"{name}: {value}")

    print("\nClassification report:")
    print(
        classification_report(
            y_test,
            predictions,
            labels=[0, 1],
            target_names=["Not Churn", "Churn"],
            zero_division=0,
        )
    )

    print("Confusion matrix:")
    print(
        confusion_matrix(
            y_test,
            predictions,
            labels=[0, 1],
        )
    )

    # Step 6: Check artifacts and quality gate.
    print("\n[6/6] Training completed.")
    print(f"Model saved:   {MODEL_PATH}")
    print(f"Metrics saved: {METRICS_PATH}")
    print(f"MLflow runs:   {MLFLOW_DIR}")

    if args.check_quality_gate:
        if auc_score < MINIMUM_AUC:
            raise SystemExit(
                f"QUALITY GATE FAILED: AUC {auc_score:.4f} "
                f"is below {MINIMUM_AUC:.2f}."
            )

        print(
            f"\nQUALITY GATE PASSED: AUC {auc_score:.4f} "
            f">= {MINIMUM_AUC:.2f}"
        )
    else:
        print(
            f"\nROC-AUC: {auc_score:.4f}. "
            "Run with --check-quality-gate to enforce the threshold."
        )


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"\nPIPELINE FAILED: {exc}")
        raise