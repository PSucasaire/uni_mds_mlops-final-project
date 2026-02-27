from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier


def load_training_data(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Training dataset not found: {path}")
    return pd.read_parquet(path)


def split_xy(df: pd.DataFrame, target: str = "diabetes"):
    if target not in df.columns:
        raise KeyError(f"Target column '{target}' not found.")
    y = df[target].astype(int)
    X = df.drop(columns=[target])
    return X, y


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    # Detect numeric vs categorical
    numeric_features = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])]
    categorical_features = [c for c in X.columns if c not in numeric_features]

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ],
        remainder="drop",
    )
    return preprocessor


def evaluate(y_true, y_pred, y_proba) -> dict:
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }
    # ROC-AUC needs probabilities and both classes present
    try:
        metrics["roc_auc"] = float(roc_auc_score(y_true, y_proba))
    except Exception:
        metrics["roc_auc"] = None
    return metrics


def main(
    data_path: str,
    out_dir: str,
    test_size: float,
    random_state: int,
) -> None:
    data_path = Path(data_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = load_training_data(data_path)
    X, y = split_xy(df, target="diabetes")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )

    preprocessor = build_preprocessor(X_train)

    # Logistic Regression pipeline
    logreg = Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("model", LogisticRegression(
                max_iter=2000,
                class_weight="balanced",
                n_jobs=None
            )),
        ]
    )

    # Random Forest pipeline
    rf = Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("model", RandomForestClassifier(
                n_estimators=300,
                random_state=random_state,
                class_weight="balanced_subsample",
                n_jobs=-1
            )),
        ]
    )

    models = {
        "logistic_regression": logreg,
        "random_forest": rf,
    }

    results = {}
    fitted = {}

    for name, pipe in models.items():
        pipe.fit(X_train, y_train)

        y_pred = pipe.predict(X_test)
        # proba for class 1
        y_proba = pipe.predict_proba(X_test)[:, 1] if hasattr(pipe, "predict_proba") else None

        results[name] = evaluate(y_test, y_pred, y_proba)
        fitted[name] = pipe

    # Choose champion: highest F1; tie-breaker ROC-AUC
    def score_key(m):
        f1 = m["f1"]
        auc = m["roc_auc"] if m["roc_auc"] is not None else -1.0
        return (f1, auc)

    champion_name = max(results.keys(), key=lambda k: score_key(results[k]))
    champion_model = fitted[champion_name]

    # Save artifacts
    model_path = out_dir / "champion_model.joblib"
    joblib.dump(champion_model, model_path)

    metrics_path = out_dir / "metrics.json"
    payload = {
        "champion": champion_name,
        "metrics": results,
        "data_path": str(data_path),
        "n_rows": int(df.shape[0]),
        "n_features": int(df.shape[1] - 1),
        "test_size": test_size,
        "random_state": random_state,
    }
    metrics_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print("✅ Training completed")
    print("Champion:", champion_name)
    print("Saved model:", model_path)
    print("Saved metrics:", metrics_path)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and select champion model for ENDES diabetes project.")
    parser.add_argument(
        "--data-path",
        type=str,
        default="data/training/endes_diabetes_features.parquet",
        help="Path to training dataset (parquet).",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="models",
        help="Output directory for model and metrics.",
    )
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()

    main(
        data_path=args.data_path,
        out_dir=args.out_dir,
        test_size=args.test_size,
        random_state=args.random_state,
    )
