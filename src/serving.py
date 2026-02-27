from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel


REPO_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = REPO_ROOT / "models" / "champion_model.joblib"


app = FastAPI(title="ENDES Diabetes Risk API", version="1.0.0")


class PredictRequest(BaseModel):
    # Accept any feature dict (we validate columns at runtime)
    features: Dict[str, Any]


class PredictResponse(BaseModel):
    prediction: int
    probability: float
    model_path: str


_model = None  # loaded lazily


def load_model():
    global _model
    if _model is None:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(f"Model not found at: {MODEL_PATH}")
        _model = joblib.load(MODEL_PATH)
    return _model


@app.get("/health")
def health():
    return {"status": "ok", "model_exists": MODEL_PATH.exists()}


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    try:
        model = load_model()
    except FileNotFoundError as e:
        raise HTTPException(status_code=500, detail=str(e))

    # Model expects a DataFrame with feature columns
    X = pd.DataFrame([req.features])

    # Predict class
    try:
        pred = int(model.predict(X)[0])
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Prediction failed. Check feature names/types. Error: {e}",
        )

    # Predict probability if available
    proba = None
    if hasattr(model, "predict_proba"):
        try:
            proba = float(model.predict_proba(X)[:, 1][0])
        except Exception:
            proba = None

    if proba is None:
        proba = float(pred)

    return PredictResponse(
        prediction=pred,
        probability=proba,
        model_path=str(MODEL_PATH),
    )
