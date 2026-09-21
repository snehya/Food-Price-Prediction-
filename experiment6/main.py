from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

APP_DIR = Path(__file__).resolve().parent
MODEL_PATH = APP_DIR / "best_model.pkl"

FEATURES = [
    "Tomato_Lag1",
    "Tomato_Lag2",
    "Tomato_PastRollingMean3",
    "Month",
    "Quarter",
    "Week",
    "Month_Sin",
    "Month_Cos",
]

app = FastAPI(
    title="Essential Food Item Inflation Forecaster API",
    description="FastAPI service for Tomato retail-price prediction using the tuned Random Forest model from Experiment 4.",
    version="1.0.0",
)

try:
    model = joblib.load(MODEL_PATH)
except Exception as exc:
    model = None
    MODEL_LOAD_ERROR = str(exc)


class PredictionRequest(BaseModel):
    Tomato_Lag1: float = Field(..., description="Tomato price at the previous observation")
    Tomato_Lag2: float = Field(..., description="Tomato price two observations earlier")
    Tomato_PastRollingMean3: float = Field(..., description="Mean of the previous three Tomato prices")
    Month: int = Field(..., ge=1, le=12)
    Quarter: int = Field(..., ge=1, le=4)
    Week: int = Field(..., ge=1, le=53)
    Month_Sin: float
    Month_Cos: float


@app.get("/")
def root():
    return {
        "message": "Essential Food Item Inflation Forecaster API",
        "target": "Tomato retail price",
        "problem_type": "Regression",
        "endpoint": "/predict",
    }


@app.get("/health")
def health():
    if model is None:
        return {"status": "error", "model_loaded": False, "error": MODEL_LOAD_ERROR}
    return {"status": "ok", "model_loaded": True}


@app.post("/predict")
def predict(request: PredictionRequest):
    if model is None:
        raise HTTPException(status_code=500, detail=f"Model could not be loaded: {MODEL_LOAD_ERROR}")

    input_df = pd.DataFrame(
        [[
            request.Tomato_Lag1,
            request.Tomato_Lag2,
            request.Tomato_PastRollingMean3,
            request.Month,
            request.Quarter,
            request.Week,
            request.Month_Sin,
            request.Month_Cos,
        ]],
        columns=FEATURES,
    )

    try:
        prediction = float(model.predict(input_df)[0])
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Prediction failed: {exc}")

    return {
        "predicted_tomato_price": round(prediction, 4),
        "unit": "INR/kg",
        "model": "Tuned Random Forest Regressor",
        "features_used": FEATURES,
    }
