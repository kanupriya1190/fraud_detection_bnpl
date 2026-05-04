from __future__ import annotations

import logging
import math
import time
from pathlib import Path
from typing import List, Optional

import numpy as np
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

MODEL_PATH = Path(__file__).resolve().parent.parent.parent / "data" / "best_model.joblib"
FRAUD_THRESHOLD = 0.5

MERCHANT_CATEGORIES = [
    "education", "electronics", "fashion", "gaming", "grocery",
    "health_beauty", "home_furniture", "jewelry_luxury", "sports_outdoor", "travel",
]
PLAN_TYPES = ["pay_in_4", "pay_in_6", "pay_in_12"]


class TransactionInput(BaseModel):
    amount: float = Field(250.0, ge=0)
    merchant_category: str = "electronics"
    plan_type: str = "pay_in_4"
    num_installments: int = Field(4, ge=1)
    credit_score: int = Field(700, ge=300, le=850)
    annual_income: float = Field(65000.0, ge=0)
    account_age_days: int = Field(180, ge=0)
    orders_last_30d: int = Field(1, ge=0)
    is_vpn: bool = False
    num_unique_devices: int = Field(1, ge=0)
    days_past_due_avg: float = Field(0.0, ge=0)
    historical_on_time_rate: float = Field(0.95, ge=0.0, le=1.0)
    num_prior_defaults: int = Field(0, ge=0)
    shared_device_count: int = Field(0, ge=0)


class PredictionOutput(BaseModel):
    fraud_probability: float
    is_fraud: bool
    confidence: str
    risk_level: str
    top_risk_factors: List[str]
    explanation: str
    processing_time_ms: float


class BatchInput(BaseModel):
    transactions: List[TransactionInput]


class BatchSummary(BaseModel):
    total: int
    flagged: int
    flag_rate: float


class BatchOutput(BaseModel):
    predictions: List[PredictionOutput]
    summary: BatchSummary


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    version: str


_model = None
_model_loaded = False


def _load_model() -> None:
    global _model, _model_loaded
    if not MODEL_PATH.exists():
        logger.warning("Model file not found at %s", MODEL_PATH)
        return
    try:
        import joblib
        _model = joblib.load(MODEL_PATH)
        _model_loaded = True
        logger.info("Loaded model from %s (%s)", MODEL_PATH, type(_model).__name__)
    except Exception:
        logger.exception("Failed to load model from %s", MODEL_PATH)
        _model = None
        _model_loaded = False


def _build_feature_dict(data: TransactionInput) -> dict[str, float]:
    features: dict[str, float] = {}

    features["amount"] = data.amount
    features["log_amount"] = math.log1p(data.amount)
    features["amount_per_installment"] = data.amount / max(data.num_installments, 1)
    features["num_installments"] = float(data.num_installments)
    features["credit_score"] = float(data.credit_score)
    features["annual_income"] = data.annual_income
    features["account_age_days"] = float(data.account_age_days)
    features["orders_last_30d"] = float(data.orders_last_30d)
    features["is_vpn"] = float(data.is_vpn)
    features["num_unique_devices"] = float(data.num_unique_devices)
    features["days_past_due_avg"] = data.days_past_due_avg
    features["historical_on_time_rate"] = data.historical_on_time_rate
    features["num_prior_defaults"] = float(data.num_prior_defaults)
    features["shared_device_count"] = float(data.shared_device_count)
    features["amount_to_income_ratio"] = data.amount / max(data.annual_income, 1.0)

    for cat in MERCHANT_CATEGORIES:
        features[f"merchant_category_{cat}"] = 1.0 if data.merchant_category == cat else 0.0

    for plan in PLAN_TYPES:
        features[f"plan_type_{plan}"] = 1.0 if data.plan_type == plan else 0.0

    return features


def _features_to_array(feature_dict: dict[str, float], feature_names: Optional[list[str]] = None) -> np.ndarray:
    if feature_names is not None:
        return np.array([[feature_dict.get(name, 0.0) for name in feature_names]])
    return np.array([[v for v in feature_dict.values()]])


def _identify_risk_factors(data: TransactionInput) -> list[str]:
    factors: list[tuple[float, str]] = []

    if data.is_vpn:
        factors.append((0.20, "vpn_usage"))
    if data.orders_last_30d > 5:
        factors.append((data.orders_last_30d * 0.03, "high_velocity"))
    if data.amount > 500:
        factors.append((data.amount / 1000, "high_amount"))
    if data.credit_score < 600:
        factors.append(((650 - data.credit_score) / 100, "low_credit_score"))
    if data.account_age_days < 30:
        factors.append(((30 - data.account_age_days) / 30, "new_account"))
    if data.num_prior_defaults > 0:
        factors.append((data.num_prior_defaults * 0.10, "prior_defaults"))
    if data.historical_on_time_rate < 0.8:
        factors.append((1 - data.historical_on_time_rate, "poor_payment_history"))
    if data.days_past_due_avg > 10:
        factors.append((data.days_past_due_avg / 30, "high_days_past_due"))
    if data.shared_device_count > 2:
        factors.append((data.shared_device_count * 0.05, "shared_devices"))
    if data.num_unique_devices > 3:
        factors.append((data.num_unique_devices * 0.03, "multiple_devices"))
    if data.amount / max(data.annual_income, 1) > 0.05:
        factors.append((data.amount / max(data.annual_income, 1), "high_amount_to_income"))

    factors.sort(key=lambda x: x[0], reverse=True)
    return [f[1] for f in factors[:5]]


def _confidence_label(prob: float) -> str:
    if prob < 0.1 or prob > 0.8:
        return "high"
    return "medium"


def _risk_level(prob: float) -> str:
    if prob > 0.7:
        return "high"
    if prob >= 0.3:
        return "medium"
    return "low"


def _build_explanation(is_fraud: bool, prob: float, risk_factors: list[str]) -> str:
    if is_fraud:
        base = f"High fraud risk detected ({prob:.0%} probability)."
    elif prob >= 0.3:
        base = f"Moderate fraud risk ({prob:.0%} probability). Review recommended."
    else:
        base = "Low fraud risk. Account has good payment history."

    if risk_factors:
        base += " Top factors: " + ", ".join(risk_factors) + "."
    return base


def _score_transaction(data: TransactionInput) -> PredictionOutput:
    start = time.perf_counter()

    if not _model_loaded or _model is None:
        elapsed = (time.perf_counter() - start) * 1000
        return PredictionOutput(
            fraud_probability=0.0,
            is_fraud=False,
            confidence="high",
            risk_level="low",
            top_risk_factors=[],
            explanation="Model not loaded. Run training notebook first.",
            processing_time_ms=round(elapsed, 2),
        )

    feature_dict = _build_feature_dict(data)

    model_features = getattr(_model, "feature_names_in_", None)
    if model_features is not None:
        feature_names = list(model_features)
    else:
        feature_names = None

    features = _features_to_array(feature_dict, feature_names)

    try:
        proba = _model.predict_proba(features)[0]
        fraud_prob = float(proba[1]) if len(proba) > 1 else float(proba[0])
    except Exception:
        logger.exception("Model prediction failed, returning error")
        elapsed = (time.perf_counter() - start) * 1000
        return PredictionOutput(
            fraud_probability=0.0,
            is_fraud=False,
            confidence="medium",
            risk_level="low",
            top_risk_factors=[],
            explanation="Prediction failed due to model error.",
            processing_time_ms=round(elapsed, 2),
        )

    is_fraud = fraud_prob >= FRAUD_THRESHOLD
    risk_factors = _identify_risk_factors(data)
    confidence = _confidence_label(fraud_prob)
    risk = _risk_level(fraud_prob)
    explanation = _build_explanation(is_fraud, fraud_prob, risk_factors)

    elapsed = (time.perf_counter() - start) * 1000

    return PredictionOutput(
        fraud_probability=round(fraud_prob, 4),
        is_fraud=is_fraud,
        confidence=confidence,
        risk_level=risk,
        top_risk_factors=risk_factors,
        explanation=explanation,
        processing_time_ms=round(elapsed, 2),
    )


app = FastAPI(title="BNPL Fraud Detection API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def startup() -> None:
    _load_model()


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    return HealthResponse(status="healthy", model_loaded=_model_loaded, version="1.0.0")


@app.post("/predict", response_model=PredictionOutput)
async def predict(data: TransactionInput) -> PredictionOutput:
    return _score_transaction(data)


@app.post("/predict/batch", response_model=BatchOutput)
async def predict_batch(batch: BatchInput) -> BatchOutput:
    predictions = [_score_transaction(txn) for txn in batch.transactions]
    total = len(predictions)
    flagged = sum(1 for p in predictions if p.is_fraud)
    return BatchOutput(
        predictions=predictions,
        summary=BatchSummary(
            total=total,
            flagged=flagged,
            flag_rate=round(flagged / total, 4) if total > 0 else 0.0,
        ),
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
