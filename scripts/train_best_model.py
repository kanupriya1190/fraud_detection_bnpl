"""
Train fraud classifiers on data/feature_matrix.{parquet,csv}, pick best by PR-AUC,
and write data/best_model.joblib + data/model_metrics.csv.

Mirrors notebooks/06_fraud_detection_ml.ipynb (same split, features, and hyperparameters).
Requires: pandas, scikit-learn, joblib, lightgbm; optional xgboost for full metrics table.
"""
from __future__ import annotations

import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

RANDOM_STATE = 42
DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def load_feature_matrix() -> pd.DataFrame:
    parquet_path = DATA_DIR / "feature_matrix.parquet"
    csv_path = DATA_DIR / "feature_matrix.csv"
    if parquet_path.exists():
        df = pd.read_parquet(parquet_path)
        print(f"Loaded feature matrix from {parquet_path} — {df.shape}")
        return df
    if csv_path.exists():
        df = pd.read_csv(csv_path)
        print(f"Loaded feature matrix from {csv_path} — {df.shape}")
        return df
    raise FileNotFoundError(
        "No feature matrix found. Build data/feature_matrix.parquet (or .csv) first — "
        "see notebooks/04_feature_engineering.ipynb."
    )


def evaluate_model(name: str, model, X_eval, y_true: np.ndarray) -> dict:
    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_eval)[:, 1]
    else:
        y_prob = model.decision_function(X_eval)

    pr_auc = average_precision_score(y_true, y_prob)
    roc_auc = roc_auc_score(y_true, y_prob)

    prec_arr, rec_arr, thr_arr = precision_recall_curve(y_true, y_prob)
    f1_arr = 2 * prec_arr * rec_arr / (prec_arr + rec_arr + 1e-8)
    best_idx = int(np.argmax(f1_arr))
    best_thr = thr_arr[min(best_idx, len(thr_arr) - 1)]

    y_pred = (y_prob >= best_thr).astype(int)

    return {
        "Model": name,
        "PR-AUC": pr_auc,
        "ROC-AUC": roc_auc,
        "Best Threshold": best_thr,
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "F1": f1_score(y_true, y_pred, zero_division=0),
    }


def main() -> None:
    try:
        from xgboost import XGBClassifier

        has_xgb = True
    except ImportError:
        has_xgb = False
        print("⚠ xgboost not installed — skipping XGBoost.")

    try:
        from lightgbm import LGBMClassifier

        has_lgb = True
    except ImportError:
        has_lgb = False
        raise SystemExit("lightgbm is required to save best_model.joblib (install lightgbm).")

    df = load_feature_matrix()
    if "is_fraud" not in df.columns:
        raise SystemExit("Feature matrix must include an `is_fraud` target column.")

    ID_AND_DATE_COLS = [
        "order_id", "customer_id", "merchant_id", "plan_id",
        "fingerprint_id", "installment_id", "device_id",
        "order_date", "start_date", "end_date", "due_date",
        "ip_address", "browser", "os", "device_type",
        "order_status", "plan_status", "status", "segment",
        "plan_type", "channel",
    ]

    date_col = None
    for col in ["order_date", "start_date"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")
            if df[col].notna().any():
                date_col = col
                break

    if date_col:
        df = df.sort_values(date_col).reset_index(drop=True)
    else:
        print("No date column found — using row order as proxy for time.")

    split_idx = int(len(df) * 0.75)
    train_df = df.iloc[:split_idx].copy()
    test_df = df.iloc[split_idx:].copy()
    print(f"Train: {len(train_df):,} rows  |  Test: {len(test_df):,} rows")

    drop_cols = [c for c in ID_AND_DATE_COLS if c in df.columns] + ["is_fraud"]
    feature_cols = [
        c for c in df.columns
        if c not in drop_cols and df[c].dtype in ["int64", "float64", "int32", "float32", "int8", "uint8"]
    ]

    X_train = train_df[feature_cols].copy()
    y_train = train_df["is_fraud"].values
    X_test = test_df[feature_cols].copy()
    y_test = test_df["is_fraud"].values

    for c in feature_cols:
        median_val = X_train[c].median()
        X_train[c] = X_train[c].fillna(median_val)
        X_test[c] = X_test[c].fillna(median_val)

    print(f"Features used ({len(feature_cols)})")

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    models: dict[str, tuple] = {}

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    lr = LogisticRegression(
        class_weight="balanced",
        max_iter=1000,
        random_state=RANDOM_STATE,
        solver="lbfgs",
    )
    lr_cv = cross_val_score(lr, X_train_scaled, y_train, cv=cv, scoring="average_precision")
    lr.fit(X_train_scaled, y_train)
    models["Logistic Regression"] = (lr, X_test_scaled)
    print(f"Logistic Regression — CV PR-AUC: {lr_cv.mean():.4f} ± {lr_cv.std():.4f}")

    rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=15,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    rf_cv = cross_val_score(rf, X_train, y_train, cv=cv, scoring="average_precision")
    rf.fit(X_train, y_train)
    models["Random Forest"] = (rf, X_test)
    print(f"Random Forest — CV PR-AUC: {rf_cv.mean():.4f} ± {rf_cv.std():.4f}")

    if has_xgb:
        n_neg = (y_train == 0).sum()
        n_pos = (y_train == 1).sum()
        xgb = XGBClassifier(
            scale_pos_weight=n_neg / max(n_pos, 1),
            n_estimators=300,
            max_depth=6,
            learning_rate=0.1,
            eval_metric="aucpr",
            use_label_encoder=False,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            verbosity=0,
        )
        xgb_cv = cross_val_score(xgb, X_train, y_train, cv=cv, scoring="average_precision")
        xgb.fit(X_train, y_train)
        models["XGBoost"] = (xgb, X_test)
        print(f"XGBoost — CV PR-AUC: {xgb_cv.mean():.4f} ± {xgb_cv.std():.4f}")

    lgb = LGBMClassifier(
        is_unbalance=True,
        n_estimators=300,
        max_depth=6,
        learning_rate=0.1,
        metric="average_precision",
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbose=-1,
    )
    lgb_cv = cross_val_score(lgb, X_train, y_train, cv=cv, scoring="average_precision")
    lgb.fit(X_train, y_train)
    models["LightGBM"] = (lgb, X_test)
    print(f"LightGBM — CV PR-AUC: {lgb_cv.mean():.4f} ± {lgb_cv.std():.4f}")

    results = {name: evaluate_model(name, model, X_eval, y_test) for name, (model, X_eval) in models.items()}
    metrics_df = pd.DataFrame(list(results.values())).set_index("Model")
    print(metrics_df.round(4))

    best_name = metrics_df["PR-AUC"].idxmax()
    best_model_obj = models[best_name][0]
    model_path = DATA_DIR / "best_model.joblib"
    metrics_path = DATA_DIR / "model_metrics.csv"
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_model_obj, model_path)
    metrics_df.reset_index().to_csv(metrics_path, index=False)
    print(f"\nSaved best model ({best_name}) to {model_path}")
    print(f"Saved metrics to {metrics_path}")


if __name__ == "__main__":
    main()
