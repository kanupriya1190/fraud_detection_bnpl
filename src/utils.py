"""Shared utilities for the BNPL fraud detection project."""

import os
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DB_PATH = DATA_DIR / "bnpl.db"
CLEAN_DB_PATH = DATA_DIR / "bnpl_clean.db"
FEATURE_MATRIX_PATH = DATA_DIR / "feature_matrix.csv"
MODEL_PATH = DATA_DIR / "best_model.joblib"
METRICS_PATH = DATA_DIR / "model_metrics.csv"


def get_db_path(prefer_clean: bool = True) -> Path:
    if prefer_clean and CLEAN_DB_PATH.exists():
        return CLEAN_DB_PATH
    return RAW_DB_PATH


def get_connection(prefer_clean: bool = True) -> sqlite3.Connection:
    return sqlite3.connect(str(get_db_path(prefer_clean)))


def load_table(table_name: str, conn: sqlite3.Connection = None, prefer_clean: bool = True) -> pd.DataFrame:
    close = False
    if conn is None:
        conn = get_connection(prefer_clean)
        close = True
    df = pd.read_sql(f"SELECT * FROM [{table_name}]", conn)
    if close:
        conn.close()
    return df


def clean_amount_column(series: pd.Series) -> pd.Series:
    """Strip dollar signs, convert to float, take abs, cap at 10000."""
    s = series.astype(str).str.replace("$", "", regex=False)
    s = pd.to_numeric(s, errors="coerce")
    s = s.abs()
    s = s.clip(upper=10_000)
    return s


def parse_dates_flexible(series: pd.Series) -> pd.Series:
    """Parse dates in mixed formats (ISO, US, verbose)."""
    return pd.to_datetime(series, format="mixed", dayfirst=False, errors="coerce")


CREDIT_STATES = {
    "CURRENT": 0,
    "DPD_1_30": 1,
    "DPD_31_60": 2,
    "DPD_61_90": 3,
    "DPD_90_PLUS": 4,
    "DEFAULT": 5,
    "PAID_OFF": 6,
}


def assign_credit_state(days_past_due: int, plan_status: str = None) -> str:
    if plan_status == "defaulted":
        return "DEFAULT"
    if plan_status == "completed":
        return "PAID_OFF"
    if days_past_due == 0:
        return "CURRENT"
    elif days_past_due <= 30:
        return "DPD_1_30"
    elif days_past_due <= 60:
        return "DPD_31_60"
    elif days_past_due <= 90:
        return "DPD_61_90"
    else:
        return "DPD_90_PLUS"
