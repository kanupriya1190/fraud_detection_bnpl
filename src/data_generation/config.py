"""
Configuration for BNPL synthetic data generation.

All distribution parameters, customer segments, fraud patterns,
merchant categories, and data quality issue injection rates.
"""

from dataclasses import dataclass, field
from datetime import date

# ─────────────────────────────────────────────
# Global Parameters
# ─────────────────────────────────────────────

SEED = 42
DATA_START_DATE = date(2022, 1, 1)
DATA_END_DATE = date(2024, 6, 30)   # 2.5 years of data

NUM_CUSTOMERS = 50_000
NUM_MERCHANTS = 500
TARGET_FRAUD_RATE = 0.06  # ~6% of orders labeled fraud (probabilistic per-order labeling)

# ─────────────────────────────────────────────
# Customer Segments
# ─────────────────────────────────────────────

CUSTOMER_SEGMENTS = {
    "good_payer": {
        "weight": 0.60,
        "credit_score_range": (640, 850),
        "income_range": (35_000, 150_000),
        "on_time_prob": 0.90,
        "default_prob": 0.02,
        "orders_range": (1, 5),
        "plan_preference": {"pay_in_4": 0.55, "pay_in_6": 0.28, "pay_in_12": 0.17},
        "order_amount_range": (50, 900),
        "dispute_prob": 0.02,
    },
    "occasional_late": {
        "weight": 0.20,
        "credit_score_range": (550, 730),
        "income_range": (22_000, 85_000),
        "on_time_prob": 0.62,
        "default_prob": 0.08,
        "orders_range": (1, 5),
        "plan_preference": {"pay_in_4": 0.45, "pay_in_6": 0.32, "pay_in_12": 0.23},
        "order_amount_range": (60, 700),
        "dispute_prob": 0.05,
    },
    "slow_deterioration": {
        "weight": 0.10,
        "credit_score_range": (480, 680),
        "income_range": (18_000, 65_000),
        "on_time_prob": 0.50,
        "default_prob": 0.30,
        "orders_range": (2, 6),
        "plan_preference": {"pay_in_4": 0.30, "pay_in_6": 0.35, "pay_in_12": 0.35},
        "order_amount_range": (80, 1_200),
        "dispute_prob": 0.10,
    },
    "fraudster": {
        "weight": 0.05,
        "credit_score_range": (450, 750),
        "income_range": (20_000, 160_000),
        "on_time_prob": 0.10,
        "default_prob": 0.85,
        "orders_range": (2, 7),
        "plan_preference": {"pay_in_4": 0.55, "pay_in_6": 0.28, "pay_in_12": 0.17},
        "order_amount_range": (100, 1_800),
        "dispute_prob": 0.25,
    },
    "first_time_defaulter": {
        "weight": 0.05,
        "credit_score_range": (580, 750),
        "income_range": (28_000, 95_000),
        "on_time_prob": 0.82,
        "default_prob": 0.55,
        "orders_range": (1, 4),
        "plan_preference": {"pay_in_4": 0.42, "pay_in_6": 0.35, "pay_in_12": 0.23},
        "order_amount_range": (100, 1_000),
        "dispute_prob": 0.06,
    },
}

# ─────────────────────────────────────────────
# Merchant Categories & Risk Tiers
# ─────────────────────────────────────────────

MERCHANT_CATEGORIES = {
    "electronics":       {"weight": 0.20, "risk_tier": "high",   "avg_order": 450, "order_std": 200},
    "fashion":           {"weight": 0.25, "risk_tier": "medium", "avg_order": 120, "order_std": 80},
    "home_furniture":    {"weight": 0.10, "risk_tier": "medium", "avg_order": 600, "order_std": 300},
    "health_beauty":     {"weight": 0.12, "risk_tier": "low",    "avg_order": 80,  "order_std": 50},
    "travel":            {"weight": 0.08, "risk_tier": "high",   "avg_order": 800, "order_std": 400},
    "gaming":            {"weight": 0.07, "risk_tier": "high",   "avg_order": 350, "order_std": 150},
    "grocery":           {"weight": 0.05, "risk_tier": "low",    "avg_order": 60,  "order_std": 30},
    "jewelry_luxury":    {"weight": 0.05, "risk_tier": "high",   "avg_order": 900, "order_std": 500},
    "sports_outdoor":    {"weight": 0.05, "risk_tier": "low",    "avg_order": 150, "order_std": 100},
    "education":         {"weight": 0.03, "risk_tier": "low",    "avg_order": 200, "order_std": 100},
}

RISK_TIER_FRAUD_MULTIPLIER = {
    "low": 0.5,
    "medium": 1.0,
    "high": 2.0,
}

# ─────────────────────────────────────────────
# Payment Plan Types
# ─────────────────────────────────────────────

PLAN_TYPES = {
    "pay_in_4": {"num_installments": 4,  "frequency_days": 14, "apr": 0.0},
    "pay_in_6": {"num_installments": 6,  "frequency_days": 30, "apr": 0.10},
    "pay_in_12": {"num_installments": 12, "frequency_days": 30, "apr": 0.15},
}

# ─────────────────────────────────────────────
# Fraud Patterns
# ─────────────────────────────────────────────

FRAUD_PATTERNS = {
    "bust_out": {
        "description": "New account, high velocity, address mismatch, rapid default",
        "weight": 0.40,
        "account_age_max_days": 30,
        "min_orders_7d": 3,
        "use_vpn": True,
        "address_mismatch": True,
    },
    "high_value": {
        "description": "Targets $600+ transactions, often electronics/luxury",
        "weight": 0.25,
        "min_amount": 600,
        "preferred_categories": ["electronics", "jewelry_luxury", "travel"],
    },
    "synthetic_identity": {
        "description": "Shared devices/IPs across multiple fake identities",
        "weight": 0.20,
        "shared_device_pool_size": 15,
        "shared_ip_pool_size": 10,
    },
    "friendly_fraud": {
        "description": "Disputes legitimate purchases, claims non-receipt",
        "weight": 0.15,
        "dispute_rate": 0.80,
        "dispute_reasons": ["item_not_received", "unauthorized_transaction", "not_as_described"],
    },
}

# ─────────────────────────────────────────────
# Device & Channel Configuration
# ─────────────────────────────────────────────

DEVICE_TYPES = ["mobile_ios", "mobile_android", "desktop_windows", "desktop_mac", "tablet"]
DEVICE_WEIGHTS = [0.30, 0.25, 0.20, 0.15, 0.10]

BROWSERS = ["Chrome", "Safari", "Firefox", "Edge", "Samsung Internet", "Opera"]
BROWSER_WEIGHTS = [0.45, 0.25, 0.12, 0.08, 0.06, 0.04]

OPERATING_SYSTEMS = ["iOS", "Android", "Windows", "macOS", "Linux"]
OS_WEIGHTS = [0.30, 0.25, 0.25, 0.15, 0.05]

CHANNELS = ["web", "mobile_app", "in_store_qr", "partner_checkout"]
CHANNEL_WEIGHTS = [0.35, 0.40, 0.10, 0.15]

PAYMENT_METHODS = ["debit_card", "credit_card", "bank_account", "digital_wallet"]
PAYMENT_METHOD_WEIGHTS = [0.35, 0.25, 0.25, 0.15]

# ─────────────────────────────────────────────
# US States (for address generation)
# ─────────────────────────────────────────────

US_STATES = {
    "CA": "California", "TX": "Texas", "FL": "Florida", "NY": "New York",
    "PA": "Pennsylvania", "IL": "Illinois", "OH": "Ohio", "GA": "Georgia",
    "NC": "North Carolina", "MI": "Michigan", "NJ": "New Jersey",
    "VA": "Virginia", "WA": "Washington", "AZ": "Arizona", "MA": "Massachusetts",
    "TN": "Tennessee", "IN": "Indiana", "MO": "Missouri", "MD": "Maryland",
    "WI": "Wisconsin", "CO": "Colorado", "MN": "Minnesota", "SC": "South Carolina",
    "AL": "Alabama", "LA": "Louisiana", "KY": "Kentucky", "OR": "Oregon",
    "OK": "Oklahoma", "CT": "Connecticut", "UT": "Utah",
}

_RAW_STATE_WEIGHTS = [
    0.12, 0.09, 0.07, 0.06, 0.04, 0.04, 0.04, 0.03, 0.03, 0.03,
    0.03, 0.03, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02,
    0.02, 0.02, 0.02, 0.015, 0.015, 0.014, 0.013, 0.012, 0.011, 0.01,
]
STATE_POPULATION_WEIGHTS = [w / sum(_RAW_STATE_WEIGHTS) for w in _RAW_STATE_WEIGHTS]

# ─────────────────────────────────────────────
# Employment Statuses
# ─────────────────────────────────────────────

EMPLOYMENT_STATUSES = ["employed_full_time", "employed_part_time", "self_employed",
                       "unemployed", "student", "retired"]
EMPLOYMENT_WEIGHTS = [0.50, 0.15, 0.12, 0.08, 0.10, 0.05]

# ─────────────────────────────────────────────
# Data Quality Issue Injection Rates
# ─────────────────────────────────────────────

DATA_QUALITY_ISSUES = {
    "missing_email_rate": 0.05,
    "missing_phone_rate": 0.08,
    "missing_income_rate": 0.03,
    "missing_credit_score_rate": 0.15,
    "missing_dob_rate": 0.02,
    "missing_employment_rate": 0.04,
    "duplicate_customer_rate": 0.02,
    "inconsistent_date_format_rate": 0.10,
    "inconsistent_state_format_rate": 0.15,   # full name vs abbreviation
    "dollar_sign_in_amount_rate": 0.05,
    "leading_zero_zip_loss_rate": 0.20,        # NJ, CT, MA zips lose leading 0
    "orphaned_installment_rate": 0.005,
    "negative_amount_rate": 0.002,
    "future_dob_rate": 0.001,
    "absurd_order_amount_rate": 0.003,
    "payment_before_order_rate": 0.008,
    "overpayment_rate": 0.005,
}

# ─────────────────────────────────────────────
# Dispute Reasons & Resolutions
# ─────────────────────────────────────────────

DISPUTE_REASONS = [
    "item_not_received", "unauthorized_transaction", "not_as_described",
    "duplicate_charge", "billing_error", "defective_product",
]
DISPUTE_REASON_WEIGHTS = [0.25, 0.20, 0.20, 0.10, 0.10, 0.15]

DISPUTE_RESOLUTIONS = ["resolved_merchant", "resolved_customer", "chargeback", "pending", "denied"]
DISPUTE_RESOLUTION_WEIGHTS = [0.30, 0.25, 0.20, 0.15, 0.10]

# ─────────────────────────────────────────────
# Credit Decision Configuration
# ─────────────────────────────────────────────

CREDIT_DECISION_RULES = {
    "auto_approve_min_score": 680,
    "auto_decline_max_score": 450,
    "manual_review_range": (450, 680),
    "approval_rate_overall": 0.75,
    "decline_reasons": [
        "insufficient_credit_score", "high_debt_to_income",
        "recent_delinquency", "new_account_velocity",
        "suspected_fraud", "insufficient_income",
    ],
}
