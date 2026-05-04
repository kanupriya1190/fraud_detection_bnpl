# BNPL Fraud Detection

Fraud detection system for Buy Now, Pay Later transactions. Built from scratch with synthetic data that mirrors real BNPL schemas (Affirm/Klarna/Afterpay), this project covers the full pipeline from raw transactional data through a deployable model with monitoring.

**Result:** LightGBM achieves **0.62 PR-AUC** on a time-based holdout with probabilistic fraud labels, catching ~70% of fraud at ~69% precision — realistic for BNPL data with noisy ground truth.

## Why BNPL Fraud Is Different

Traditional payment fraud models don't transfer well to BNPL because:

- **No upfront payment.** Approvals happen at checkout with minimal friction. Bust-out fraudsters exploit this by making purchases they never intend to pay for.
- **Installment patterns carry signal.** A customer's path through payment states (current → late → missed → default) contains information that single-transaction models can't use. This project borrows transition matrix methods from credit risk to extract these signals.
- **Small amounts, aggregate losses.** Individual BNPL orders ($50-500) don't trigger traditional high-value fraud rules, but coordinated fraud rings placing dozens of orders through shared devices and IPs add up quickly.

## Approach

### Data

51K synthetic customers, 150K+ orders across 8 normalized tables. Data includes intentionally injected quality issues (mixed date formats, dollar signs in numeric fields, orphaned foreign keys, near-duplicate records) because real production data is messy and cleaning it is half the work.

Customer segments: 60% good payers, 20% occasionally late, 10% gradual deterioration, 5% fraudsters, 5% first-time defaulters. Fraud labels are **probabilistic per order** — fraudsters don't commit fraud on every order (trust-building phase), and some legitimate customers commit opportunistic fraud — which introduces realistic label noise that prevents trivial segment memorization.

### Feature Engineering (50+ features)

| Category | Examples | Why It Matters |
|----------|----------|---------------|
| Payment behavior | On-time rate, max DPD, first-payment-default flag | Past behavior is the strongest predictor of future default |
| Velocity | Orders in last 7/30/90 days, time since last order | Bust-out fraudsters compress activity into short windows |
| Device signals | Shared device count, VPN usage, IP-geo mismatch | Fraud rings share devices across synthetic identities |
| Merchant risk | Historical fraud rate, dispute rate by merchant | Certain merchant categories concentrate fraud |
| Transition matrix | Worst state reached, path entropy, DPD acceleration, status volatility | Captures the *trajectory* of payment behavior, not just snapshots |

All features are computed with **point-in-time correctness** — each order only uses information available before that order was placed.

### Models

| Model | PR-AUC | ROC-AUC | Precision | Recall |
|-------|--------|---------|-----------|--------|
| Logistic Regression | 0.584 | 0.916 | 63.9% | 67.6% |
| Random Forest | 0.618 | 0.919 | 62.8% | 72.8% |
| XGBoost | 0.611 | 0.917 | 66.7% | 70.7% |
| **LightGBM** | **0.618** | **0.917** | **68.8%** | **69.9%** |

Evaluated on a **time-based split** (not random), so the test set contains only orders that came after all training orders. Primary metric is PR-AUC because with ~6% fraud prevalence, accuracy is meaningless.

### Monitoring

Feature drift detection (KS test), concept drift tracking (fraud rate shifts over time), model performance degradation alerts, and auto-retrain trigger logic. The monitoring notebook simulates 6 deployment windows and checks each for drift.

## Where This Goes Next: Risk-Aware Threshold Optimization

The model above optimizes a static threshold over the full dataset — implicitly assuming symmetric losses and a stationary fraud process. Neither holds. Fraud losses are heavy-tailed, and attack patterns shift with seasonality, adversarial adaptation, and macro conditions. The natural next step is treating threshold selection as a **risk management problem**, not just a classification problem:

- **Tail-aware thresholds (CVaR).** Instead of minimizing expected daily loss, minimize Expected Shortfall at 95% — the average loss on the worst 5% of days. This is structurally equivalent to Rockafellar–Uryasev CVaR optimization from portfolio construction. The ES-optimal threshold is typically more conservative, accepting modestly higher average cost for substantially reduced tail risk.

- **Regime-aware policy.** Replace a fixed threshold with τ(state) that adapts to identifiable risk regimes — calendar-based rules for high-risk windows (Black Friday, tax season) where fraud concentrates and review teams saturate, escalating to dynamic triggers based on rolling fraud rate and volume anomalies. This Pareto-dominates both fixed-conservative and fixed-aggressive baselines on the (expected loss, tail loss) frontier.

- **Macro-conditional extension.** Fraud loss distributions are sensitive to macro regime — rising rates and deteriorating consumer credit correlate with elevated first-party and synthetic identity fraud with 6–18 month lags. Incorporating macro state lets the policy tighten preemptively during stress periods. This reframes fraud detection as an implicit macro hedge: tightening during rate-hike cycles is functionally analogous to buying credit protection, with false-positive cost as the premium.

All three are backtestable with walk-forward simulation on the existing dataset. Key caveats: tail estimates need bootstrap CIs when fraud events are rare; regime boundaries create customer-experience artifacts (smooth transitions > step functions); adversarial response requires policy randomization; and macro-conditional models need multiple rate cycles to estimate reliably — Bayesian shrinkage toward industry priors is a practical substitute with limited history.

## Project Structure

```
├── data/                          # Generated databases and model artifacts
├── notebooks/
│   ├── 01_data_exploration.ipynb  # EDA and data quality audit
│   ├── 02_sql_analysis.ipynb      # SQL: CTEs, window functions, fraud rings
│   ├── 03_data_cleaning.ipynb     # Standardization, deduplication
│   ├── 04_feature_engineering.ipynb # 50+ features, point-in-time correct
│   ├── 05_transition_matrix.ipynb # Monthly transitions, Markov chains
│   ├── 06_fraud_detection_ml.ipynb # LR, RF, XGBoost, LightGBM + SHAP
│   ├── 07_monitoring_drift.ipynb  # Drift detection and retrain triggers
│   └── 08_results_summary.ipynb   # Executive summary
├── src/
│   ├── data_generation/           # Synthetic data generator
│   ├── api/app.py                 # FastAPI prediction endpoint
│   └── utils.py                   # Shared utilities
├── streamlit_app.py               # Interactive dashboard (6 tabs)
└── requirements.txt
```

## Running It

```bash
pip install -r requirements.txt
python -m src.data_generation.generate_raw_data   # generates data/bnpl.db
# then run notebooks 01-08 in order

# Optional: API and dashboard
python -m src.api.app              # FastAPI at localhost:8000
streamlit run streamlit_app.py     # Dashboard at localhost:8501
```

## Tech Stack

Python, pandas, NumPy, SQLite, scikit-learn, XGBoost, LightGBM, SHAP, matplotlib, seaborn, FastAPI, Streamlit
