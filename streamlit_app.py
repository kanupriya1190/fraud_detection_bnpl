from __future__ import annotations

import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
DB_PATH = DATA_DIR / "bnpl.db"
CLEAN_DB_PATH = DATA_DIR / "bnpl_clean.db"
FEATURE_CSV = DATA_DIR / "feature_matrix.csv"
FEATURE_PARQUET = DATA_DIR / "feature_matrix.parquet"
MODEL_PATH = DATA_DIR / "best_model.joblib"
METRICS_PATH = DATA_DIR / "model_metrics.csv"

MERCHANT_CATEGORIES = [
    "education", "electronics", "fashion", "gaming", "grocery",
    "health_beauty", "home_furniture", "jewelry_luxury", "sports_outdoor", "travel",
]

st.set_page_config(
    page_title="BNPL Fraud Detection Dashboard",
    layout="wide",
    page_icon="🛡️",
)

# ── Sidebar ──────────────────────────────────────────────────────────────────

st.sidebar.title("BNPL Fraud Detection")
st.sidebar.markdown(
    """
    End-to-end fraud detection system for Buy Now, Pay Later transactions.

    **Pipeline stages**
    1. Synthetic data generation
    2. Data cleaning & feature engineering
    3. Model training & evaluation
    4. Real-time scoring API
    5. Monitoring & drift detection

    Navigate between the tabs above to explore each stage.
    """
)
st.sidebar.divider()
st.sidebar.caption(f"DB: `{DB_PATH.name}` {'✓' if DB_PATH.exists() else '✗'}")
st.sidebar.caption(f"Clean DB: `{CLEAN_DB_PATH.name}` {'✓' if CLEAN_DB_PATH.exists() else '✗'}")
st.sidebar.caption(f"Feature matrix: {'✓' if FEATURE_CSV.exists() or FEATURE_PARQUET.exists() else '✗'}")
st.sidebar.caption(f"Model: `{MODEL_PATH.name}` {'✓' if MODEL_PATH.exists() else '✗'}")

# ── Data loaders ─────────────────────────────────────────────────────────────


@st.cache_data(show_spinner=False)
def query_db(sql: str, db: str = "raw") -> pd.DataFrame:
    path = CLEAN_DB_PATH if db == "clean" and CLEAN_DB_PATH.exists() else DB_PATH
    with sqlite3.connect(path) as conn:
        return pd.read_sql_query(sql, conn)


@st.cache_data(show_spinner=False)
def load_feature_matrix() -> pd.DataFrame | None:
    try:
        if FEATURE_PARQUET.exists():
            return pd.read_parquet(FEATURE_PARQUET)
        if FEATURE_CSV.exists():
            return pd.read_csv(FEATURE_CSV)
    except Exception:
        pass
    return None


@st.cache_data(show_spinner=False)
def load_model():
    if not MODEL_PATH.exists():
        return None
    try:
        import joblib
        return joblib.load(MODEL_PATH)
    except Exception:
        return None


@st.cache_data(show_spinner=False)
def load_metrics() -> pd.DataFrame | None:
    if not METRICS_PATH.exists():
        return None
    try:
        return pd.read_csv(METRICS_PATH)
    except Exception:
        return None


# ── Tabs ─────────────────────────────────────────────────────────────────────

st.title("BNPL Fraud Detection Dashboard")

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "Data Quality",
    "Model Comparison",
    "Feature Importance",
    "SHAP Explanations",
    "Predictions",
    "Monitoring",
])

# ═════════════════════════════════════════════════════════════════════════════
# TAB 1 – Data Quality
# ═════════════════════════════════════════════════════════════════════════════

with tab1:
    if not DB_PATH.exists():
        st.error("Database not found at `data/bnpl.db`. Please run the data generation pipeline first.")
    else:
        st.header("Dataset Overview")

        n_cust = query_db("SELECT count(*) AS n FROM customers").iloc[0, 0]
        n_orders = query_db("SELECT count(*) AS n FROM orders").iloc[0, 0]
        gmv_df = query_db(
            "SELECT sum(CAST(REPLACE(REPLACE(order_amount, '$', ''), ',', '') AS REAL)) AS gmv FROM orders"
        )
        total_gmv = gmv_df.iloc[0, 0] or 0

        c1, c2, c3 = st.columns(3)
        c1.metric("Customers", f"{n_cust:,}")
        c2.metric("Orders", f"{n_orders:,}")
        c3.metric("Total GMV", f"${total_gmv:,.0f}")

        st.subheader("Fraud Distribution (Ground Truth Segments)")
        seg = query_db("SELECT segment, count(*) AS cnt FROM customer_segments_ground_truth GROUP BY segment")
        if seg.empty:
            st.info("No ground truth segments found.")
        else:
            fig_seg = px.pie(seg, names="segment", values="cnt", title="Customer Segments",
                             color_discrete_sequence=px.colors.qualitative.Set2)
            st.plotly_chart(fig_seg, width="stretch")

        st.subheader("Missing Value Summary")
        tables_cols = {
            "customers": [
                "customer_id", "first_name", "last_name", "email", "phone",
                "address", "city", "state", "zip_code", "dob", "signup_date",
                "ssn_last4", "annual_income", "credit_score", "employment_status",
            ],
            "orders": [
                "order_id", "customer_id", "merchant_id", "order_amount",
                "order_date", "order_status", "channel",
            ],
        }
        missing_rows = []
        for tbl, cols in tables_cols.items():
            total = query_db(f"SELECT count(*) AS n FROM {tbl}").iloc[0, 0]
            for col in cols:
                null_cnt = query_db(
                    f"SELECT count(*) AS n FROM {tbl} WHERE {col} IS NULL OR TRIM({col}) = ''"
                ).iloc[0, 0]
                if null_cnt > 0:
                    missing_rows.append({
                        "Table": tbl,
                        "Column": col,
                        "Missing": null_cnt,
                        "% Missing": round(null_cnt / total * 100, 2) if total else 0,
                    })
        if missing_rows:
            st.dataframe(pd.DataFrame(missing_rows), width="stretch")
        else:
            st.success("No missing values detected in customers / orders.")

        st.subheader("Data Quality Issues")
        dup_cust = query_db(
            "SELECT count(*) AS n FROM ("
            "  SELECT email, count(*) AS c FROM customers WHERE email IS NOT NULL "
            "  GROUP BY email HAVING c > 1"
            ")"
        ).iloc[0, 0]
        orphan_inst = query_db(
            "SELECT count(*) AS n FROM installments i "
            "LEFT JOIN payment_plans p ON i.plan_id = p.plan_id WHERE p.plan_id IS NULL"
        ).iloc[0, 0]
        dollar_fmt = query_db(
            "SELECT count(*) AS n FROM orders WHERE order_amount LIKE '$%'"
        ).iloc[0, 0]

        q1, q2, q3 = st.columns(3)
        q1.metric("Duplicate Emails", f"{dup_cust:,}")
        q2.metric("Orphaned Installments", f"{orphan_inst:,}")
        q3.metric("Dollar-sign Amounts", f"{dollar_fmt:,}")

        st.subheader("Feature Statistics (Numeric Columns)")
        cust_df = query_db("SELECT annual_income, credit_score FROM customers")
        cust_df["annual_income"] = pd.to_numeric(cust_df["annual_income"], errors="coerce")
        cust_df["credit_score"] = pd.to_numeric(cust_df["credit_score"], errors="coerce")
        st.dataframe(cust_df.describe().T.round(2), width="stretch")

# ═════════════════════════════════════════════════════════════════════════════
# TAB 2 – Model Comparison
# ═════════════════════════════════════════════════════════════════════════════

with tab2:
    st.header("Model Comparison")
    metrics_df = load_metrics()
    if metrics_df is not None:
        for col in metrics_df.select_dtypes(include="object").columns:
            metrics_df[col] = metrics_df[col].str.strip()
        display_cols = [c for c in metrics_df.columns if c.lower() != "index"]
        st.dataframe(metrics_df[display_cols], width="stretch")

        prauc_col = next((c for c in metrics_df.columns if "pr" in c.lower() and "auc" in c.lower()), None)
        model_col = next((c for c in metrics_df.columns if "model" in c.lower()), metrics_df.columns[0])

        if prauc_col and model_col:
            fig_bar = px.bar(
                metrics_df, x=model_col, y=prauc_col,
                title="PR-AUC Comparison Across Models",
                labels={model_col: "Model", prauc_col: "PR-AUC"},
            )
            colors = px.colors.qualitative.Bold
            fig_bar.update_traces(
                marker_color=[colors[i % len(colors)] for i in range(len(metrics_df))]
            )
            st.plotly_chart(fig_bar, width="stretch")

            best_idx = metrics_df[prauc_col].idxmax()
            best_model = metrics_df.loc[best_idx, model_col]
            best_score = metrics_df.loc[best_idx, prauc_col]
            st.success(
                f"**Best model: {best_model}** with PR-AUC = {best_score:.4f}. "
                "Selected because PR-AUC is the primary metric for imbalanced fraud detection — "
                "it captures performance on the minority (fraud) class better than ROC-AUC."
            )
        else:
            st.info("Could not identify PR-AUC or model name columns automatically. "
                    "Showing raw table above.")
    else:
        st.info("Model metrics file not found at `data/model_metrics.csv`.")
        st.markdown("#### Expected format")
        placeholder = pd.DataFrame({
            "Model": ["Logistic Regression", "Random Forest", "XGBoost", "LightGBM"],
            "PR-AUC": [0.584, 0.618, 0.611, 0.618],
            "ROC-AUC": [0.916, 0.919, 0.917, 0.917],
            "Precision": [0.639, 0.628, 0.667, 0.688],
            "Recall": [0.676, 0.728, 0.707, 0.699],
            "F1": [0.657, 0.674, 0.686, 0.693],
        })
        st.dataframe(placeholder, width="stretch")
        st.caption("This is example data. Train models to populate real metrics.")

# ═════════════════════════════════════════════════════════════════════════════
# TAB 3 – Feature Importance
# ═════════════════════════════════════════════════════════════════════════════

with tab3:
    st.header("Feature Importance")
    model = load_model()
    fm = load_feature_matrix()

    if model is not None:
        importances = None
        feature_names = None

        if hasattr(model, "feature_names_in_"):
            feature_names = list(model.feature_names_in_)
        elif fm is not None:
            feature_names = [c for c in fm.columns if c.lower() not in ("is_fraud", "customer_id", "order_id")]

        if hasattr(model, "feature_importances_"):
            importances = model.feature_importances_
        elif hasattr(model, "coef_"):
            importances = np.abs(model.coef_).flatten()

        if importances is not None and feature_names is not None and len(importances) == len(feature_names):
            imp_df = (
                pd.DataFrame({"Feature": feature_names, "Importance": importances})
                .sort_values("Importance", ascending=False)
                .head(20)
            )
            fig_imp = px.bar(
                imp_df, x="Importance", y="Feature", orientation="h",
                title="Top 20 Feature Importances",
                color="Importance",
                color_continuous_scale="Tealgrn",
            )
            fig_imp.update_layout(yaxis=dict(autorange="reversed"))
            st.plotly_chart(fig_imp, width="stretch")

            st.subheader("Feature Descriptions")
            st.dataframe(imp_df.reset_index(drop=True), width="stretch")
        else:
            st.warning("Model does not expose feature importances or feature count mismatch.")
    else:
        st.info("Feature importance requires a trained model (`data/best_model.joblib`). "
                "Run the training pipeline to generate it.")
        st.markdown(
            """
            **What this tab shows when data is available:**
            - A horizontal bar chart of the top 20 features ranked by importance
            - Feature names mapped to human-readable descriptions
            """
        )

# ═════════════════════════════════════════════════════════════════════════════
# TAB 4 – SHAP Explanations
# ═════════════════════════════════════════════════════════════════════════════

with tab4:
    st.header("SHAP Explanations")
    model = load_model()
    fm = load_feature_matrix()

    if model is not None and fm is not None:
        import matplotlib.pyplot as plt

        if hasattr(model, "feature_names_in_"):
            feature_names = list(model.feature_names_in_)
            for feat in feature_names:
                if feat not in fm.columns:
                    fm[feat] = 0
        else:
            feature_names = [c for c in fm.columns if c.lower() not in ("is_fraud", "customer_id", "order_id")]
        X = fm[feature_names].fillna(0)

        try:
            import shap

            sample_size = min(200, len(X))
            X_sample = X.sample(sample_size, random_state=42)

            shap_ok = False
            with st.spinner("Computing SHAP values (this may take a moment)…"):
                try:
                    explainer = shap.TreeExplainer(model)
                    shap_values = explainer(X_sample)
                    shap_ok = True
                except Exception:
                    pass

            if shap_ok:
                st.subheader("SHAP Summary Plot")
                fig_shap, ax = plt.subplots(figsize=(10, 6))
                vals = shap_values.values if hasattr(shap_values, "values") else shap_values
                if vals.ndim == 3:
                    vals = vals[:, :, 1]
                shap.summary_plot(vals, X_sample, show=False)
                st.pyplot(fig_shap)
                plt.close(fig_shap)

                st.subheader("Individual Explanation")
                idx = st.selectbox("Select sample index", range(sample_size), index=0)
                force_vals = vals[idx]
                shap.waterfall_plot(
                    shap.Explanation(
                        values=force_vals,
                        base_values=explainer.expected_value[1] if isinstance(explainer.expected_value, (list, np.ndarray)) else explainer.expected_value,
                        data=X_sample.iloc[idx],
                        feature_names=feature_names,
                    ),
                    show=False,
                )
                st.pyplot(plt.gcf())
                plt.close("all")
            else:
                st.warning("SHAP TreeExplainer is incompatible with the current model. "
                           "Showing permutation-based feature importance instead.")

                from sklearn.inspection import permutation_importance
                y = fm["is_fraud"].values if "is_fraud" in fm.columns else None
                if y is not None:
                    perm = permutation_importance(model, X_sample, y[:sample_size],
                                                  n_repeats=10, random_state=42,
                                                  scoring="average_precision")
                    perm_df = (pd.DataFrame({"Feature": feature_names, "Importance": perm.importances_mean})
                               .sort_values("Importance", ascending=False).head(20))
                    fig_perm = px.bar(perm_df, x="Importance", y="Feature", orientation="h",
                                     title="Permutation Feature Importance (top 20)")
                    fig_perm.update_layout(yaxis=dict(autorange="reversed"))
                    st.plotly_chart(fig_perm, width="stretch")
                else:
                    st.info("Target column `is_fraud` not found in feature matrix.")
        except ImportError:
            st.warning("Install `shap` (`pip install shap`) to enable this tab.")
    else:
        st.info("SHAP explanations require a trained model and feature matrix.")
        st.markdown(
            """
            **SHAP (SHapley Additive exPlanations)** decomposes each prediction into
            per-feature contributions, answering *"why did the model predict this?"*.

            When available this tab shows:
            - **Summary plot** — global feature importance with direction of effect
            - **Waterfall / force plot** — per-transaction explanation
            """
        )

# ═════════════════════════════════════════════════════════════════════════════
# TAB 5 – Predictions
# ═════════════════════════════════════════════════════════════════════════════

with tab5:
    st.header("Fraud Prediction")

    col_left, col_right = st.columns([1, 1])

    with col_left:
        amount = st.slider("Amount ($)", 10, 5000, 250, step=10)
        merchant_category = st.selectbox("Merchant Category", MERCHANT_CATEGORIES, index=2)
        plan_map = {"pay_in_4": 4, "pay_in_6": 6, "pay_in_12": 12}
        plan_type = st.selectbox("Plan Type", list(plan_map.keys()))
        credit_score = st.slider("Credit Score", 300, 850, 700)
        annual_income = st.slider("Annual Income ($)", 10_000, 200_000, 50_000, step=1_000)

    with col_right:
        account_age_days = st.slider("Account Age (days)", 0, 1000, 365)
        orders_last_30d = st.slider("Orders Last 30 Days", 0, 10, 1)
        vpn_used = st.checkbox("VPN Used")
        on_time_rate = st.slider("Historical On-Time Rate", 0.0, 1.0, 0.95, step=0.01)
        prior_defaults = st.slider("Prior Defaults", 0, 5, 0)

    if st.button("Predict", type="primary"):
        fraud_prob: float | None = None
        source = "rule-based"

        payload = {
            "amount": float(amount),
            "merchant_category": merchant_category,
            "num_installments": plan_map[plan_type],
            "account_age_days": account_age_days,
            "credit_score": credit_score,
            "annual_income": float(annual_income),
            "is_vpn": vpn_used,
            "velocity_7d": min(orders_last_30d, 3),
            "velocity_30d": orders_last_30d,
            "phone_verified": True,
            "address_match": True,
            "device_age_days": 180,
            "prior_defaults": prior_defaults,
            "on_time_rate": on_time_rate,
        }

        try:
            import requests
            resp = requests.post("http://localhost:8000/predict", json=payload, timeout=3)
            if resp.status_code == 200:
                result = resp.json()
                fraud_prob = result["fraud_probability"]
                source = "API"
        except Exception:
            pass

        if fraud_prob is None:
            mdl = load_model()
            if mdl is not None and hasattr(mdl, "feature_names_in_"):
                input_map = {
                    "order_amount": float(amount),
                    "num_installments": float(plan_map[plan_type]),
                    "account_age_days": float(account_age_days),
                    "credit_score": float(credit_score),
                    "annual_income": float(annual_income),
                    "is_vpn_used": float(vpn_used),
                    "orders_7d": float(min(orders_last_30d, 3)),
                    "orders_30d": float(orders_last_30d),
                    "prior_defaults": float(prior_defaults),
                    "ontime_rate": on_time_rate,
                }
                row = {feat: input_map.get(feat, 0.0) for feat in mdl.feature_names_in_}
                features = pd.DataFrame([row])[list(mdl.feature_names_in_)]
                try:
                    proba = mdl.predict_proba(features)[0]
                    fraud_prob = float(proba[1]) if len(proba) > 1 else float(proba[0])
                    source = "local model"
                except Exception:
                    pass

        if fraud_prob is None:
            score = 0.0
            if amount > 600:
                score += 0.15 + min((amount - 600) / 2000, 0.15)
            if vpn_used:
                score += 0.15
            if account_age_days < 30:
                score += 0.15
            elif account_age_days < 90:
                score += 0.07
            if orders_last_30d > 3:
                score += 0.10
            if credit_score < 550:
                score += 0.15
            elif credit_score < 650:
                score += 0.08
            if prior_defaults > 0:
                score += 0.10 + min(prior_defaults * 0.05, 0.15)
            if on_time_rate < 0.7:
                score += 0.12
            elif on_time_rate < 0.85:
                score += 0.06
            fraud_prob = min(score, 0.99)
            source = "rule-based"

        if fraud_prob >= 0.8:
            risk_level, risk_color = "CRITICAL", "#d32f2f"
        elif fraud_prob >= 0.5:
            risk_level, risk_color = "HIGH", "#f57c00"
        elif fraud_prob >= 0.3:
            risk_level, risk_color = "MEDIUM", "#fbc02d"
        else:
            risk_level, risk_color = "LOW", "#388e3c"

        st.divider()
        g1, g2, g3 = st.columns([2, 1, 1])
        with g1:
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=fraud_prob * 100,
                number={"suffix": "%"},
                title={"text": "Fraud Probability"},
                gauge={
                    "axis": {"range": [0, 100]},
                    "bar": {"color": risk_color},
                    "steps": [
                        {"range": [0, 30], "color": "#e8f5e9"},
                        {"range": [30, 50], "color": "#fff9c4"},
                        {"range": [50, 80], "color": "#ffe0b2"},
                        {"range": [80, 100], "color": "#ffcdd2"},
                    ],
                },
            ))
            fig_gauge.update_layout(height=280, margin=dict(t=50, b=10, l=30, r=30))
            st.plotly_chart(fig_gauge, width="stretch")

        with g2:
            st.markdown(f"### Risk Level")
            st.markdown(
                f"<h1 style='color:{risk_color};text-align:center'>{risk_level}</h1>",
                unsafe_allow_html=True,
            )
        with g3:
            st.markdown("### Scoring Source")
            st.markdown(f"<h3 style='text-align:center'>{source}</h3>", unsafe_allow_html=True)

        factors = []
        if amount > 600:
            factors.append(f"High transaction amount (${amount:,})")
        if vpn_used:
            factors.append("VPN detected")
        if account_age_days < 30:
            factors.append(f"New account ({account_age_days} days)")
        if credit_score < 650:
            factors.append(f"Low credit score ({credit_score})")
        if prior_defaults > 0:
            factors.append(f"Prior defaults ({prior_defaults})")
        if on_time_rate < 0.85:
            factors.append(f"Low on-time rate ({on_time_rate:.0%})")

        if factors:
            st.markdown("**Key risk factors:** " + " · ".join(factors))

# ═════════════════════════════════════════════════════════════════════════════
# TAB 6 – Monitoring
# ═════════════════════════════════════════════════════════════════════════════

with tab6:
    st.header("Model & Data Monitoring")
    fm = load_feature_matrix()

    if fm is not None and DB_PATH.exists():
        orders_df = query_db("SELECT o.order_id, o.order_date, "
                             "COALESCE(f.is_fraud, 0) AS is_fraud "
                             "FROM orders o "
                             "LEFT JOIN order_fraud_labels f ON o.order_id = f.order_id "
                             "WHERE o.order_date IS NOT NULL")
        orders_df["order_date"] = pd.to_datetime(orders_df["order_date"], errors="coerce")
        orders_df = orders_df.dropna(subset=["order_date"])
        orders_df["month"] = orders_df["order_date"].dt.to_period("M").astype(str)

        st.subheader("Fraud Rate Over Time")
        monthly = orders_df.groupby("month").agg(
            fraud_rate=("is_fraud", "mean"),
            total=("is_fraud", "count"),
        ).reset_index()

        mean_rate = monthly["fraud_rate"].mean()
        std_rate = monthly["fraud_rate"].std()
        monthly["UCL"] = mean_rate + 2 * std_rate
        monthly["LCL"] = (mean_rate - 2 * std_rate).clip(min=0)
        monthly["Mean"] = mean_rate

        fig_fr = go.Figure()
        fig_fr.add_trace(go.Scatter(x=monthly["month"], y=monthly["fraud_rate"],
                                    mode="lines+markers", name="Fraud Rate"))
        fig_fr.add_trace(go.Scatter(x=monthly["month"], y=monthly["UCL"],
                                    mode="lines", name="UCL (2σ)",
                                    line=dict(dash="dash", color="red")))
        fig_fr.add_trace(go.Scatter(x=monthly["month"], y=monthly["LCL"],
                                    mode="lines", name="LCL (2σ)",
                                    line=dict(dash="dash", color="red")))
        fig_fr.add_trace(go.Scatter(x=monthly["month"], y=monthly["Mean"],
                                    mode="lines", name="Mean",
                                    line=dict(dash="dot", color="gray")))
        fig_fr.update_layout(title="Monthly Fraud Rate with Control Limits",
                             xaxis_title="Month", yaxis_title="Fraud Rate",
                             yaxis_tickformat=".2%")
        st.plotly_chart(fig_fr, width="stretch")

        st.subheader("Model Performance Trend (Simulated)")
        mdl = load_model()
        if mdl is not None and "is_fraud" in fm.columns:
            from sklearn.metrics import average_precision_score

            if hasattr(mdl, "feature_names_in_"):
                feat_cols = list(mdl.feature_names_in_)
                for feat in feat_cols:
                    if feat not in fm.columns:
                        fm[feat] = 0
            else:
                feat_cols = [c for c in fm.columns if c.lower() not in ("is_fraud", "customer_id", "order_id")]
            try:
                probas = mdl.predict_proba(fm[feat_cols].fillna(0))[:, 1]
                n = len(fm)
                window = n // 6
                perf_rows = []
                for i in range(6):
                    s, e = i * window, (i + 1) * window
                    if e > n:
                        e = n
                    ap = average_precision_score(fm["is_fraud"].iloc[s:e], probas[s:e])
                    perf_rows.append({"Window": i + 1, "PR-AUC": ap})
                perf_df = pd.DataFrame(perf_rows)
                fig_perf = px.line(perf_df, x="Window", y="PR-AUC",
                                   title="PR-AUC Across Chronological Windows",
                                   markers=True)
                fig_perf.update_layout(xaxis_title="Time Window", yaxis_title="PR-AUC")
                st.plotly_chart(fig_perf, width="stretch")
            except Exception as e:
                st.warning(f"Could not compute performance trend: {e}")
        else:
            st.info("Load a trained model and a feature matrix with `is_fraud` column to see performance trends.")

        st.subheader("Feature Drift Heatmap")
        numeric_cols = fm.select_dtypes(include="number").columns.tolist()
        numeric_cols = [c for c in numeric_cols if c.lower() not in ("is_fraud", "customer_id", "order_id")]
        if numeric_cols:
            n = len(fm)
            half = n // 2
            ref = fm[numeric_cols].iloc[:half]
            cur = fm[numeric_cols].iloc[half:]
            drift_scores = {}
            for col in numeric_cols[:20]:
                ref_mean = ref[col].mean()
                ref_std = ref[col].std()
                if ref_std > 0:
                    drift_scores[col] = abs(cur[col].mean() - ref_mean) / ref_std
                else:
                    drift_scores[col] = 0.0
            drift_df = pd.DataFrame(list(drift_scores.items()), columns=["Feature", "Drift (σ)"])
            drift_df = drift_df.sort_values("Drift (σ)", ascending=False)
            fig_drift = px.bar(drift_df, x="Drift (σ)", y="Feature", orientation="h",
                               title="Feature Drift (Mean Shift in Std Devs)",
                               color="Drift (σ)", color_continuous_scale="RdYlGn_r")
            fig_drift.update_layout(yaxis=dict(autorange="reversed"))
            st.plotly_chart(fig_drift, width="stretch")

        st.subheader("Overall Health")
        latest_rate = monthly["fraud_rate"].iloc[-1] if not monthly.empty else 0
        ucl = monthly["UCL"].iloc[-1] if not monthly.empty else 0.2
        max_drift = max(drift_scores.values()) if drift_scores else 0

        if latest_rate > ucl or max_drift > 2.0:
            st.error("Status: **CRITICAL** — Fraud rate or feature drift exceeds thresholds.")
        elif latest_rate > mean_rate + std_rate or max_drift > 1.0:
            st.warning("Status: **WARNING** — Metrics approaching control limits.")
        else:
            st.success("Status: **HEALTHY** — All metrics within expected ranges.")
    elif DB_PATH.exists():
        orders_df = query_db("SELECT o.order_id, o.order_date, "
                             "COALESCE(f.is_fraud, 0) AS is_fraud "
                             "FROM orders o "
                             "LEFT JOIN order_fraud_labels f ON o.order_id = f.order_id "
                             "WHERE o.order_date IS NOT NULL")
        orders_df["order_date"] = pd.to_datetime(orders_df["order_date"], errors="coerce")
        orders_df = orders_df.dropna(subset=["order_date"])
        orders_df["month"] = orders_df["order_date"].dt.to_period("M").astype(str)

        st.subheader("Fraud Rate Over Time (from DB)")
        monthly = orders_df.groupby("month").agg(
            fraud_rate=("is_fraud", "mean"),
            total=("is_fraud", "count"),
        ).reset_index()

        mean_rate = monthly["fraud_rate"].mean()
        std_rate = monthly["fraud_rate"].std()
        monthly["UCL"] = mean_rate + 2 * std_rate
        monthly["LCL"] = (mean_rate - 2 * std_rate).clip(min=0)

        fig_fr = go.Figure()
        fig_fr.add_trace(go.Scatter(x=monthly["month"], y=monthly["fraud_rate"],
                                    mode="lines+markers", name="Fraud Rate"))
        fig_fr.add_trace(go.Scatter(x=monthly["month"], y=monthly["UCL"],
                                    mode="lines", name="UCL (2σ)",
                                    line=dict(dash="dash", color="red")))
        fig_fr.add_trace(go.Scatter(x=monthly["month"], y=monthly["LCL"],
                                    mode="lines", name="LCL (2σ)",
                                    line=dict(dash="dash", color="red")))
        fig_fr.update_layout(title="Monthly Fraud Rate with Control Limits",
                             xaxis_title="Month", yaxis_title="Fraud Rate",
                             yaxis_tickformat=".2%")
        st.plotly_chart(fig_fr, width="stretch")

        st.info("Feature drift and model performance trends require the feature matrix "
                "(`data/feature_matrix.csv` or `.parquet`) and trained model (`data/best_model.joblib`).")
    else:
        st.info("Monitoring requires the database and optionally the feature matrix and trained model.")
        st.markdown(
            """
            **When available, this tab shows:**
            - Fraud rate over time with statistical control limits
            - Model PR-AUC across chronological evaluation windows
            - Feature drift heatmap (mean shift in standard deviations)
            - Overall system health indicator (HEALTHY / WARNING / CRITICAL)
            """
        )
