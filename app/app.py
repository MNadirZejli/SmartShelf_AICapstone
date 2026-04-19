"""
app.py — SmartShelf v2
4 pages: Order Assistant | Model Comparison | Cost Dashboard | Model Insights
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import joblib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.data.features import get_feature_columns
from src.models.predict import forecast_item, compute_order_quantity, detect_stockout_risk
from src.cost.simulator import compute_costs, summary_by_store, summary_by_category, summary_by_model

st.set_page_config(page_title="SmartShelf", page_icon="📦", layout="wide")
st.markdown("""
<style>
.big-number { font-size:2rem; font-weight:700; color:#1D9E75; }
.label      { font-size:0.8rem; color:#888; }
.alert-red  { background:#fff0f0; border-left:4px solid #E24B4A; padding:0.75rem 1rem; border-radius:6px; }
.alert-amber{ background:#fffbf0; border-left:4px solid #EF9F27; padding:0.75rem 1rem; border-radius:6px; }
.alert-green{ background:#f0fff8; border-left:4px solid #1D9E75; padding:0.75rem 1rem; border-radius:6px; }
</style>
""", unsafe_allow_html=True)

PROCESSED = Path("data/processed")
MODELS    = Path("outputs/models")

MODEL_NAMES = {
    "lgbm":            "LightGBM",
    "xgb":             "XGBoost",
    "prophet":         "Prophet",
    "seasonal_naive":  "Seasonal Naive",
    "rolling_mean_28": "Rolling Mean 28d",
}
MODEL_COLORS = {
    "lgbm":            "#1D9E75",
    "xgb":             "#378ADD",
    "prophet":         "#EF9F27",
    "seasonal_naive":  "#888780",
    "rolling_mean_28": "#E24B4A",
}

# ── Cached loaders ─────────────────────────────────────────────────────────────
@st.cache_data
def load_features():
    return pd.read_parquet(PROCESSED / "sales_features.parquet")

@st.cache_data
def load_predictions():
    return pd.read_parquet(PROCESSED / "val_predictions.parquet")

@st.cache_resource
def load_lgbm():
    return joblib.load(MODELS / "lgbm_model.pkl")

@st.cache_resource
def load_encoders():
    return joblib.load(MODELS / "label_encoders.pkl")

@st.cache_data
def load_metrics():
    with open(MODELS / "metrics.json") as f:
        return json.load(f)

# ── Sidebar ────────────────────────────────────────────────────────────────────
st.sidebar.title("📦 SmartShelf")
st.sidebar.caption("AI-powered replenishment assistant")
st.sidebar.divider()
page = st.sidebar.radio("Navigation", [
    "🛒 Order Assistant",
    "📊 Model Comparison",
    "💶 Cost Dashboard",
    "🔬 Model Insights",
])
st.sidebar.divider()
st.sidebar.markdown("**Cost assumptions**")
holding_rate  = st.sidebar.slider("Holding cost (%/day)", 0.01, 0.20, 0.068, 0.001, format="%.3f%%") / 100
stockout_rate = st.sidebar.slider("Stockout cost (% of price)", 10, 150, 75, 5, format="%d%%") / 100


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — ORDER ASSISTANT
# ══════════════════════════════════════════════════════════════════════════════
if page == "🛒 Order Assistant":
    st.title("🛒 Order Assistant")
    st.caption("Real-time replenishment recommendation powered by LightGBM.")

    try:
        df = load_features()
        df["date"] = pd.to_datetime(df["date"])

        c1, c2, c3 = st.columns(3)
        store = c1.selectbox("Store", sorted(df["store_id"].unique()))
        items = sorted(df[df["store_id"] == store]["item_id"].unique())
        item  = c2.selectbox("Product", items)
        stock = c3.number_input("Current stock (units)", min_value=0, value=10, step=1)

        st.divider()

        item_df = df[(df["item_id"] == item) & (df["store_id"] == store)].copy()

        with st.spinner("Generating forecast ..."):
            forecast_df = forecast_item(item_df, horizon=7)

        order = compute_order_quantity(forecast_df, current_stock=stock)
        risk  = detect_stockout_risk(item_df, forecast_df, stock)

        # ── Stockout alert ────────────────────────────────────────────────
        if risk["risk_level"] == "critical":
            st.markdown(f"""<div class='alert-red'>
            ⚠️ <b>Critical stockout risk</b> — current stock covers only <b>{risk['days_cover']} days</b>.
            Demand is <b>{risk['trend']}</b>. Urgent reorder recommended.
            </div>""", unsafe_allow_html=True)
        elif risk["risk_level"] == "high":
            st.markdown(f"""<div class='alert-amber'>
            ⚠️ <b>Low stock</b> — {risk['days_cover']} days of cover remaining.
            Demand trend: <b>{risk['trend']}</b>.
            </div>""", unsafe_allow_html=True)
        else:
            st.markdown(f"""<div class='alert-green'>
            ✓ Stock level is adequate — {risk['days_cover']} days of cover.
            Demand trend: <b>{risk['trend']}</b>.
            </div>""", unsafe_allow_html=True)

        st.markdown("")

        # ── Metrics ───────────────────────────────────────────────────────
        ca, cb, cc, cd = st.columns(4)
        ca.markdown(f"<div class='label'>Recommended order</div><div class='big-number'>{order['order_quantity']} units</div>", unsafe_allow_html=True)
        cb.metric("7-day demand forecast", f"{order['expected_7d_demand']} units")
        cc.metric("Daily average (28d)",   f"{order['daily_avg']} units")
        cd.metric("Safety stock",          f"{order['safety_stock']} units")

        # ── SHAP explanation ──────────────────────────────────────────────
        st.subheader("Why this recommendation?")
        try:
            from src.explainer.shap_explainer import explain_prediction
            encoders = load_encoders()
            feat_row = item_df.tail(1).copy()
            for col, le in encoders.items():
                if col not in feat_row.columns:
                    continue
                feat_row[col] = feat_row[col].astype(str)
                known = set(le.classes_)
                feat_row[col] = feat_row[col].apply(lambda x: x if x in known else le.classes_[0])
                feat_row[col] = le.transform(feat_row[col]).astype(np.int32)
            for col in feat_row.select_dtypes(include="object").columns:
                feat_row = feat_row.drop(columns=[col])
            X_row   = feat_row[get_feature_columns()]
            bullets = explain_prediction(X_row, top_n=4)
            for b in bullets:
                st.markdown(f"• {b}")
        except Exception:
            daily_avg  = order["daily_avg"]
            days_cover = stock / daily_avg if daily_avg > 0 else 99
            reasons = []
            if days_cover < 3:
                reasons.append(f"Stock critically low — only {days_cover:.1f} days of cover")
            if pd.Timestamp.now().dayofweek >= 4:
                reasons.append("Weekend approaching — demand typically 15–25% higher")
            if not reasons:
                reasons.append("Standard replenishment based on 28-day demand forecast")
            for r in reasons:
                st.markdown(f"• {r}")

        # ── Forecast chart ────────────────────────────────────────────────
        st.subheader("Sales history + 7-day forecast")
        hist = item_df.tail(60)[["date", "sales"]]

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=hist["date"], y=hist["sales"],
            mode="lines", name="Historical sales",
            line=dict(color="#888780", width=1.5)
        ))
        fig.add_trace(go.Scatter(
            x=pd.concat([forecast_df["date"], forecast_df["date"][::-1]]),
            y=pd.concat([forecast_df["upper"], forecast_df["lower"][::-1]]),
            fill="toself", fillcolor="rgba(29,158,117,0.1)",
            line=dict(color="rgba(0,0,0,0)"), name="Confidence band"
        ))
        fig.add_trace(go.Scatter(
            x=forecast_df["date"], y=forecast_df["forecast"],
            mode="lines+markers", name="LightGBM forecast",
            line=dict(color="#1D9E75", width=2.5), marker=dict(size=7)
        ))
        fig.update_layout(height=320, margin=dict(l=0,r=0,t=10,b=0),
            yaxis_title="Units", hovermode="x unified",
            legend=dict(orientation="h", y=1.12))
        st.plotly_chart(fig, use_container_width=True)

    except FileNotFoundError:
        st.error("Run `python run_pipeline.py` first.")
    except Exception as e:
        st.error(f"Error: {e}")
        import traceback; st.code(traceback.format_exc())


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — MODEL COMPARISON (the new page that shows we're serious)
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📊 Model Comparison":
    st.title("📊 Model Comparison")
    st.caption("Walk-forward cross-validation + holdout evaluation. 3 models vs 2 baselines.")

    try:
        metrics = load_metrics()
        val     = load_predictions()
        val["date"] = pd.to_datetime(val["date"])

        # ── CV results ────────────────────────────────────────────────────
        st.subheader("Walk-forward cross-validation (3 folds)")
        st.caption("Each fold trains on all data up to a cutoff, predicts the next 28 days. This is the correct way to evaluate time series models.")

        cv = metrics.get("cv_mae", {})
        cv_rows = []
        for model, mae in cv.items():
            if mae is not None:
                cv_rows.append({"Model": MODEL_NAMES.get(model, model), "CV MAE": mae})
        cv_df = pd.DataFrame(cv_rows).sort_values("CV MAE")
        st.dataframe(cv_df.style.highlight_min(subset=["CV MAE"], color="#E1F5EE"), use_container_width=True)

        st.divider()

        # ── Holdout results ───────────────────────────────────────────────
        st.subheader("Holdout evaluation (56 days — never seen during training)")
        holdout = metrics.get("holdout", {})

        cols = st.columns(len(holdout))
        for i, (model, m) in enumerate(holdout.items()):
            with cols[i]:
                name  = MODEL_NAMES.get(model, model)
                color = MODEL_COLORS.get(model, "#888")
                st.markdown(f"**{name}**")
                st.metric("MAE",  m["mae"])
                st.metric("RMSE", m["rmse"])

        st.divider()

        # ── MAE comparison chart ──────────────────────────────────────────
        st.subheader("MAE comparison — all models")
        mae_data = {MODEL_NAMES.get(k, k): v["mae"] for k, v in holdout.items()}
        mae_df   = pd.DataFrame(list(mae_data.items()), columns=["Model", "MAE"]).sort_values("MAE")

        fig_mae = px.bar(mae_df, x="MAE", y="Model", orientation="h",
            color="MAE", color_continuous_scale=[[0,"#1D9E75"],[1,"#E24B4A"]],
            text="MAE")
        fig_mae.update_traces(texttemplate="%{text:.4f}", textposition="outside")
        fig_mae.update_layout(height=320, margin=dict(l=0,r=0,t=10,b=0),
            coloraxis_showscale=False, yaxis=dict(autorange="reversed"))
        st.plotly_chart(fig_mae, use_container_width=True)

        # ── Per-item error distribution ───────────────────────────────────
        st.subheader("Forecast error distribution — LightGBM vs seasonal naive")
        lgbm_errors    = (val["pred_lgbm"] - val["sales"]).values
        seasonal_errors = (val["pred_seasonal_naive"] - val["sales"]).values

        fig_err = go.Figure()
        fig_err.add_trace(go.Histogram(x=lgbm_errors, name="LightGBM",
            opacity=0.7, marker_color="#1D9E75", nbinsx=60, xbins=dict(start=-10, end=10)))
        fig_err.add_trace(go.Histogram(x=seasonal_errors, name="Seasonal Naive",
            opacity=0.7, marker_color="#888780", nbinsx=60, xbins=dict(start=-10, end=10)))
        fig_err.update_layout(barmode="overlay", height=300,
            margin=dict(l=0,r=0,t=10,b=0), xaxis_title="Forecast error (units)",
            yaxis_title="Count", legend=dict(orientation="h", y=1.12))
        fig_err.add_vline(x=0, line_dash="dash", line_color="gray")
        st.plotly_chart(fig_err, use_container_width=True)
        st.caption("A tighter distribution centered on 0 = better model. LightGBM should show fewer large errors than the naive baseline.")

        best = metrics.get("best_model", "lgbm")
        st.info(f"**Best model on holdout:** {MODEL_NAMES.get(best, best)} (lowest MAE)")

    except FileNotFoundError:
        st.error("Run `python run_pipeline.py` first.")
    except Exception as e:
        st.error(f"Error: {e}")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — COST DASHBOARD
# ══════════════════════════════════════════════════════════════════════════════
elif page == "💶 Cost Dashboard":
    st.title("💶 Cost Dashboard")
    st.caption("Inventory cost simulation — LightGBM vs all baselines. Baseline: seasonal naive (same week last year).")

    try:
        val = load_predictions()
        val["date"] = pd.to_datetime(val["date"])
        sim = compute_costs(val, holding_rate, stockout_rate)

        # Best model costs
        lgbm_cost    = sim["lgbm_total_cost"].sum()
        baseline_cost = sim["seasonal_naive_total_cost"].sum()
        savings      = baseline_cost - lgbm_cost
        pct          = savings / baseline_cost * 100 if baseline_cost > 0 else 0

        ca, cb, cc, cd = st.columns(4)
        ca.metric("Seasonal naive cost",  f"€{baseline_cost:,.0f}")
        cb.metric("LightGBM cost",        f"€{lgbm_cost:,.0f}")
        cc.metric("Savings (LightGBM)",   f"€{savings:,.0f}", delta=f"{pct:.1f}% less")
        cd.metric("Days evaluated",       str(sim["date"].nunique()))

        st.divider()

        # ── All models cost comparison ─────────────────────────────────────
        st.subheader("Total cost — all models vs baselines")
        cost_rows = []
        for model in ["lgbm", "xgb", "prophet", "seasonal_naive", "rolling_mean_28"]:
            col = f"{model}_total_cost"
            if col in sim.columns:
                cost_rows.append({
                    "Model":      MODEL_NAMES.get(model, model),
                    "Total cost": round(sim[col].sum(), 0),
                    "type":       "ML Model" if model in ["lgbm","xgb","prophet"] else "Baseline",
                })
        cost_df = pd.DataFrame(cost_rows).sort_values("Total cost")

        fig_cost = px.bar(cost_df, x="Total cost", y="Model", orientation="h",
            color="type",
            color_discrete_map={"ML Model":"#1D9E75","Baseline":"#E24B4A"},
            text="Total cost")
        fig_cost.update_traces(texttemplate="€%{text:,.0f}", textposition="outside")
        fig_cost.update_layout(height=320, margin=dict(l=0,r=0,t=10,b=0),
            yaxis=dict(autorange="reversed"), legend=dict(orientation="h", y=1.12))
        st.plotly_chart(fig_cost, use_container_width=True)

        cl, cr = st.columns(2)
        with cl:
            st.subheader("Savings by store (LightGBM vs seasonal naive)")
            sdf = summary_by_store(sim, "lgbm")
            if not sdf.empty:
                fig_s = px.bar(sdf, x="store_id", y="savings",
                    color="savings", color_continuous_scale=["#9FE1CB","#085041"])
                fig_s.update_layout(height=280, margin=dict(l=0,r=0,t=10,b=0),
                    coloraxis_showscale=False)
                st.plotly_chart(fig_s, use_container_width=True)

        with cr:
            st.subheader("Savings by category (LightGBM vs seasonal naive)")
            cdf = summary_by_category(sim, "lgbm")
            if not cdf.empty:
                fig_c = px.pie(cdf, names="cat", values="savings",
                    color_discrete_sequence=["#1D9E75","#5DCAA5","#9FE1CB"])
                fig_c.update_layout(height=280, margin=dict(l=0,r=0,t=10,b=0))
                st.plotly_chart(fig_c, use_container_width=True)

        st.subheader("Daily cost over time")
        daily = sim.groupby("date")[
            ["lgbm_total_cost", "seasonal_naive_total_cost"]
        ].sum().reset_index()
        fig_t = go.Figure()
        fig_t.add_trace(go.Scatter(x=daily["date"], y=daily["seasonal_naive_total_cost"],
            name="Seasonal naive", line=dict(color="#E24B4A", width=1.5)))
        fig_t.add_trace(go.Scatter(x=daily["date"], y=daily["lgbm_total_cost"],
            name="LightGBM", line=dict(color="#1D9E75", width=2)))
        fig_t.update_layout(height=300, margin=dict(l=0,r=0,t=10,b=0),
            yaxis_title='Daily cost (€)', hovermode='x unified',
            legend=dict(orientation="h", y=1.12))
        st.plotly_chart(fig_t, use_container_width=True)

    except FileNotFoundError:
        st.error("Run `python run_pipeline.py` first.")
    except Exception as e:
        st.error(f"Error: {e}")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 4 — MODEL INSIGHTS
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🔬 Model Insights":
    st.title("🔬 Model Insights")
    st.caption("Feature importance, SHAP explanations, and model transparency.")

    try:
        metrics = load_metrics()
        holdout = metrics.get("holdout", {})

        st.subheader("Holdout performance summary")
        ca, cb, cc, cd = st.columns(4)
        ca.metric("LightGBM MAE",  holdout.get("lgbm",{}).get("mae","—"))
        cb.metric("XGBoost MAE",   holdout.get("xgb",{}).get("mae","—"))
        cc.metric("Prophet MAE",   holdout.get("prophet",{}).get("mae","—"))
        cd.metric("Seasonal naive MAE", holdout.get("seasonal_naive",{}).get("mae","—"))

        st.divider()
        st.subheader("Feature importance — LightGBM")

        try:
            from src.explainer.shap_explainer import get_feature_importance
            fi = get_feature_importance(top_n=15)
            fig_fi = px.bar(fi, x="importance", y="label", orientation="h",
                color="importance", color_continuous_scale=["#9FE1CB","#085041"])
            fig_fi.update_layout(height=420, margin=dict(l=0,r=0,t=10,b=0),
                yaxis=dict(autorange="reversed"), coloraxis_showscale=False,
                xaxis_title="Importance score", yaxis_title="")
            st.plotly_chart(fig_fi, use_container_width=True)
        except Exception:
            model = load_lgbm()
            fi_df = pd.DataFrame({
                "feature":    get_feature_columns(),
                "importance": model.feature_importances_,
            }).sort_values("importance", ascending=False).head(15)
            fig_fi = px.bar(fi_df, x="importance", y="feature", orientation="h",
                color="importance", color_continuous_scale=["#9FE1CB","#085041"])
            fig_fi.update_layout(height=420, margin=dict(l=0,r=0,t=10,b=0),
                yaxis=dict(autorange="reversed"), coloraxis_showscale=False)
            st.plotly_chart(fig_fi, use_container_width=True)

        st.divider()
        st.subheader("CV MAE by fold — model stability")
        cv = metrics.get("cv_mae", {})
        cv_df = pd.DataFrame([
            {"Model": MODEL_NAMES.get(k,k), "CV MAE": v}
            for k, v in cv.items() if v is not None
        ]).sort_values("CV MAE")
        fig_cv = px.bar(cv_df, x="Model", y="CV MAE",
            color="CV MAE", color_continuous_scale=["#1D9E75","#E24B4A"],
            text="CV MAE")
        fig_cv.update_traces(texttemplate="%{text:.4f}", textposition="outside")
        fig_cv.update_layout(height=300, margin=dict(l=0,r=0,t=10,b=0),
            coloraxis_showscale=False)
        st.plotly_chart(fig_cv, use_container_width=True)

    except FileNotFoundError:
        st.error("Run `python run_pipeline.py` first.")
    except Exception as e:
        st.error(f"Error: {e}")
