"""
shap_explainer.py
SHAP-based explanations for LightGBM predictions.
Converts feature contributions into plain-language bullets.
"""

import shap
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from src.data.features import get_feature_columns

MODEL_DIR = Path("outputs/models")

FEATURE_LABELS = {
    "lag_7":            "sales 1 week ago",
    "lag_14":           "sales 2 weeks ago",
    "lag_21":           "sales 3 weeks ago",
    "lag_28":           "sales 4 weeks ago",
    "rolling_mean_7":   "avg sales last 7 days",
    "rolling_mean_14":  "avg sales last 14 days",
    "rolling_mean_28":  "avg sales last 28 days",
    "rolling_std_7":    "sales variability (7d)",
    "rolling_std_28":   "sales variability (28d)",
    "rolling_max_7":    "peak sales (7d)",
    "rolling_max_28":   "peak sales (28d)",
    "sell_price":       "current selling price",
    "price_change":     "recent price change",
    "price_vs_avg":     "price vs. store average",
    "day_of_week":      "day of week",
    "day_of_month":     "day of month",
    "week_of_year":     "week of year",
    "month":            "month of year",
    "quarter":          "quarter",
    "is_weekend":       "weekend effect",
    "is_month_end":     "month-end effect",
    "has_event":        "special event nearby",
    "snap_day":         "SNAP benefit day",
    "demand_7d":        "total demand last 7 days",
    "demand_28d":       "total demand last 28 days",
    "sales_velocity":   "recent vs long-term trend",
    "zero_streak":      "consecutive zero-sales days",
    "item_id":          "item identity",
    "store_id":         "store identity",
    "cat_id":           "product category",
    "dept_id":          "department",
    "state_id":         "state",
}


def explain_prediction(X_row: pd.DataFrame, top_n: int = 4) -> list:
    model    = joblib.load(MODEL_DIR / "lgbm_model.pkl")
    explainer = shap.TreeExplainer(model)
    sv        = explainer.shap_values(X_row)
    if isinstance(sv, list):
        sv = sv[0]
    vals     = sv[0]
    features = get_feature_columns()

    df = pd.DataFrame({
        "feature":   features,
        "shap":      vals,
        "abs_shap":  np.abs(vals),
    }).sort_values("abs_shap", ascending=False)

    bullets = []
    for _, row in df.head(top_n).iterrows():
        label     = FEATURE_LABELS.get(row["feature"], row["feature"])
        direction = "increases" if row["shap"] > 0 else "decreases"
        magnitude = "strongly" if abs(row["shap"]) > 2 else "moderately" if abs(row["shap"]) > 0.5 else "slightly"
        sign      = "+" if row["shap"] > 0 else ""
        bullets.append(f"{label.capitalize()} {magnitude} {direction} demand ({sign}{row['shap']:.2f} units)")

    return bullets


def get_feature_importance(top_n: int = 15) -> pd.DataFrame:
    model    = joblib.load(MODEL_DIR / "lgbm_model.pkl")
    features = get_feature_columns()
    return (
        pd.DataFrame({
            "feature":    features,
            "label":      [FEATURE_LABELS.get(f, f) for f in features],
            "importance": model.feature_importances_,
        })
        .sort_values("importance", ascending=False)
        .head(top_n)
    )
