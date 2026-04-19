"""
features.py
Feature engineering for SmartShelf demand forecasting.
No exotic distribution assumptions. Clean, interpretable features.
"""

import pandas as pd
import numpy as np
from pathlib import Path

PROCESSED_DIR = Path("data/processed")
LAG_DAYS      = [7, 14, 21, 28]
ROLL_WINDOWS  = [7, 14, 28]


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    print("Engineering features ...")
    df = df.sort_values(["id", "date"]).copy()

    # ── Lag features ──────────────────────────────────────────────────────
    for lag in LAG_DAYS:
        df[f"lag_{lag}"] = df.groupby("id")["sales"].shift(lag)

    # ── Rolling statistics ────────────────────────────────────────────────
    for w in ROLL_WINDOWS:
        shifted = df.groupby("id")["sales"].shift(1)
        df[f"rolling_mean_{w}"] = shifted.rolling(w).mean().reset_index(level=0, drop=True)
        df[f"rolling_std_{w}"]  = shifted.rolling(w).std().reset_index(level=0, drop=True)
        df[f"rolling_max_{w}"]  = shifted.rolling(w).max().reset_index(level=0, drop=True)

    # ── Seasonal naive: same weekday 52 weeks ago (real retail benchmark) ─
    df["seasonal_naive"] = df.groupby("id")["sales"].shift(364)

    # ── Price features ────────────────────────────────────────────────────
    df["price_lag_7"]    = df.groupby("id")["sell_price"].shift(7)
    df["price_change"]   = df["sell_price"] - df["price_lag_7"]
    store_avg            = df.groupby(["store_id", "wm_yr_wk"])["sell_price"].transform("mean")
    df["price_vs_avg"]   = df["sell_price"] / (store_avg + 1e-8)

    # ── Calendar features ─────────────────────────────────────────────────
    df["day_of_week"]  = df["date"].dt.dayofweek
    df["day_of_month"] = df["date"].dt.day
    df["week_of_year"] = df["date"].dt.isocalendar().week.astype(int)
    df["month"]        = df["date"].dt.month
    df["quarter"]      = df["date"].dt.quarter
    df["is_weekend"]   = (df["day_of_week"] >= 5).astype(np.int8)
    df["is_month_end"] = df["date"].dt.is_month_end.astype(np.int8)
    df["has_event"]    = df["event_name_1"].notna().astype(np.int8)

    # ── SNAP flag ─────────────────────────────────────────────────────────
    snap_map = {"CA": "snap_CA", "TX": "snap_TX", "WI": "snap_WI"}
    df["snap_day"] = 0
    for state, col in snap_map.items():
        if col in df.columns:
            df.loc[df["state_id"] == state, "snap_day"] = (
                df.loc[df["state_id"] == state, col].astype(np.int8)
            )

    # ── Demand pressure ───────────────────────────────────────────────────
    df["demand_7d"]  = df.groupby("id")["sales"].transform(
        lambda x: x.shift(1).rolling(7).sum()
    )
    df["demand_28d"] = df.groupby("id")["sales"].transform(
        lambda x: x.shift(1).rolling(28).sum()
    )
    df["sales_velocity"] = df["rolling_mean_7"] / (df["rolling_mean_28"] + 1e-8)

    # ── Zero-sales streak (stockout signal) ───────────────────────────────
    df["is_zero"]    = (df["sales"] == 0).astype(int)
    df["zero_streak"] = df.groupby("id")["is_zero"].transform(
        lambda x: x.groupby((x != x.shift()).cumsum()).cumcount()
    )
    df = df.drop(columns=["is_zero"])

    # ── Drop NaN warmup rows (first ~28 rows per item) ───────────────────
    lag_cols = [f"lag_{l}" for l in LAG_DAYS]
    before   = len(df)
    df       = df.dropna(subset=lag_cols)
    print(f"  Dropped {before - len(df):,} warmup rows. Final shape: {df.shape}")
    return df


def get_feature_columns() -> list:
    return [
        "lag_7", "lag_14", "lag_21", "lag_28",
        "rolling_mean_7", "rolling_mean_14", "rolling_mean_28",
        "rolling_std_7",  "rolling_std_28",
        "rolling_max_7",  "rolling_max_28",
        "sell_price", "price_change", "price_vs_avg",
        "day_of_week", "day_of_month", "week_of_year",
        "month", "quarter", "is_weekend", "is_month_end",
        "has_event", "snap_day",
        "demand_7d", "demand_28d", "sales_velocity", "zero_streak",
        "item_id", "store_id", "cat_id", "dept_id", "state_id",
    ]


if __name__ == "__main__":
    df = pd.read_parquet(PROCESSED_DIR / "sales_merged.parquet")
    df = build_features(df)
    df.to_parquet(PROCESSED_DIR / "sales_features.parquet", index=False)
    print("Features saved.")
