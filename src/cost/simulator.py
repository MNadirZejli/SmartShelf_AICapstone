"""
simulator.py
Inventory cost simulation for all models.
Overstock + stockout cost translation using industry-standard rates.
"""

import pandas as pd
import numpy as np
from pathlib import Path

PROCESSED_DIR    = Path("data/processed")
DEFAULT_HOLDING  = 0.00068   # 25%/year = 0.068%/day (Silver, Pyke & Thomas, 1998)
DEFAULT_STOCKOUT = 0.75      # 75% of item value (ECR Europe, 2003)

MODELS = ["lgbm", "xgb", "prophet", "seasonal_naive", "rolling_mean_28"]


def compute_costs(val: pd.DataFrame,
                  holding_rate: float  = DEFAULT_HOLDING,
                  stockout_rate: float = DEFAULT_STOCKOUT) -> pd.DataFrame:
    val   = val.copy()
    price = val["sell_price"].values

    for model in MODELS:
        col = f"pred_{model}"
        if col not in val.columns:
            continue
        err  = val[col].values - val["sales"].values
        over = np.clip(err,  0, None) * price * holding_rate
        out  = np.clip(-err, 0, None) * price * stockout_rate
        val[f"{model}_overstock_cost"] = over
        val[f"{model}_stockout_cost"]  = out
        val[f"{model}_total_cost"]     = over + out

    # Savings vs seasonal naive (the real-world benchmark)
    if "pred_seasonal_naive" in val.columns:
        for model in ["lgbm", "xgb", "prophet"]:
            if f"{model}_total_cost" in val.columns:
                val[f"{model}_savings"] = (
                    val["seasonal_naive_total_cost"] - val[f"{model}_total_cost"]
                )
    return val


def summary_by_model(val: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for model in MODELS:
        col = f"{model}_total_cost"
        if col not in val.columns:
            continue
        rows.append({
            "model":      model,
            "total_cost": round(val[col].sum(), 2),
            "mae":        round(mean_absolute_error_simple(val["sales"].values,
                                                           val[f"pred_{model}"].values), 4),
        })
    df = pd.DataFrame(rows).sort_values("total_cost")
    return df


def mean_absolute_error_simple(y_true, y_pred):
    return float(np.mean(np.abs(y_true - y_pred)))


def summary_by_store(val: pd.DataFrame, model: str = "lgbm") -> pd.DataFrame:
    col = f"{model}_total_cost"
    ref = "seasonal_naive_total_cost"
    if col not in val.columns or ref not in val.columns:
        return pd.DataFrame()
    return (
        val.groupby("store_id")
        .agg(model_cost=(col, "sum"), baseline_cost=(ref, "sum"))
        .assign(savings=lambda d: d["baseline_cost"] - d["model_cost"])
        .round(2).reset_index()
        .sort_values("savings", ascending=False)
    )


def summary_by_category(val: pd.DataFrame, model: str = "lgbm") -> pd.DataFrame:
    col = f"{model}_total_cost"
    ref = "seasonal_naive_total_cost"
    v   = val.copy()
    v["cat"] = v["item_id"].str.split("_").str[0]
    if col not in v.columns:
        return pd.DataFrame()
    return (
        v.groupby("cat")
        .agg(model_cost=(col, "sum"), baseline_cost=(ref, "sum"))
        .assign(savings=lambda d: d["baseline_cost"] - d["model_cost"])
        .round(2).reset_index()
        .sort_values("savings", ascending=False)
    )


if __name__ == "__main__":
    val = pd.read_parquet(PROCESSED_DIR / "val_predictions.parquet")
    sim = compute_costs(val)
    print(summary_by_model(sim))
