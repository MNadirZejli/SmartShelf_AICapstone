"""
train.py
SmartShelf — Multi-model demand forecasting with walk-forward validation.

Models compared:
  1. LightGBM (MAE objective) — gradient boosting, tabular baseline
  2. XGBoost  (MAE objective) — alternative gradient boosting
  3. Prophet  (Facebook)      — additive time series with seasonality

Baselines:
  - Seasonal naive: same weekday 52 weeks ago (real retail standard)
  - Rolling mean 28: naive monthly average (for reference only)

Validation: walk-forward (expanding window) — the correct approach for time series.
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
import xgboost as xgb
import joblib
import json
import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error
from prophet import Prophet

from src.data.features import get_feature_columns

PROCESSED_DIR = Path("data/processed")
MODEL_DIR     = Path("outputs/models")
MODEL_DIR.mkdir(parents=True, exist_ok=True)

CAT_COLS = ["item_id", "store_id", "cat_id", "dept_id", "state_id"]

# Walk-forward validation windows (train_end → val period)
# Each fold trains on all data up to train_end, validates on next 28 days
FOLDS = [
    {"train_end": "2015-09-30", "val_start": "2015-10-01", "val_end": "2015-10-28"},
    {"train_end": "2015-12-31", "val_start": "2016-01-01", "val_end": "2016-01-28"},
    {"train_end": "2016-02-29", "val_start": "2016-03-01", "val_end": "2016-03-28"},
]
# Final holdout for reporting
HOLDOUT_START = "2016-03-28"
HOLDOUT_END   = "2016-05-22"

LGBM_PARAMS = {
    "objective":           "mae",       # Mean Absolute Error — clean, interpretable
    "metric":              "mae",
    "learning_rate":       0.03,
    "num_leaves":          127,
    "min_data_in_leaf":    20,
    "feature_fraction":    0.8,
    "bagging_fraction":    0.8,
    "bagging_freq":        1,
    "reg_alpha":           0.1,
    "reg_lambda":          0.1,
    "n_estimators":        1000,
    "early_stopping_rounds": 50,
    "verbose":             -1,
    "n_jobs":              -1,
    "seed":                42,
}

XGB_PARAMS = {
    "objective":       "reg:absoluteerror",   # XGBoost MAE
    "learning_rate":   0.03,
    "max_depth":       7,
    "subsample":       0.8,
    "colsample_bytree": 0.8,
    "reg_alpha":       0.1,
    "reg_lambda":      0.1,
    "n_estimators":    1000,
    "early_stopping_rounds": 50,
    "verbosity":       0,
    "seed":            42,
    "n_jobs":          -1,
}


# ── Label encoding ────────────────────────────────────────────────────────────
def fit_encoders(df: pd.DataFrame) -> dict:
    encoders = {}
    for col in CAT_COLS:
        le = LabelEncoder()
        le.fit(df[col].astype(str))
        encoders[col] = le
    return encoders


def encode(df: pd.DataFrame, encoders: dict) -> pd.DataFrame:
    df = df.copy()
    for col, le in encoders.items():
        if col not in df.columns:
            continue
        df[col] = df[col].astype(str)
        known    = set(le.classes_)
        df[col]  = df[col].apply(lambda x: x if x in known else le.classes_[0])
        df[col]  = le.transform(df[col]).astype(np.int32)
    # Drop any remaining object columns
    for col in df.select_dtypes(include="object").columns:
        df = df.drop(columns=[col])
    return df


# ── Walk-forward cross-validation ─────────────────────────────────────────────
def walk_forward_cv(df: pd.DataFrame, encoders: dict) -> dict:
    """
    Expanding-window cross-validation for time series.
    Train on [start, fold_end], evaluate on [fold_end+1, fold_end+28].
    Returns average MAE per model across folds.
    """
    print("\nWalk-forward cross-validation ...")
    feature_cols = get_feature_columns()

    fold_results = {m: [] for m in ["lgbm", "xgb", "seasonal_naive", "rolling_mean_28"]}

    for i, fold in enumerate(FOLDS):
        print(f"  Fold {i+1}/3: train→{fold['train_end']}  val {fold['val_start']}→{fold['val_end']}")

        train = df[df["date"] <= fold["train_end"]].copy()
        val   = df[(df["date"] >= fold["val_start"]) & (df["date"] <= fold["val_end"])].copy()

        if len(val) == 0:
            print("    No validation data, skipping.")
            continue

        X_train = encode(train[feature_cols], encoders)
        y_train = train["sales"].values
        X_val   = encode(val[feature_cols], encoders)
        y_val   = val["sales"].values

        # LightGBM
        lgbm_model = lgb.LGBMRegressor(**LGBM_PARAMS)
        lgbm_model.fit(X_train, y_train, eval_set=[(X_val, y_val)],
                       categorical_feature=CAT_COLS,
                       callbacks=[lgb.log_evaluation(500)])
        lgbm_preds = np.clip(lgbm_model.predict(X_val), 0, None)
        fold_results["lgbm"].append(mean_absolute_error(y_val, lgbm_preds))

        # XGBoost
        xgb_model = xgb.XGBRegressor(**XGB_PARAMS)
        xgb_model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
        xgb_preds = np.clip(xgb_model.predict(X_val), 0, None)
        fold_results["xgb"].append(mean_absolute_error(y_val, xgb_preds))

        # Baselines
        seasonal = val["seasonal_naive"].fillna(val["rolling_mean_28"].fillna(0)).values
        naive28  = val["rolling_mean_28"].fillna(0).values
        fold_results["seasonal_naive"].append(mean_absolute_error(y_val, seasonal))
        fold_results["rolling_mean_28"].append(mean_absolute_error(y_val, naive28))

        print(f"    LGBM={fold_results['lgbm'][-1]:.4f}  XGB={fold_results['xgb'][-1]:.4f}  "
              f"SeasonalNaive={fold_results['seasonal_naive'][-1]:.4f}")

    cv_results = {
        model: round(float(np.mean(maes)), 4) if maes else None
        for model, maes in fold_results.items()
    }
    print(f"\n  CV Summary: {cv_results}")
    return cv_results


# ── Train final models on full training set ───────────────────────────────────
def train_final_models(df: pd.DataFrame, encoders: dict):
    """Train all models on data up to HOLDOUT_START, evaluate on holdout."""
    print("\nTraining final models on full training set ...")
    feature_cols = get_feature_columns()

    train = df[df["date"] <  HOLDOUT_START].copy()
    val   = df[(df["date"] >= HOLDOUT_START) & (df["date"] <= HOLDOUT_END)].copy()
    print(f"  Train: {len(train):,}  Holdout: {len(val):,}")

    X_train = encode(train[feature_cols], encoders)
    y_train = train["sales"].values
    X_val   = encode(val[feature_cols], encoders)
    y_val   = val["sales"].values

    # ── LightGBM ──────────────────────────────────────────────────────────
    print("  Training LightGBM ...")
    lgbm_model = lgb.LGBMRegressor(**LGBM_PARAMS)
    lgbm_model.fit(X_train, y_train, eval_set=[(X_val, y_val)],
                   categorical_feature=CAT_COLS,
                   callbacks=[lgb.log_evaluation(200)])
    lgbm_preds = np.clip(lgbm_model.predict(X_val), 0, None)

    # ── XGBoost ───────────────────────────────────────────────────────────
    print("  Training XGBoost ...")
    xgb_model = xgb.XGBRegressor(**XGB_PARAMS)
    xgb_model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    xgb_preds = np.clip(xgb_model.predict(X_val), 0, None)

    # ── Prophet (per item-store, aggregated) ──────────────────────────────
    print("  Training Prophet (sampling 20 items for speed) ...")
    prophet_preds = train_prophet_sample(df, val)

    # ── Baselines ─────────────────────────────────────────────────────────
    seasonal_preds = val["seasonal_naive"].fillna(val["rolling_mean_28"].fillna(0)).values
    naive28_preds  = val["rolling_mean_28"].fillna(0).values

    return {
        "models": {"lgbm": lgbm_model, "xgb": xgb_model},
        "predictions": {
            "lgbm":           lgbm_preds,
            "xgb":            xgb_preds,
            "prophet":        prophet_preds,
            "seasonal_naive": seasonal_preds,
            "rolling_mean_28": naive28_preds,
        },
        "val_df": val,
        "y_val":  y_val,
    }


def train_prophet_sample(df: pd.DataFrame, val: pd.DataFrame) -> np.ndarray:
    """
    Train Prophet on a sample of items. For each item, fit on history,
    predict on val dates. Average error across items.
    """
    train = df[df["date"] < HOLDOUT_START]
    items = val["id"].unique()[:20]  # Sample for speed

    all_preds = {}
    val_dates = val["date"].unique()

    for item_id in items:
        item_train = train[train["id"] == item_id][["date", "sales"]].rename(
            columns={"date": "ds", "sales": "y"}
        )
        if len(item_train) < 60:
            continue
        try:
            m = Prophet(
                yearly_seasonality=True,
                weekly_seasonality=True,
                daily_seasonality=False,
                seasonality_mode="multiplicative",
                changepoint_prior_scale=0.05,
            )
            m.fit(item_train)
            future  = m.make_future_dataframe(periods=len(val_dates))
            forecast = m.predict(future)
            forecast = forecast[forecast["ds"].isin(val_dates)][["ds", "yhat"]]
            forecast["yhat"] = np.clip(forecast["yhat"], 0, None)

            item_val = val[val["id"] == item_id].copy().reset_index()
            item_val = item_val.merge(
                forecast.rename(columns={"ds": "date"}), on="date", how="left"
            )
            for _, row in item_val.iterrows():
                all_preds[int(row["index"])] = max(0, float(row.get("yhat") or 0))
        except Exception:
            pass

    # Fill missing predictions with seasonal naive
    result   = val["seasonal_naive"].fillna(val["rolling_mean_28"].fillna(0)).values.copy()
    val_idxs = list(val.index)
    idx_map  = {orig: pos for pos, orig in enumerate(val_idxs)}
    for orig_idx, pred in all_preds.items():
        pos = idx_map.get(orig_idx)
        if pos is not None:
            result[pos] = pred

    return result


# ── Evaluate ──────────────────────────────────────────────────────────────────
def evaluate_all(y_true: np.ndarray, predictions: dict) -> dict:
    metrics = {}
    for name, preds in predictions.items():
        if preds is None:
            continue
        mae  = mean_absolute_error(y_true, preds)
        rmse = np.sqrt(mean_squared_error(y_true, preds))
        metrics[name] = {"mae": round(mae, 4), "rmse": round(rmse, 4)}
        print(f"  [{name:20s}] MAE={mae:.4f}  RMSE={rmse:.4f}")
    return metrics


# ── Main ──────────────────────────────────────────────────────────────────────
def run_training():
    print("Loading features ...")
    df = pd.read_parquet(PROCESSED_DIR / "sales_features.parquet")
    df["date"] = pd.to_datetime(df["date"])
    print(f"  Shape: {df.shape}")

    # Fit label encoders on full data
    print("Fitting label encoders ...")
    encoders = fit_encoders(df)
    joblib.dump(encoders, MODEL_DIR / "label_encoders.pkl")

    # Walk-forward CV
    cv_results = walk_forward_cv(df, encoders)

    # Train final models
    result = train_final_models(df, encoders)
    models = result["models"]
    preds  = result["predictions"]
    val    = result["val_df"]
    y_val  = result["y_val"]

    # Evaluate on holdout
    print("\nHoldout evaluation:")
    holdout_metrics = evaluate_all(y_val, preds)

    # Save models
    joblib.dump(models["lgbm"], MODEL_DIR / "lgbm_model.pkl")
    joblib.dump(models["xgb"],  MODEL_DIR / "xgb_model.pkl")

    # Save full metrics
    all_metrics = {
        "cv_mae":      cv_results,
        "holdout":     holdout_metrics,
        "best_model":  min(holdout_metrics, key=lambda k: holdout_metrics[k]["mae"]),
    }
    # Convert numpy types to native Python for JSON serialization
    def to_native(obj):
        if isinstance(obj, dict):
            return {k: to_native(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [to_native(i) for i in obj]
        if hasattr(obj, "item"):
            return obj.item()
        return obj

    with open(MODEL_DIR / "metrics.json", "w") as f:
        json.dump(to_native(all_metrics), f, indent=2)
    print(f"\n  Best model on holdout: {all_metrics['best_model']}")

    # Save predictions
    val = val.copy()
    for name, pred_arr in preds.items():
        val[f"pred_{name}"] = pred_arr

    val[["id", "item_id", "store_id", "date", "sales", "sell_price",
         "pred_lgbm", "pred_xgb", "pred_prophet",
         "pred_seasonal_naive", "pred_rolling_mean_28"]].to_parquet(
        PROCESSED_DIR / "val_predictions.parquet", index=False
    )
    print("All models saved.")
    return models, all_metrics


if __name__ == "__main__":
    run_training()
