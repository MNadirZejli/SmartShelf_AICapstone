"""
leave_one_store_out.py
SmartShelf — Leave-One-Store-Out generalization test.

Question: can the model forecast demand for a store it has NEVER seen in training?
We hide store identity (store_id & state_id are removed from the features),
train LightGBM on 2 stores, and predict the 3rd. Repeat for all 3 stores and
compare against the seasonal-naive baseline.

Same LGBM_PARAMS as train.py (imported, not copied) → guaranteed in sync.
Evaluation is restricted to the holdout period (date >= HOLDOUT_START) so the
numbers stay comparable with train.py.
Read-only: this script does not modify any existing file or saved model.
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error

from src.data.features import get_feature_columns
from src.models.train import LGBM_PARAMS, HOLDOUT_START

PROCESSED_DIR = Path("data/processed")
SEED = 42  # reproducibility (LGBM_PARAMS already carries seed=42; no other randomness here)

# Categorical columns to encode — store_id & state_id are intentionally EXCLUDED.
CAT_COLS = ["item_id", "cat_id", "dept_id"]


# ── Label encoding (same approach as train.py, minus store_id/state_id) ─────────
def fit_encoders(df: pd.DataFrame, cat_cols: list) -> dict:
    encoders = {}
    for col in cat_cols:
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
        known   = set(le.classes_)
        df[col] = df[col].apply(lambda x: x if x in known else le.classes_[0])
        df[col] = le.transform(df[col]).astype(np.int32)
    # Drop any remaining text columns (e.g. store_id/state_id) so they can't be used
    for col in df.select_dtypes(include="object").columns:
        df = df.drop(columns=[col])
    return df


# ── Leave-One-Store-Out ─────────────────────────────────────────────────────────
def run_leave_one_store_out() -> pd.DataFrame:
    print("Loading features ...")
    df = pd.read_parquet(PROCESSED_DIR / "sales_features.parquet")
    df["date"] = pd.to_datetime(df["date"])
    print(f"  Shape: {df.shape}")

    # The whole point: features WITHOUT store identity.
    feature_cols = [c for c in get_feature_columns() if c not in ("store_id", "state_id")]
    print(f"  Features used: {len(feature_cols)} (store_id & state_id removed)")

    # Fit encoders on the FULL data so items only present in the held-out store
    # are still known at encoding time.
    encoders = fit_encoders(df, CAT_COLS)

    stores  = sorted(df["store_id"].unique())
    results = []

    for test_store in stores:
        print(f"\n── Holding out store: {test_store} ──")
        train_pool = df[df["store_id"] != test_store].copy()
        # Test store, restricted to the holdout period (comparable with train.py).
        test_df = df[(df["store_id"] == test_store) &
                     (df["date"] >= HOLDOUT_START)].copy()

        # Early-stopping validation set carved from the TRAINING stores only
        # (the held-out store is never seen during training → no leakage).
        tr = train_pool[train_pool["date"] <  HOLDOUT_START]
        va = train_pool[train_pool["date"] >= HOLDOUT_START]

        X_tr = encode(tr[feature_cols], encoders)
        y_tr = tr["sales"].values
        X_va = encode(va[feature_cols], encoders)
        y_va = va["sales"].values

        model = lgb.LGBMRegressor(**LGBM_PARAMS)
        model.fit(
            X_tr, y_tr,
            eval_set=[(X_va, y_va)],
            categorical_feature=CAT_COLS,
            callbacks=[lgb.log_evaluation(0)],
        )

        # Predict on the UNSEEN store.
        X_test = encode(test_df[feature_cols], encoders)
        y_test = test_df["sales"].values
        preds  = np.clip(model.predict(X_test), 0, None)
        mae_model = mean_absolute_error(y_test, preds)

        # Baseline: seasonal naive (NaN → 0).
        baseline = test_df["seasonal_naive"].fillna(0).values
        mae_base = mean_absolute_error(y_test, baseline)

        improvement = (mae_base - mae_model) / mae_base * 100 if mae_base > 0 else 0.0
        results.append({
            "Held-out store": test_store,
            "Model MAE":      round(mae_model, 4),
            "Baseline MAE":   round(mae_base, 4),
            "Improvement %":  round(improvement, 1),
        })
        print(f"  Model MAE={mae_model:.4f}  Baseline MAE={mae_base:.4f}  "
              f"Improvement={improvement:.1f}%")

    table = pd.DataFrame(results)
    print("\n" + "=" * 64)
    print("LEAVE-ONE-STORE-OUT — generalization to an unseen store")
    print("=" * 64)
    print(table.to_string(index=False))
    print("\nReading: a positive 'Improvement %' means the model beats the "
          "seasonal-naive baseline\neven on a store it never saw during training.")
    return table


if __name__ == "__main__":
    run_leave_one_store_out()
