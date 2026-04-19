"""
run_pipeline.py
SmartShelf v2 — Full pipeline.
3 models (LightGBM + XGBoost + Prophet) + walk-forward CV + seasonal naive baseline.

Usage: python run_pipeline.py
"""

import sys
import time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))


def step(n, msg):
    print(f"\n{'='*65}")
    print(f"  Step {n} — {msg}")
    print(f"{'='*65}")


def main():
    t0 = time.time()

    step("1/4", "Loading M5 data (3 stores × top 100 products)")
    from src.data.loader import run_pipeline as load_data
    load_data()

    step("2/4", "Feature engineering (28 features)")
    import pandas as pd
    from src.data.features import build_features
    df = pd.read_parquet("data/processed/sales_merged.parquet")
    df = build_features(df)
    df.to_parquet("data/processed/sales_features.parquet", index=False)
    print(f"  Features saved. Shape: {df.shape}")
    del df

    step("3/4", "Training 3 models + walk-forward CV")
    from src.models.train import run_training
    models, metrics = run_training()
    print(f"\n  Best model: {metrics['best_model']}")
    print(f"  Holdout results:")
    for name, m in metrics["holdout"].items():
        print(f"    {name:22s} MAE={m['mae']}  RMSE={m['rmse']}")

    step("4/4", "Cost simulation — all models vs seasonal naive")
    import pandas as pd
    from src.cost.simulator import compute_costs, summary_by_model
    val = pd.read_parquet("data/processed/val_predictions.parquet")
    sim = compute_costs(val)
    print(f"\n  Cost summary:")
    print(summary_by_model(sim).to_string(index=False))

    elapsed = time.time() - t0
    print(f"\n{'='*65}")
    print(f"  Pipeline complete in {elapsed/60:.1f} minutes")
    print(f"  Launch app: streamlit run app/app.py")
    print(f"{'='*65}\n")


if __name__ == "__main__":
    main()
