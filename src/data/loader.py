"""
loader.py
Loads M5 data for 3 stores (CA_1, TX_1, WI_1) and top 100 products.
Memory-efficient: writes each store separately, combines via ParquetWriter.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import pyarrow as pa
import pyarrow.parquet as pq
import gc

RAW_DIR       = Path("data/raw")
PROCESSED_DIR = Path("data/processed")
STORE_DIR     = PROCESSED_DIR / "stores"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
STORE_DIR.mkdir(parents=True, exist_ok=True)

STORES     = ["CA_1", "TX_1", "WI_1"]
N_PRODUCTS = 100
CAL_COLS   = ["d", "date", "wm_yr_wk", "weekday", "wday", "month", "year",
               "event_name_1", "event_type_1", "snap_CA", "snap_TX", "snap_WI"]
ID_COLS    = ["id", "item_id", "dept_id", "cat_id", "store_id", "state_id"]


def run_pipeline() -> None:
    out_path = PROCESSED_DIR / "sales_merged.parquet"
    if out_path.exists():
        print("  sales_merged.parquet already exists, skipping step 1.")
        return

    print("Loading calendar.csv ...")
    calendar = pd.read_csv(RAW_DIR / "calendar.csv", usecols=CAL_COLS)
    calendar["date"] = pd.to_datetime(calendar["date"])

    print("Loading sell_prices.csv ...")
    prices = pd.read_csv(RAW_DIR / "sell_prices.csv", dtype={"sell_price": np.float32})

    print("Loading sales data ...")
    sales = pd.read_csv(RAW_DIR / "sales_train_evaluation.csv")
    sales = sales[sales["store_id"].isin(STORES)].copy()

    # Top N products by total sales across all stores
    day_cols = [c for c in sales.columns if c.startswith("d_")]
    sales["total_sales"] = sales[day_cols].sum(axis=1)
    top_items = (
        sales.groupby("item_id")["total_sales"].sum()
        .nlargest(N_PRODUCTS).index.tolist()
    )
    sales = sales[sales["item_id"].isin(top_items)].drop(columns=["total_sales"])
    print(f"  {len(STORES)} stores × {N_PRODUCTS} products selected")

    writer = None
    for store in STORES:
        store_path = STORE_DIR / f"{store}.parquet"
        if store_path.exists():
            print(f"  {store} already processed, skipping.")
        else:
            print(f"  Processing {store} ...")
            gc.collect()
            store_sales = sales[sales["store_id"] == store].copy()

            long = store_sales.melt(
                id_vars=ID_COLS, value_vars=day_cols,
                var_name="d", value_name="sales"
            )
            del store_sales

            long = long.merge(calendar, on="d", how="left")
            store_prices = prices[prices["store_id"] == store][
                ["item_id", "wm_yr_wk", "sell_price"]
            ]
            long = long.merge(store_prices, on=["item_id", "wm_yr_wk"], how="left")
            long = long.sort_values(["id", "date"])
            long["sell_price"] = (
                long.groupby("id")["sell_price"]
                .transform(lambda x: x.ffill().bfill())
            )
            long = long.dropna(subset=["sell_price"])
            long["sales"]      = long["sales"].astype(np.int16)
            long["sell_price"] = long["sell_price"].astype(np.float32)
            long.to_parquet(str(store_path), index=False)
            print(f"    Saved {len(long):,} rows → {store_path.name}")
            del long
            gc.collect()

        # Append to master file
        table = pq.read_table(str(store_path))
        if writer is None:
            writer = pq.ParquetWriter(str(out_path), table.schema)
        writer.write_table(table)
        del table

    if writer:
        writer.close()
    print(f"Done. Saved to {out_path}")


if __name__ == "__main__":
    run_pipeline()
