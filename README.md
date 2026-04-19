# SmartShelf v2 — AI Replenishment Assistant

> Multi-model demand forecasting with walk-forward validation, translating forecast errors into measurable inventory cost savings.

---

## What changed in v2

| v1 (rejected) | v2 (this version) |
|---|---|
| Single model (LightGBM Tweedie) | **3 models**: LightGBM (MAE) + XGBoost + Prophet |
| Single train/test split | **Walk-forward cross-validation** (3 folds, expanding window) |
| Rolling 28-day average as baseline | **Seasonal naive** (same weekday 52 weeks ago — real retail standard) |
| Basic order recommendation | **Stockout risk alerts** + SHAP explanations per recommendation |
| 3-page app | **4-page app** with dedicated model comparison page |

---

## Models

| Model | Objective | Why |
|---|---|---|
| **LightGBM** | MAE (mean absolute error) | Industry standard for tabular retail forecasting — fast, handles categoricals natively |
| **XGBoost** | MAE | Alternative gradient boosting for comparison — different regularization approach |
| **Prophet** | Additive decomposition | Facebook's production forecasting model — handles seasonality and holidays natively |

### Baselines

| Baseline | Definition | Why it's realistic |
|---|---|---|
| **Seasonal naive** | Predict same weekday 52 weeks ago | What retailers actually use — accounts for yearly seasonality |
| Rolling mean 28d | 28-day average (kept for reference) | Simple heuristic — shown to be clearly inferior |

---

## Validation strategy

We use **walk-forward (expanding window) cross-validation** — the correct approach for time series:

```
Fold 1: Train [Jan 2011 → Sep 2015]  → Predict [Oct 2015]
Fold 2: Train [Jan 2011 → Dec 2015]  → Predict [Jan 2016]
Fold 3: Train [Jan 2011 → Feb 2016]  → Predict [Mar 2016]
Final holdout: Train [Jan 2011 → Mar 2016] → Predict [Apr–May 2016]
```

This avoids data leakage and simulates real deployment conditions.

---

## Features (28 total)

| Family | Features |
|---|---|
| Historical demand | lag_7, lag_14, lag_21, lag_28 |
| Rolling statistics | rolling_mean_7/14/28, rolling_std_7/28, rolling_max_7/28 |
| Price signals | sell_price, price_change, price_vs_store_avg |
| Calendar & events | day_of_week, day_of_month, week_of_year, month, quarter, is_weekend, is_month_end, has_event, snap_day |
| Demand pressure | demand_7d, demand_28d, sales_velocity, zero_streak |
| Identifiers | item_id, store_id, cat_id, dept_id, state_id |

---

## Project structure

```
smartshelf/
├── src/
│   ├── data/
│   │   ├── loader.py        ← M5 data loading (3 stores, top 100 products)
│   │   └── features.py      ← 28 features across 5 families
│   ├── models/
│   │   ├── train.py         ← 3 models + walk-forward CV + baselines
│   │   └── predict.py       ← Inference + order quantity + risk detection
│   ├── cost/
│   │   └── simulator.py     ← Overstock/stockout cost engine (all models)
│   └── explainer/
│       └── shap_explainer.py ← SHAP feature explanations
├── app/
│   └── app.py               ← 4-page Streamlit dashboard
├── data/raw/                ← M5 CSV files (not tracked)
├── data/processed/          ← Generated parquets (not tracked)
├── outputs/models/          ← Trained models (not tracked)
├── run_pipeline.py          ← Run everything in one command
└── requirements.txt
```

---

## Setup

```bash
git clone https://github.com/MNadirZejli/smartshelf_capstone.git
cd smartshelf_capstone
python -m venv venv
venv\Scripts\activate        # Windows
pip install -r requirements.txt
```

Place M5 CSV files in `data/raw/`:
- `sales_train_evaluation.csv`
- `calendar.csv`
- `sell_prices.csv`

---

## Run

```bash
python run_pipeline.py    # ~10 minutes, runs all 4 steps
streamlit run app/app.py  # Opens at http://localhost:8501
```

---

## Cost model

| Parameter | Value | Source |
|---|---|---|
| Holding cost | 0.068%/day (25%/year) | Silver, Pyke & Thomas (1998) |
| Stockout cost | 75% of item value | ECR Europe (2003) |

---

## Scope

- **3 stores:** CA_1 (California), TX_1 (Texas), WI_1 (Wisconsin) — one per state
- **100 products:** Top 100 by total sales volume
- **~582,000 rows** — runs on a standard laptop
- **Architecture designed to scale** to all 10 stores and 3,049 products

---

## References

- Makridakis et al. (2022). M5 accuracy competition. *International Journal of Forecasting.*
- Ke et al. (2017). LightGBM. *NeurIPS.*
- Chen & Guestrin (2016). XGBoost. *KDD.*
- Taylor & Letham (2018). Prophet. *The American Statistician.*
- Silver, Pyke & Thomas (1998). *Inventory and Production Management in Supply Chains.*
- Lundberg & Lee (2017). SHAP. *NeurIPS.*
