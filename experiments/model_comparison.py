"""Logistic regression vs XGBoost for the two move-size models, on the current inputs.

Usage:
    uv run --with xgboost experiments/model_comparison.py

Same protocol as train_model.py: train on earlier years, test on each of 2022-2026.
For XGBoost the number of trees is picked by early stopping on the last training
year, then the model is refit on all training years with that many trees.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

sys.path.insert(0, str(Path(__file__).parent.parent))
import train_model as tm  # noqa: E402
from build_dataset import BIG_MOVE  # noqa: E402

XGB_CONFIGS = {
    "xgboost depth 3": dict(max_depth=3, min_child_weight=20),
    "xgboost depth 4": dict(max_depth=4, min_child_weight=20),
    "xgboost depth 6": dict(max_depth=6, min_child_weight=50),
}


def matrix(df: pd.DataFrame, num: list[str], sectors: list[str]) -> pd.DataFrame:
    x = df[num].astype(float).copy()
    for s in sectors:
        x[f"sector_{s}"] = (df["Sector"] == s).astype(float)
    return x


def fit_xgb(train: pd.DataFrame, test: pd.DataFrame, num: list[str], target: str, params: dict) -> np.ndarray:
    sectors = sorted(train["Sector"].dropna().unique())
    y = (train[target].abs() >= BIG_MOVE).astype(int)
    common = dict(learning_rate=0.03, subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
                  objective="binary:logistic", eval_metric="logloss", n_jobs=8, random_state=0, **params)
    last = train["earnings_date"].dt.year == train["earnings_date"].dt.year.max()
    probe = xgb.XGBClassifier(n_estimators=3000, early_stopping_rounds=100, **common)
    probe.fit(matrix(train[~last], num, sectors), y[~last],
              eval_set=[(matrix(train[last], num, sectors), y[last])], verbose=False)
    model = xgb.XGBClassifier(n_estimators=probe.best_iteration + 1, **common)
    model.fit(matrix(train, num, sectors), y)
    return model.predict_proba(matrix(test, num, sectors))[:, 1]


def evaluate(df: pd.DataFrame, prob: pd.Series, target: str) -> dict:
    year = df["earnings_date"].dt.year
    y = (df[target].abs() >= BIG_MOVE).astype(int)
    ok = prob.notna()
    by_year = [roc_auc_score(y[year == t], prob[year == t]) for t in tm.TEST_YEARS]
    return {"auc": np.mean(by_year), "by year": " ".join(f"{a:.3f}" for a in by_year),
            "log loss": log_loss(y[ok], prob[ok]), "brier": brier_score_loss(y[ok], prob[ok])}


if __name__ == "__main__":
    df = pd.read_parquet(tm.EVENTS)
    df = df[(df["lead"] == 1) & ~df["upcoming"]].replace([np.inf, -np.inf], np.nan).reset_index(drop=True)
    df["earnings_date"] = pd.to_datetime(df["earnings_date"])
    year = df["earnings_date"].dt.year

    for label, num, target in [("by the close", tm.NUM, "target_raw"), ("at the open", tm.GAP_NUM, "target_gap")]:
        rows_df = df.dropna(subset=[target])
        yr = rows_df["earnings_date"].dt.year
        results = [{"model": "logistic (current)", **evaluate(rows_df, tm.out_of_time(rows_df, num, target), target)}]
        for name, params in XGB_CONFIGS.items():
            prob = pd.Series(np.nan, index=rows_df.index)
            for t in tm.TEST_YEARS:
                prob[yr == t] = fit_xgb(rows_df[yr < t], rows_df[yr == t], num, target, params)
            results.append({"model": name, **evaluate(rows_df, prob, target)})
        print(f"\n== Move {label}: {BIG_MOVE:.0%}+ either way, test 2022-2026 ==")
        print(pd.DataFrame(results).round(4).to_string(index=False))
