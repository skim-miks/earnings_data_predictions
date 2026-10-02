"""Class balance at 3% and 5%, raw vs SPY-adjusted, and signal by feature lead time.

Usage:
    uv run experiments/target_and_lead_time.py

Validation is expanding-window by calendar year: train on everything before
the test year, test on that year. Reported AUCs are the mean across test years.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

sys.path.insert(0, str(Path(__file__).parent.parent))
from build_dataset import OUT  # noqa: E402

TARGETS = ["target_raw", "target_vs_spy"]
THRESHOLDS = [0.03, 0.05]
TEST_YEARS = [2022, 2023, 2024, 2025, 2026]
CAT = ["quarter", "Sector", "Industry", "HeadquartersState", "report_timing"]
NOT_FEATURES = {"upcoming", "surprise_actual", "ticker", "earnings_date", "reaction_date", "lead", "feature_date", *TARGETS, *CAT}
CLIP = (0.005, 0.995)   # per-feature quantile clip, fit on train


def models(num: list[str]) -> dict[str, Pipeline]:
    numeric = Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())])
    onehot = OneHotEncoder(handle_unknown="ignore", min_frequency=30)
    return {
        "logistic": Pipeline([
            ("prep", ColumnTransformer([("num", numeric, num), ("cat", onehot, CAT)])),
            ("clf", LogisticRegression(max_iter=2000, C=0.1)),
        ]),
        "boosting": Pipeline([
            ("prep", ColumnTransformer([("num", "passthrough", num), ("cat", onehot, CAT)], sparse_threshold=0)),
            ("clf", HistGradientBoostingClassifier(max_depth=4, learning_rate=0.05, max_iter=300, random_state=0)),
        ]),
    }


def class_balance(events: pd.DataFrame) -> None:
    print("\n== Share of events at or above each threshold ==")
    rows = []
    for target in TARGETS:
        for thr in THRESHOLDS:
            hit = events[target] >= thr
            by_year = hit.groupby(events["earnings_date"].dt.year).mean()
            rows.append({"target": target, "threshold": f"{thr:.0%}", "all": hit.mean(),
                         "min_year": by_year.min(), "max_year": by_year.max(),
                         "drop": (events[target] <= -thr).mean()})
    print(pd.DataFrame(rows).round(3).to_string(index=False))
    print("\nBy year (raw return >= 3% / >= 5%):")
    year = events["earnings_date"].dt.year
    print(pd.DataFrame({"events": events.groupby(year).size(),
                        ">=3%": (events["target_raw"] >= 0.03).groupby(year).mean(),
                        ">=5%": (events["target_raw"] >= 0.05).groupby(year).mean()}).round(3).to_string())


def evaluate(df: pd.DataFrame, num: list[str]) -> pd.DataFrame:
    year = df["earnings_date"].dt.year
    out = []
    for target in TARGETS:
        for thr in THRESHOLDS:
            y = (df[target] >= thr).astype(int)
            for name in models(num):
                for test_year in TEST_YEARS:
                    train, test = year < test_year, year == test_year
                    X_train, X_test = df.loc[train].copy(), df.loc[test].copy()
                    lo, hi = X_train[num].quantile(CLIP[0]), X_train[num].quantile(CLIP[1])
                    X_train[num] = X_train[num].clip(lo, hi, axis=1)
                    X_test[num] = X_test[num].clip(lo, hi, axis=1)
                    model = models(num)[name].fit(X_train, y[train])
                    prob = model.predict_proba(X_test)[:, 1]
                    out.append({"target": target, "threshold": thr, "model": name, "test_year": test_year,
                                "auc": roc_auc_score(y[test], prob), "pr_auc": average_precision_score(y[test], prob),
                                "base_rate": y[test].mean()})
    return pd.DataFrame(out)


if __name__ == "__main__":
    data = pd.read_parquet(OUT)
    data["earnings_date"] = pd.to_datetime(data["earnings_date"])
    data = data[~data["upcoming"]].replace([np.inf, -np.inf], np.nan)
    num = [c for c in data.columns if c not in NOT_FEATURES]
    class_balance(data[data["lead"] == 1])

    results = []
    for lead in sorted(data["lead"].unique()):
        r = evaluate(data[data["lead"] == lead].reset_index(drop=True), num)
        r["lead"] = lead
        results.append(r)
    results = pd.concat(results)
    results.to_csv(OUT.parent / "target_and_lead_time_results.csv", index=False)

    print(f"\n== Signal by feature lead time (mean over test years {TEST_YEARS[0]}-{TEST_YEARS[-1]}) ==")
    summary = results.groupby(["target", "threshold", "model", "lead"]).agg(
        auc=("auc", "mean"), auc_sd=("auc", "std"), pr_auc=("pr_auc", "mean"), base_rate=("base_rate", "mean"))
    summary["lift"] = summary["pr_auc"] / summary["base_rate"]
    print(summary.round(3).to_string())

    print("\n== AUC by lead and test year (raw, 3%, logistic) ==")
    one = results[(results.target == "target_raw") & (results.threshold == 0.03) & (results.model == "logistic")]
    print(one.pivot(index="test_year", columns="lead", values="auc").round(3).to_string())
