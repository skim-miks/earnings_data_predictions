"""Move-size model: chance that a stock moves 5% or more, in either direction, on its earnings reaction.

Usage:
    uv run build_dataset.py
    uv run train_model.py

Prints an out-of-time evaluation (train on earlier years, test on each of
2022-2026), then fits on all history and scores the upcoming reports into
data/processed/upcoming_scores.parquet for the calendar.

The model says how big the move is likely to be. It does not predict direction.
"""

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from build_dataset import OUT as EVENTS

SCORES = EVENTS.parent / "upcoming_scores.parquet"
BIG_MOVE = 0.05
TEST_YEARS = [2022, 2023, 2024, 2025, 2026]
CAT = ["quarter", "Sector", "Industry", "report_timing"]
PAST_MOVES = ["past_move_last", "past_move_mean_8", "past_move_max_8"]
NOT_FEATURES = {"ticker", "earnings_date", "reaction_date", "lead", "feature_date", "upcoming",
                "target_raw", "target_vs_spy", "HeadquartersCity", "HeadquartersState", *CAT}
CLIP = (0.005, 0.995)   # per-feature quantile clip, fit on train


class Clipper:
    """Clip numeric features to quantiles learned on the training rows."""

    def __init__(self, cols: list[str]):
        self.cols = cols

    def fit(self, X: pd.DataFrame) -> "Clipper":
        self.lo, self.hi = X[self.cols].quantile(CLIP[0]), X[self.cols].quantile(CLIP[1])
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        X[self.cols] = X[self.cols].clip(self.lo, self.hi, axis=1)
        return X


def make_model(num: list[str]) -> Pipeline:
    numeric = Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())])
    onehot = OneHotEncoder(handle_unknown="ignore", min_frequency=30)
    return Pipeline([
        ("prep", ColumnTransformer([("num", numeric, num), ("cat", onehot, CAT)])),
        ("clf", LogisticRegression(max_iter=2000, C=0.1)),
    ])


def fit_predict(train: pd.DataFrame, test: pd.DataFrame, num: list[str]) -> np.ndarray:
    clip = Clipper(num).fit(train)
    y = (train["target_raw"].abs() >= BIG_MOVE).astype(int)
    model = make_model(num).fit(clip.transform(train), y)
    return model.predict_proba(clip.transform(test))[:, 1]


def out_of_time(hist: pd.DataFrame, num: list[str]) -> pd.Series:
    year = hist["earnings_date"].dt.year
    prob = pd.Series(np.nan, index=hist.index)
    for test_year in TEST_YEARS:
        prob[year == test_year] = fit_predict(hist[year < test_year], hist[year == test_year], num)
    return prob


def report(hist: pd.DataFrame, prob: pd.Series, label: str) -> None:
    scored = prob.notna()
    big = hist.loc[scored, "target_raw"].abs() >= BIG_MOVE
    p = prob[scored]
    by_year = [roc_auc_score(big[hist.loc[scored, "earnings_date"].dt.year == y],
                             p[hist.loc[scored, "earnings_date"].dt.year == y]) for y in TEST_YEARS]
    print(f"{label}: AUC {np.mean(by_year):.3f} (by year {', '.join(f'{a:.3f}' for a in by_year)}), "
          f"PR-AUC {average_precision_score(big, p):.3f}, base rate {big.mean():.3f}")


if __name__ == "__main__":
    df = pd.read_parquet(EVENTS)
    df = df[df["lead"] == 1].replace([np.inf, -np.inf], np.nan).reset_index(drop=True)
    df["earnings_date"] = pd.to_datetime(df["earnings_date"])
    hist, upcoming = df[~df["upcoming"]], df[df["upcoming"]]
    num = [c for c in df.columns if c not in NOT_FEATURES]

    print(f"Out-of-time test, {TEST_YEARS[0]}-{TEST_YEARS[-1]}: move of {BIG_MOVE:.0%} or more in either direction")
    report(hist, out_of_time(hist, [c for c in num if c not in PAST_MOVES]), "without past earnings moves")
    prob = out_of_time(hist, num)
    report(hist, prob, "with past earnings moves   ")

    scored = prob.notna()
    table = pd.DataFrame({"predicted": prob[scored], "big_move": hist.loc[scored, "target_raw"].abs() >= BIG_MOVE,
                          "abs_move": hist.loc[scored, "target_raw"].abs()})
    table["decile"] = pd.qcut(table["predicted"], 10, labels=False) + 1
    print("\nPredicted vs actual by score decile:")
    print(table.groupby("decile").agg(predicted=("predicted", "mean"), actual=("big_move", "mean"),
                                      median_abs_move=("abs_move", "median")).round(3).to_string())

    upcoming = upcoming.assign(p_big_move=fit_predict(hist, upcoming, num))
    upcoming[["ticker", "earnings_date", "p_big_move", "past_move_mean_8", "past_move_last", "feature_date"]].to_parquet(SCORES, index=False)
    print(f"\nScored {len(upcoming)} upcoming reports -> {SCORES}")
    print(upcoming["p_big_move"].describe().round(3).to_string())
