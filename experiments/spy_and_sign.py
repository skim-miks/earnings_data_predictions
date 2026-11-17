"""SPY's previous-day and previous-week returns as features, and up/down (no 5% line) as the target.

Usage:
    uv run experiments/spy_and_sign.py

SPY returns are measured up to the feature date (the last close before the report),
so they are known before the report. Train on earlier years, test on each of 2022-2026.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))
import direction as dr  # noqa: E402
import train_model as tm  # noqa: E402
from build_dataset import BIG_MOVE  # noqa: E402
from collect_data import RAW  # noqa: E402

SPY = ["spy_ret_1d", "spy_ret_1w"]


def add_spy(df: pd.DataFrame) -> pd.DataFrame:
    p = pd.read_parquet(RAW / "prices.parquet")
    spy = p[p["ticker"] == "SPY"].assign(date=lambda x: pd.to_datetime(x["date"])).set_index("date")["Adj Close"].sort_index()
    s = pd.DataFrame({"spy_ret_1d": spy.pct_change(1), "spy_ret_1w": spy.pct_change(5),
                      "spy_next_day": spy.shift(-1) / spy - 1})   # the market on the reaction session (diagnostic only)
    df = df.drop(columns=[c for c in ("spy_ret_1w",) if c in df.columns])
    df = df.merge(s[SPY].rename_axis("feature_date").reset_index(), on="feature_date", how="left")
    # Market move over the same window as the target, for the diagnostic below.
    react = s["spy_next_day"].rename_axis("feature_date").reset_index()
    return df.merge(react, on="feature_date", how="left")


def size_auc(df: pd.DataFrame, num: list[str], target: str) -> tuple[float, str]:
    rows = df.dropna(subset=[target])
    prob = tm.out_of_time(rows, num, target)
    year, big = rows["earnings_date"].dt.year, rows[target].abs() >= BIG_MOVE
    by_year = [roc_auc_score(big[year == t], prob[year == t]) for t in tm.TEST_YEARS]
    return float(np.mean(by_year)), " ".join(f"{a:.3f}" for a in by_year)


if __name__ == "__main__":
    df = pd.read_parquet(tm.EVENTS)
    df = df[(df["lead"] == 1) & ~df["upcoming"]].replace([np.inf, -np.inf], np.nan)
    for col in ("earnings_date", "feature_date"):
        df[col] = pd.to_datetime(df[col])
    df = add_spy(df).sort_values(["ticker", "earnings_date"]).reset_index(drop=True)
    print(f"{len(df):,} reports; SPY features filled for {df[SPY].notna().all(axis=1).mean():.1%}")

    print("\n== 1. Adding SPY's previous-day and previous-week return to the move-size models ==")
    rows = []
    for label, num, target in [("by the close", tm.NUM, "target_raw"), ("at the open", tm.GAP_NUM, "target_gap")]:
        base, base_years = size_auc(df, num, target)
        plus, plus_years = size_auc(df, num + SPY, target)
        rows.append({"model": label, "current": base, "with SPY": plus, "change": plus - base,
                     "current by year": base_years, "with SPY by year": plus_years})
    print(pd.DataFrame(rows).round(4).to_string(index=False))
    for col in SPY:
        r, p = spearmanr(df[col], df["target_raw"].abs(), nan_policy="omit")
        print(f"  {col}: rank correlation with the size of the move {r:+.3f} (p={p:.3f})")

    print("\n== 2. Up or down as the target (any move, no 5% line) ==")
    groups = dr.add_features(df)
    df = df.replace([np.inf, -np.inf], np.nan)
    direction = list(dict.fromkeys(c for cols in groups.values() for c in cols if c not in SPY))
    year = df["earnings_date"].dt.year
    tested = year.isin(tm.TEST_YEARS)
    sets = {
        "SPY previous day + week only": SPY,
        "move-size inputs": tm.NUM,
        "move-size inputs + SPY": tm.NUM + SPY,
        "direction features": direction,
        "direction features + SPY": direction + SPY,
    }
    for target_label, target in [("close to close", "target_raw"), ("close to next open", "target_gap")]:
        y = (df[target] > 0).astype(int)
        print(f"\n{target_label}: {y[tested].mean():.1%} of test reports went up")
        out = []
        for name, cols in sets.items():
            for kind in ("logistic", "boosting"):
                prob = dr.out_of_time(df, y, cols, kind)
                lo, hi = dr.auc_range(y.values[tested], prob.values[tested], n=300)
                by_year = [roc_auc_score(y[year == t], prob[year == t]) for t in tm.TEST_YEARS]
                out.append({"features": name, "model": kind, "auc": roc_auc_score(y[tested], prob[tested]),
                            "low": lo, "high": hi, "years above 0.5": sum(a > 0.5 for a in by_year)})
        print(pd.DataFrame(out).round(3).to_string(index=False))

    print("\n== For context: how much does the market on the reaction day itself explain? (not knowable in advance) ==")
    t = df[tested]
    y = (t["target_raw"] > 0).astype(int)
    print(f"AUC of SPY's own move on the reaction session for up/down: {roc_auc_score(y, t['spy_next_day']):.3f}")
    up_market = t["spy_next_day"] > 0
    print(f"stocks up on reaction day: {y[up_market].mean():.1%} when SPY rose that day, {y[~up_market].mean():.1%} when it fell")
    prev = t["spy_ret_1d"] > 0
    print(f"stocks up on reaction day: {y[prev].mean():.1%} after an up day for SPY, {y[~prev].mean():.1%} after a down day")
