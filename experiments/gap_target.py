"""Close-to-next-open (the overnight gap) as the target instead of close-to-close.

Usage:
    uv run experiments/gap_target.py

The gap is the reaction-session open over the last close before the report,
adjusted for dividends. Compares base rates, the move-size model trained and
tested on each target, whether history features should be rebuilt from past
gaps, and whether direction is any more predictable on the gap.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))
import direction as dr  # noqa: E402
import train_model as tm  # noqa: E402
from build_dataset import BIG_MOVE, PAST_MOVE_FEATURES, PEER_FEATURES, add_past_moves, add_peer_reactions  # noqa: E402
from collect_data import RAW  # noqa: E402


def add_gap(df: pd.DataFrame) -> pd.DataFrame:
    p = pd.read_parquet(RAW / "prices.parquet")
    p["date"] = pd.to_datetime(p["date"])
    p["adj_open"] = p["Open"] * p["Adj Close"] / p["Close"]
    df = df.merge(p[["ticker", "date", "adj_open"]].rename(columns={"date": "reaction_date"}),
                  on=["ticker", "reaction_date"], how="left")
    df = df.merge(p[["ticker", "date", "Adj Close"]].rename(columns={"date": "feature_date", "Adj Close": "base_close"}),
                  on=["ticker", "feature_date"], how="left")
    df["target_gap"] = df["adj_open"] / df["base_close"] - 1
    df["target_session"] = (1 + df["target_raw"]) / (1 + df["target_gap"]) - 1   # open to close of the reaction day
    return df


def out_of_time(df: pd.DataFrame, y: pd.Series, num: list[str]) -> pd.Series:
    year = df["earnings_date"].dt.year
    prob = pd.Series(np.nan, index=df.index)
    for test_year in tm.TEST_YEARS:
        train, test = df[year < test_year], df[year == test_year]
        clip = tm.Clipper(num).fit(train)
        model = tm.make_model(num).fit(clip.transform(train), y[train.index])
        prob[test.index] = model.predict_proba(clip.transform(test))[:, 1]
    return prob


def auc_by_year(df: pd.DataFrame, y: pd.Series, prob: pd.Series) -> tuple[float, str]:
    year = df["earnings_date"].dt.year
    by_year = [roc_auc_score(y[year == t], prob[year == t]) for t in tm.TEST_YEARS]
    return float(np.mean(by_year)), " ".join(f"{a:.3f}" for a in by_year)


if __name__ == "__main__":
    df = pd.read_parquet(tm.EVENTS)
    df = df[(df["lead"] == 1) & ~df["upcoming"]].replace([np.inf, -np.inf], np.nan)
    for col in ("earnings_date", "reaction_date", "feature_date"):
        df[col] = pd.to_datetime(df[col])
    df = add_gap(df).dropna(subset=["target_gap"]).sort_values(["ticker", "earnings_date"]).reset_index(drop=True)
    c2c, gap, session = df["target_raw"], df["target_gap"], df["target_session"]
    print(f"{len(df):,} reports")

    print("\n== How big is each piece of the reaction? ==")
    rows = []
    for name, r in [("close to close (current target)", c2c), ("close to next open (gap)", gap), ("open to close that day", session)]:
        rows.append({"move": name, "median size": r.abs().median(), "3%+ either way": (r.abs() >= 0.03).mean(),
                     "5%+ either way": (r.abs() >= BIG_MOVE).mean(), "share up": (r > 0).mean()})
    print(pd.DataFrame(rows).round(3).to_string(index=False))
    print(f"correlation gap vs close-to-close: {gap.corr(c2c):.2f}; gap vs rest of day: {gap.corr(session):.2f}")
    same_side = (np.sign(gap) == np.sign(c2c))[gap.abs() >= 0.02].mean()
    print(f"when the gap is 2%+ the day closes on the same side of the prior close {same_side:.0%} of the time")
    big_gap, big_c2c = gap.abs() >= BIG_MOVE, c2c.abs() >= BIG_MOVE
    print(f"5%+ gaps that are still 5%+ at the close: {big_c2c[big_gap].mean():.0%}; "
          f"5%+ closes that had gapped 5%+: {big_gap[big_c2c].mean():.0%}")

    # History and peer features rebuilt from past gaps instead of past close-to-close moves.
    g = df.drop(columns=PAST_MOVE_FEATURES + PEER_FEATURES).assign(target_raw=df["target_gap"])
    g = add_peer_reactions(add_past_moves(g)).replace([np.inf, -np.inf], np.nan)
    g = g.sort_values(["ticker", "earnings_date"]).reset_index(drop=True)
    assert (g["ticker"].values == df["ticker"].values).all() and (g["earnings_date"].values == df["earnings_date"].values).all()

    print("\n== Move-size model: 5%+ in either direction, train on earlier years, test 2022-2026 ==")
    y_c2c, y_gap = big_c2c.astype(int), big_gap.astype(int)
    runs = {
        "close-to-close target, current features": (df, y_c2c),
        "gap target, current features (history from close-to-close)": (df, y_gap),
        "gap target, history rebuilt from past gaps": (g, y_gap),
    }
    probs, rows = {}, []
    for name, (data, y) in runs.items():
        probs[name] = out_of_time(data, y, tm.NUM)
        auc, by_year = auc_by_year(data, y, probs[name])
        tested = probs[name].notna()
        rows.append({"model": name, "auc": auc, "by year": by_year, "base rate": y[tested].mean()})
    print(pd.DataFrame(rows).round(3).to_string(index=False))
    cross, _ = auc_by_year(df, y_gap, probs["close-to-close target, current features"])
    print(f"the current live model (trained on close-to-close) scored against 5%+ gaps: AUC {cross:.3f}")

    best = "gap target, history rebuilt from past gaps"
    scored = probs[best].notna()
    t = pd.DataFrame({"predicted": probs[best][scored], "big_gap": big_gap[scored], "gap_size": gap[scored].abs()})
    t["decile"] = pd.qcut(t["predicted"], 10, labels=False) + 1
    print("\nGap model, predicted vs actual by score decile:")
    print(t.groupby("decile").agg(predicted=("predicted", "mean"), actual=("big_gap", "mean"),
                                  median_gap=("gap_size", "median")).round(3).iloc[[0, 2, 4, 6, 8, 9]].to_string())

    print("\n== Direction: is up vs down any more predictable on the gap? ==")
    groups = dr.add_features(df)
    df = df.replace([np.inf, -np.inf], np.nan)
    everything = list(dict.fromkeys(c for cols in groups.values() for c in cols))
    year = df["earnings_date"].dt.year
    tested = year.isin(tm.TEST_YEARS)
    for name, r in [("close to close", c2c), ("close to next open (gap)", gap), ("open to close that day", session)]:
        y = (r > 0).astype(int)
        line = []
        for kind in ("logistic", "boosting"):
            prob = dr.out_of_time(df, y, everything, kind)
            lo, hi = dr.auc_range(y.values[tested], prob.values[tested], n=300)
            line.append(f"{kind} {roc_auc_score(y[tested], prob[tested]):.3f} ({lo:.3f}-{hi:.3f})")
        print(f"{name:28} {' | '.join(line)}")
