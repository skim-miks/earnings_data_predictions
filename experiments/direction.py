"""Can anything predict the direction of the earnings reaction?

Usage:
    uv run experiments/direction.py

Tests feature groups aimed at direction (beat/miss habits, the stock's own past
reactions, peer results this season, pre-report run-up, estimate context,
market backdrop). Every feature uses only information available at the feature
date. Models are trained on earlier years and tested on each of 2022-2026.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).parent.parent))
import train_model as tm  # noqa: E402
from build_dataset import BIG_MOVE, load_earnings, peer_groups  # noqa: E402


def prior_mean(values: pd.Series, mask: pd.Series, ticker: pd.Series) -> pd.Series:
    """Mean of `values` over this ticker's earlier events where `mask` is true."""
    total = (values.where(mask, 0)).groupby(ticker).transform(lambda s: s.shift(1).cumsum())
    count = mask.astype(float).groupby(ticker).transform(lambda s: s.shift(1).cumsum())
    return total / count.where(count > 0)


def add_features(df: pd.DataFrame) -> dict[str, list[str]]:
    t, ret, surprise = df["ticker"], df["target_raw"], df["surprise_actual"]
    beat = surprise > 0
    roll = lambda s, fn: s.groupby(t).transform(lambda x: fn(x.shift(1).rolling(8, min_periods=2)))  # noqa: E731

    df["surprise_mean_8"] = roll(surprise, lambda r: r.mean())
    df["beat_rate_8"] = roll(beat.astype(float), lambda r: r.mean())

    df["past_ret_last"] = ret.groupby(t).shift(1)
    df["past_ret_mean_8"] = roll(ret, lambda r: r.mean())
    df["past_up_share_8"] = roll((ret > 0).astype(float), lambda r: r.mean())
    df["past_ret_on_beat"] = prior_mean(ret, beat, t)
    df["past_ret_on_miss"] = prior_mean(ret, ~beat, t)
    df["past_ret_expected"] = (df["beat_rate_8"] * df["past_ret_on_beat"]
                               + (1 - df["beat_rate_8"]) * df["past_ret_on_miss"])

    peer_cols = []
    for level in ("Industry", "Sector"):
        tag = level.lower()
        cols = {f"peer_{tag}_ret": np.full(len(df), np.nan), f"peer_{tag}_up_share": np.full(len(df), np.nan),
                f"peer_{tag}_beat_share": np.full(len(df), np.nan), f"peer_{tag}_surprise": np.full(len(df), np.nan)}
        r, s = ret.values, surprise.values
        for i, peers in peer_groups(df, level):
            cols[f"peer_{tag}_ret"][i] = r[peers].mean()
            cols[f"peer_{tag}_up_share"][i] = (r[peers] > 0).mean()
            cols[f"peer_{tag}_beat_share"][i] = (s[peers] > 0).mean()
            cols[f"peer_{tag}_surprise"][i] = np.nanmean(s[peers])
        for k, v in cols.items():
            df[k] = v
        peer_cols += list(cols)

    e = load_earnings()
    year_ago = e.groupby("ticker")["eps_reported"].shift(4)
    e["est_growth_yoy"] = ((e["eps_est"] - year_ago) / year_ago.abs().clip(lower=0.1)).clip(-5, 5)
    df["est_growth_yoy"] = df.merge(e[["ticker", "earnings_date", "est_growth_yoy"]],
                                    on=["ticker", "earnings_date"], how="left")["est_growth_yoy"].values

    return {
        "beat/miss habit": ["surprise_prior", "surprise_mean_4q", "surprise_mean_8", "beat_rate_8"],
        "own past reactions": ["past_ret_last", "past_ret_mean_8", "past_up_share_8", "past_ret_on_beat",
                               "past_ret_on_miss", "past_ret_expected"],
        "peer results this season": peer_cols,
        "pre-report run-up": ["ret_1w", "ret_1m", "ret_3m", "ret_1m_vs_spy", "ret_3m_vs_spy", "dist_52w_high",
                              "rsi_14", "sma50_dist"],
        "estimate context": ["est_vs_prior", "eps_est_yield", "eps_ttm_yield", "est_growth_yoy"],
        "market backdrop": ["spy_ret_1w", "spy_ret_1m", "vix", "vix_chg_1w"],
    }


def make_model(kind: str, num: list[str]):
    if kind == "logistic":
        numeric = Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())])
        return Pipeline([("prep", ColumnTransformer([("num", numeric, num)])),
                         ("clf", LogisticRegression(max_iter=2000, C=0.1))])
    return Pipeline([("prep", ColumnTransformer([("num", "passthrough", num)])),
                     ("clf", HistGradientBoostingClassifier(max_depth=3, learning_rate=0.03, max_iter=300,
                                                            min_samples_leaf=100, random_state=0))])


def out_of_time(df: pd.DataFrame, y: pd.Series, num: list[str], kind: str) -> pd.Series:
    year = df["earnings_date"].dt.year
    prob = pd.Series(np.nan, index=df.index)
    for test_year in tm.TEST_YEARS:
        train, test = df[year < test_year], df[year == test_year]
        clip = tm.Clipper(num).fit(train)
        model = make_model(kind, num).fit(clip.transform(train), y[train.index])
        prob[test.index] = model.predict_proba(clip.transform(test))[:, 1]
    return prob


def auc_range(y: np.ndarray, p: np.ndarray, n: int = 500) -> tuple[float, float]:
    rng = np.random.default_rng(0)
    draws = [roc_auc_score(y[i], p[i]) for i in (rng.integers(0, len(y), len(y)) for _ in range(n))]
    return tuple(np.percentile(draws, [2.5, 97.5]))


if __name__ == "__main__":
    df = pd.read_parquet(tm.EVENTS)
    df = df[(df["lead"] == 1) & ~df["upcoming"]].replace([np.inf, -np.inf], np.nan)
    df["earnings_date"] = pd.to_datetime(df["earnings_date"])
    df = df.sort_values(["ticker", "earnings_date"]).reset_index(drop=True)
    groups = add_features(df)
    df = df.replace([np.inf, -np.inf], np.nan)
    everything = list(dict.fromkeys(c for cols in groups.values() for c in cols))
    year = df["earnings_date"].dt.year
    tested = year.isin(tm.TEST_YEARS).values
    up = (df["target_raw"] > 0).astype(int)
    print(f"{tested.sum():,} test events, {up[tested].mean():.1%} went up")

    print("\n== Up vs down: AUC by feature group (0.50 = coin flip) ==")
    rows, probs = [], {}
    for name, cols in {**groups, "all groups": everything}.items():
        for kind in ("logistic", "boosting"):
            prob = out_of_time(df, up, cols, kind)
            probs[(name, kind)] = prob
            lo, hi = auc_range(up.values[tested], prob.values[tested])
            by_year = [roc_auc_score(up[year == t], prob[year == t]) for t in tm.TEST_YEARS]
            rows.append({"features": name, "model": kind, "auc": roc_auc_score(up[tested], prob[tested]),
                         "low": lo, "high": hi, "years_above_0.5": sum(a > 0.5 for a in by_year),
                         "by_year": " ".join(f"{a:.3f}" for a in by_year)})
    print(pd.DataFrame(rows).round(3).to_string(index=False))

    best = max(rows, key=lambda r: r["auc"])
    prob = probs[(best["features"], best["model"])]
    print(f"\n== Best: {best['features']} / {best['model']} ==")
    t = df[tested].assign(score=prob[tested])
    t["decile"] = pd.qcut(t["score"], 10, labels=False) + 1
    print(t.groupby("decile").agg(events=("target_raw", "size"), share_up=("target_raw", lambda s: (s > 0).mean()),
                                   mean_return=("target_raw", "mean"), mean_vs_spy=("target_vs_spy", "mean"),
                                   median_return=("target_raw", "median")).round(4).to_string())
    top, bottom = t[t["decile"] == 10], t[t["decile"] == 1]
    spread = top.groupby(top["earnings_date"].dt.year)["target_raw"].mean() - \
        bottom.groupby(bottom["earnings_date"].dt.year)["target_raw"].mean()
    print("\nTop minus bottom decile, mean return by year:", spread.round(4).to_dict())
    big = t["target_raw"].abs() >= BIG_MOVE
    print(f"Among {big.sum():,} moves of 5%+: AUC up vs down {roc_auc_score((t['target_raw'] > 0)[big], t['score'][big]):.3f}")
    vs_spy = (t["target_vs_spy"] > 0).astype(int)
    print(f"Same scores against beating SPY that day: AUC {roc_auc_score(vs_spy, t['score']):.3f}")
