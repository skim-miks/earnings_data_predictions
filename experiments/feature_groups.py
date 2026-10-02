"""Which extra feature groups improve the move-size model, and how far can it be cut down?

Usage:
    uv run experiments/feature_groups.py

Every candidate uses only information available at the feature date. Each model
is trained on earlier years and tested on each of 2022-2026; the gain over the
current model comes with a bootstrap range from the pooled test predictions.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

sys.path.insert(0, str(Path(__file__).parent.parent))
import train_model as tm  # noqa: E402
from build_dataset import load_earnings  # noqa: E402
from collect_data import RAW  # noqa: E402

SECTOR_ETF = {
    "Technology": "XLK", "Financial Services": "XLF", "Healthcare": "XLV", "Consumer Cyclical": "XLY",
    "Consumer Defensive": "XLP", "Energy": "XLE", "Basic Materials": "XLB", "Utilities": "XLU",
    "Communication Services": "XLC", "Real Estate": "XLRE", "Industrials": "XLI",
}
PEER_WINDOW_DAYS = 45
# The 36-feature model this experiment started from (the production model is now tm.NUM + LEGACY_CAT).
LEGACY_NUM = [
    "eps_est_yield", "eps_prior_yield", "eps_ttm_yield", "est_vs_prior", "surprise_prior", "surprise_mean_4q",
    "past_move_last", "past_move_mean_8", "past_move_max_8",
    "ret_1d", "ret_1w", "ret_1m", "ret_3m", "ret_1y", "sma20_dist", "sma50_dist", "sma200_dist",
    "rsi_14", "dist_52w_high", "dist_52w_low", "vol_1w", "vol_1m", "high_low_range", "log_dollar_volume",
    "volume_vs_1m", "spy_ret_1w", "spy_ret_1m", "spy_ret_3m", "ret_1m_vs_spy", "ret_3m_vs_spy", "vix", "vix_chg_1w",
]
LEGACY_CAT = ["quarter", "Sector", "Industry", "report_timing"]


def reaction_history(df: pd.DataFrame) -> list[str]:
    move = df["target_raw"].abs()
    g = move.groupby(df["ticker"])
    prior = lambda fn: g.transform(lambda s: fn(s.shift(1)))  # noqa: E731
    df["past_move_median_8"] = prior(lambda s: s.rolling(8, min_periods=2).median())
    df["past_move_mean_4"] = prior(lambda s: s.rolling(4, min_periods=2).mean())
    df["past_move_trend"] = df["past_move_mean_4"] - df["past_move_mean_8"]
    df["past_move_same_q"] = g.shift(4)
    df["past_move_mean_all"] = prior(lambda s: s.expanding(min_periods=2).mean())
    big = (move >= tm.BIG_MOVE).astype(float).groupby(df["ticker"])
    df["past_big_share_8"] = big.transform(lambda s: s.shift(1).rolling(8, min_periods=2).mean())
    return ["past_move_median_8", "past_move_mean_4", "past_move_trend", "past_move_same_q",
            "past_move_mean_all", "past_big_share_8"]


def surprise_size(df: pd.DataFrame) -> list[str]:
    e = load_earnings()
    a = e["surprise"].abs().groupby(e["ticker"])
    e["abs_surprise_last"] = a.shift(1)
    e["abs_surprise_mean_8"] = a.transform(lambda s: s.shift(1).rolling(8, min_periods=2).mean())
    e["surprise_std_8"] = e.groupby("ticker")["surprise"].transform(lambda s: s.shift(1).rolling(8, min_periods=3).std())
    cols = ["abs_surprise_last", "abs_surprise_mean_8", "surprise_std_8"]
    merged = df.merge(e[["ticker", "earnings_date", *cols]], on=["ticker", "earnings_date"], how="left")
    df[cols] = merged[cols].values
    return cols


def long_volatility(df: pd.DataFrame, prices: pd.DataFrame) -> list[str]:
    p = prices.sort_values(["ticker", "date"])
    ret = p.groupby("ticker")["Adj Close"].pct_change()
    vol = lambda n: ret.groupby(p["ticker"]).transform(lambda s: s.rolling(n).std())  # noqa: E731
    p = p.assign(vol_3m=vol(63), vol_1y=vol(252), _vol_1m=vol(21))
    p["vol_1m_vs_1y"] = p["_vol_1m"] / p["vol_1y"]
    p["vol_1m_vs_3m"] = p["_vol_1m"] / p["vol_3m"]
    cols = ["vol_3m", "vol_1y", "vol_1m_vs_1y", "vol_1m_vs_3m"]
    merged = df.merge(p[["ticker", "date", *cols]], left_on=["ticker", "feature_date"], right_on=["ticker", "date"], how="left")
    df[cols] = merged[cols].values
    return cols


def sector_context(df: pd.DataFrame, prices: pd.DataFrame) -> list[str]:
    etf = prices[prices["ticker"].isin(SECTOR_ETF.values())].sort_values(["ticker", "date"])
    g = etf.groupby("ticker")["Adj Close"]
    etf = etf.assign(sector_ret_1m=g.pct_change(21), sector_ret_3m=g.pct_change(63),
                     sector_vol_1m=g.pct_change().groupby(etf["ticker"]).transform(lambda s: s.rolling(21).std()))
    cols = ["sector_ret_1m", "sector_ret_3m", "sector_vol_1m"]
    key = df.assign(etf=df["Sector"].map(SECTOR_ETF))
    merged = key.merge(etf[["ticker", "date", *cols]], left_on=["etf", "feature_date"], right_on=["ticker", "date"], how="left")
    df[cols] = merged[cols].values
    df["ret_1m_vs_sector"] = df["ret_1m"] - df["sector_ret_1m"]
    return cols + ["ret_1m_vs_sector"]


def peer_reactions(df: pd.DataFrame) -> list[str]:
    """How peers that already reported in the last 45 days moved, known by this event's feature date."""
    out = {}
    for level in ("Industry", "Sector"):
        mean = np.full(len(df), np.nan)
        ratio = np.full(len(df), np.nan)
        count = np.zeros(len(df))
        for _, grp in df.groupby(level):
            done = grp["reaction_date"].values
            move = grp["target_raw"].abs().values
            usual = grp["past_move_mean_8"].values
            tick = grp["ticker"].values
            for pos, (idx, row) in enumerate(grp.iterrows()):
                seen = (done <= row["feature_date"]) & (tick != row["ticker"]) & \
                       (done >= row["feature_date"] - np.timedelta64(PEER_WINDOW_DAYS, "D"))
                if seen.any():
                    count[df.index.get_loc(idx)] = seen.sum()
                    mean[df.index.get_loc(idx)] = move[seen].mean()
                    rel = move[seen] / usual[seen]
                    if np.isfinite(rel).any():
                        ratio[df.index.get_loc(idx)] = np.nanmedian(rel[np.isfinite(rel)])
        tag = level.lower()
        out[f"peer_{tag}_move"] = mean
        out[f"peer_{tag}_move_vs_usual"] = ratio
        out[f"peer_{tag}_count"] = count
    for k, v in out.items():
        df[k] = v
    return list(out)


def make_model(num: list[str], cat: list[str]) -> Pipeline:
    numeric = Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())])
    parts = [("num", numeric, num)]
    if cat:
        parts.append(("cat", OneHotEncoder(handle_unknown="ignore", min_frequency=30), cat))
    return Pipeline([("prep", ColumnTransformer(parts)), ("clf", LogisticRegression(max_iter=2000, C=0.1))])


def out_of_time(df: pd.DataFrame, num: list[str], cat: list[str]) -> pd.Series:
    year = df["earnings_date"].dt.year
    y = (df["target_raw"].abs() >= tm.BIG_MOVE).astype(int)
    prob = pd.Series(np.nan, index=df.index)
    for test_year in tm.TEST_YEARS:
        train, test = df[year < test_year], df[year == test_year]
        clip = tm.Clipper(num).fit(train)
        model = make_model(num, cat).fit(clip.transform(train), y[train.index])
        prob[test.index] = model.predict_proba(clip.transform(test))[:, 1]
    return prob


def compare(df: pd.DataFrame, specs: dict[str, tuple[list[str], list[str]]], reference: str) -> None:
    year = df["earnings_date"].dt.year
    y = (df["target_raw"].abs() >= tm.BIG_MOVE).astype(int)
    probs = {name: out_of_time(df, num, cat) for name, (num, cat) in specs.items()}
    scored = probs[reference].notna().values
    yv = y.values[scored]
    rng = np.random.default_rng(0)
    draws = [rng.integers(0, scored.sum(), scored.sum()) for _ in range(500)]
    ref = probs[reference].values[scored]
    rows = []
    for name, prob in probs.items():
        p = prob.values[scored]
        by_year = [roc_auc_score(y[year == t], prob[year == t]) for t in tm.TEST_YEARS]
        gains = [roc_auc_score(yv[i], p[i]) - roc_auc_score(yv[i], ref[i]) for i in draws]
        ref_year = [roc_auc_score(y[year == t], probs[reference][year == t]) for t in tm.TEST_YEARS]
        rows.append({"model": name, "features": len(specs[name][0]) + len(specs[name][1]),
                     "auc": np.mean(by_year), "gain": np.mean(by_year) - np.mean(ref_year),
                     "gain_low": np.percentile(gains, 2.5), "gain_high": np.percentile(gains, 97.5),
                     "years_better": sum(a > b for a, b in zip(by_year, ref_year))})
    print(pd.DataFrame(rows).round(4).to_string(index=False))


if __name__ == "__main__":
    df = pd.read_parquet(tm.EVENTS)
    df = df[(df["lead"] == 1) & ~df["upcoming"]].replace([np.inf, -np.inf], np.nan)
    df["earnings_date"] = pd.to_datetime(df["earnings_date"])
    df = df.sort_values(["ticker", "earnings_date"]).reset_index(drop=True)
    prices = pd.read_parquet(RAW / "prices.parquet")
    prices["date"] = pd.to_datetime(prices["date"])

    base_num = LEGACY_NUM
    groups = {
        "reaction history": reaction_history(df),
        "surprise size": surprise_size(df),
        "longer volatility": long_volatility(df, prices),
        "sector ETF context": sector_context(df, prices),
        "peer reactions": peer_reactions(df),
    }
    df = df.replace([np.inf, -np.inf], np.nan)
    everything = [c for cols in groups.values() for c in cols]

    print("== Step 1: add each group to the current 36-feature model ==")
    specs = {"current model": (base_num, LEGACY_CAT)}
    for name, cols in groups.items():
        specs[f"+ {name}"] = (base_num + cols, LEGACY_CAT)
    specs["+ all groups"] = (base_num + everything, LEGACY_CAT)
    compare(df, specs, "current model")

    print("\n== Step 2: cut-down models, compared with the current model ==")
    core = ["past_move_mean_8", "past_move_last", "past_move_max_8"]
    specs = {
        "current model": (base_num, LEGACY_CAT),
        "past moves only": (core, []),
        "past moves + sector": (core, ["Sector"]),
        "past moves + sector + vol_1m": (core + ["vol_1m"], ["Sector"]),
        "  + industry, timing": (core + ["vol_1m"], ["Sector", "Industry", "report_timing"]),
        "  + reaction history": (core + ["vol_1m"] + groups["reaction history"], ["Sector"]),
        "  + history, long vol, peers": (core + ["vol_1m"] + groups["reaction history"] + groups["longer volatility"]
                                         + groups["peer reactions"], ["Sector"]),
    }
    compare(df, specs, "current model")
