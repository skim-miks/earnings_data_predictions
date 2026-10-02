"""Build the modeling table: one row per earnings event and feature lead time.

Usage:
    uv run build_dataset.py

Reads data/raw/*.parquet and writes data/processed/events.parquet.

Upcoming reports are included with `upcoming` = True, no target, and features
from the latest available close, so a trained model can score them.

Target: the close-to-close return of the first session that can react to the
report. Before-open reports react the same day; after-close reports react the
next trading day. Reports during the session are dropped (under 1% of events).

Lead time: `lead` = 1 uses features from the last close before the report,
2 and 3 use the closes one and two trading days earlier.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from collect_data import EARNINGS_DIR, EVENT_START, RAW, report_timing

OUT = RAW.parent / "processed" / "events.parquet"
LEADS = [1, 2, 3]
SURPRISE_CLIP = 200   # percent; Yahoo's surprise explodes when the estimate is near zero
BIG_MOVE = 0.05       # a "big" reaction: 5% or more in either direction
PEER_WINDOW_DAYS = 45
MAX_GAP_DAYS = 5      # calendar days allowed between the report date and the reaction session


def rsi(close: pd.Series, window: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(window).mean()
    loss = (-delta.clip(upper=0)).rolling(window).mean()
    return 100 - 100 / (1 + gain / loss)


def daily_features(p: pd.DataFrame, market: pd.DataFrame) -> pd.DataFrame:
    """Scale-free price features for every trading day of one ticker."""
    adj, close = p["Adj Close"], p["Close"]
    ret = adj.pct_change()
    f = pd.DataFrame(index=p.index)
    f["date"] = p["date"]
    f["close"] = close
    for name, n in [("ret_1d", 1), ("ret_1w", 5), ("ret_1m", 21), ("ret_3m", 63), ("ret_1y", 252)]:
        f[name] = adj.pct_change(n)
    for n in (20, 50, 200):
        f[f"sma{n}_dist"] = adj / adj.rolling(n).mean() - 1
    f["rsi_14"] = rsi(adj)
    f["vol_1w"] = ret.rolling(5).std()
    f["vol_1m"] = ret.rolling(21).std()
    f["vol_3m"] = ret.rolling(63).std()
    f["vol_1y"] = ret.rolling(252).std()
    f["vol_1m_vs_3m"] = f["vol_1m"] / f["vol_3m"]
    f["vol_1m_vs_1y"] = f["vol_1m"] / f["vol_1y"]
    f["dist_52w_high"] = close / p["High"].rolling(252).max() - 1
    f["dist_52w_low"] = close / p["Low"].rolling(252).min() - 1
    f["high_low_range"] = (p["High"] - p["Low"]) / close
    f["log_dollar_volume"] = np.log1p(close * p["Volume"])
    f["volume_vs_1m"] = p["Volume"] / p["Volume"].rolling(21).mean()
    f = f.merge(market, on="date", how="left")
    f["ret_1m_vs_spy"] = f["ret_1m"] - f["spy_ret_1m"]
    f["ret_3m_vs_spy"] = f["ret_3m"] - f["spy_ret_3m"]
    return f


def market_features(prices: pd.DataFrame) -> pd.DataFrame:
    spy = prices[prices["ticker"] == "SPY"].set_index("date")["Adj Close"]
    vix = prices[prices["ticker"] == "^VIX"].set_index("date")["Close"]
    m = pd.DataFrame({"spy_adj": spy})
    m["spy_ret_1w"] = spy.pct_change(5)
    m["spy_ret_1m"] = spy.pct_change(21)
    m["spy_ret_3m"] = spy.pct_change(63)
    m["vix"] = vix
    m["vix_chg_1w"] = vix.pct_change(5)
    return m.reset_index()


def load_earnings() -> pd.DataFrame:
    """Full earnings history per ticker, so lagged EPS exists for the first 2018 events."""
    e = pd.concat([pd.read_parquet(p) for p in sorted(EARNINGS_DIR.glob("*.parquet"))], ignore_index=True)
    e["earnings_ts"] = pd.to_datetime(e["earnings_ts"], utc=True).dt.tz_convert("America/New_York")
    e["earnings_date"] = e["earnings_ts"].dt.tz_localize(None).dt.normalize()
    e["report_timing"] = report_timing(e["earnings_ts"])
    today = pd.Timestamp.now(tz="America/New_York").tz_localize(None).normalize()
    e["upcoming"] = e["earnings_date"] >= today
    # Yahoo's upcoming times are placeholders that land at 15:00 ET after the clock change.
    e.loc[e["upcoming"] & (e["earnings_ts"].dt.hour >= 15), "report_timing"] = "amc"
    e = e[e["Reported EPS"].notna() | e["upcoming"]]
    e = e.sort_values(["ticker", "earnings_ts"]).drop_duplicates(["ticker", "earnings_date"], keep="last")
    e = e.rename(columns={"EPS Estimate": "eps_est", "Reported EPS": "eps_reported", "Surprise(%)": "surprise"})
    e["surprise"] = e["surprise"].clip(-SURPRISE_CLIP, SURPRISE_CLIP)

    # Everything below uses only earlier reports of the same ticker.
    g = e.groupby("ticker")
    e["eps_prior"] = g["eps_reported"].shift(1)
    e["eps_ttm"] = g["eps_reported"].transform(lambda s: s.shift(1).rolling(4).sum())
    e["surprise_prior"] = g["surprise"].shift(1)
    e["surprise_mean_4q"] = g["surprise"].transform(lambda s: s.shift(1).rolling(4).mean())
    return e


def event_row(ticker, ev, feat, lead, reaction_date, target_raw, target_vs_spy) -> dict:
    close = feat["close"]
    return {
        "ticker": ticker, "earnings_date": ev.earnings_date, "report_timing": ev.report_timing,
        "upcoming": ev.upcoming, "reaction_date": reaction_date, "lead": lead, "feature_date": feat["date"],
        "target_raw": target_raw, "target_vs_spy": target_vs_spy,
        "surprise_actual": ev.surprise,   # this report's result: an outcome, never a feature
        "eps_est_yield": ev.eps_est / close, "eps_prior_yield": ev.eps_prior / close,
        "eps_ttm_yield": ev.eps_ttm / close, "est_vs_prior": (ev.eps_est - ev.eps_prior) / close,
        "surprise_prior": ev.surprise_prior, "surprise_mean_4q": ev.surprise_mean_4q,
        **feat.drop(["date", "close", "spy_adj"]).to_dict(),
    }


PAST_MOVE_FEATURES = ["past_move_last", "past_move_mean_8", "past_move_max_8", "past_move_median_8",
                      "past_move_mean_4", "past_move_trend", "past_move_same_q", "past_move_mean_all",
                      "past_big_share_8"]
PEER_FEATURES = [f"peer_{level}_{stat}" for level in ("industry", "sector")
                 for stat in ("move", "move_vs_usual", "count")]


def add_past_moves(df: pd.DataFrame) -> pd.DataFrame:
    """How much this stock moved on its own earlier reports (earlier events only)."""
    one = df[df["lead"] == 1].sort_values(["ticker", "earnings_date"])
    move = one["target_raw"].abs()
    g = move.groupby(one["ticker"])
    prior = lambda fn: g.transform(lambda s: fn(s.shift(1)))  # noqa: E731
    big = (move >= BIG_MOVE).where(move.notna()).astype(float).groupby(one["ticker"])
    one = one.assign(
        past_move_last=g.shift(1),
        past_move_mean_8=prior(lambda s: s.rolling(8, min_periods=2).mean()),
        past_move_max_8=prior(lambda s: s.rolling(8, min_periods=2).max()),
        past_move_median_8=prior(lambda s: s.rolling(8, min_periods=2).median()),
        past_move_mean_4=prior(lambda s: s.rolling(4, min_periods=2).mean()),
        past_move_same_q=g.shift(4),   # same quarter a year earlier
        past_move_mean_all=prior(lambda s: s.expanding(min_periods=2).mean()),
        past_big_share_8=big.transform(lambda s: s.shift(1).rolling(8, min_periods=2).mean()),
    )
    one["past_move_trend"] = one["past_move_mean_4"] - one["past_move_mean_8"]
    keys = ["ticker", "earnings_date"]
    return df.merge(one[keys + PAST_MOVE_FEATURES], on=keys, how="left")


def peer_groups(one: pd.DataFrame, level: str):
    """For each event, the positions of peers whose reaction finished within PEER_WINDOW_DAYS before its feature date.

    `one` needs a 0..n-1 index. Upcoming reports have no reaction date, so they are never a peer.
    """
    window = np.timedelta64(PEER_WINDOW_DAYS, "D")
    for _, grp in one.groupby(level):
        pos, done = grp.index.values, grp["reaction_date"].values
        fdate, tick = grp["feature_date"].values, grp["ticker"].values
        for j in range(len(grp)):
            seen = (done <= fdate[j]) & (done >= fdate[j] - window) & (tick != tick[j])
            if seen.any():
                yield pos[j], pos[seen]


def add_peer_reactions(df: pd.DataFrame) -> pd.DataFrame:
    """How same-industry and same-sector companies that already reported this season moved."""
    one = df[df["lead"] == 1].reset_index(drop=True)
    move = one["target_raw"].abs().values
    vs_usual = move / one["past_move_mean_8"].values
    for level in ("Industry", "Sector"):
        tag = level.lower()
        mean, ratio, count = np.full(len(one), np.nan), np.full(len(one), np.nan), np.zeros(len(one))
        for i, peers in peer_groups(one, level):
            count[i] = len(peers)
            mean[i] = move[peers].mean()
            rel = vs_usual[peers]
            if np.isfinite(rel).any():
                ratio[i] = np.median(rel[np.isfinite(rel)])
        one[f"peer_{tag}_move"], one[f"peer_{tag}_move_vs_usual"], one[f"peer_{tag}_count"] = mean, ratio, count
    keys = ["ticker", "earnings_date"]
    return df.merge(one[keys + PEER_FEATURES], on=keys, how="left")


def build() -> pd.DataFrame:
    prices = pd.read_parquet(RAW / "prices.parquet")
    prices["date"] = pd.to_datetime(prices["date"])
    market = market_features(prices)
    spy_adj = market.set_index("date")["spy_adj"]
    earnings = load_earnings()
    earnings = earnings[(earnings["earnings_date"] >= EVENT_START) & (earnings["report_timing"] != "during")]
    universe = pd.read_parquet(RAW / "universe.parquet")

    rows = []
    for ticker, events in earnings.groupby("ticker"):
        p = prices[prices["ticker"] == ticker].reset_index(drop=True)
        if p.empty:
            continue
        f = daily_features(p, market)
        dates, adj = p["date"].values, p["Adj Close"].values
        for ev in events.itertuples():
            day = np.datetime64(ev.earnings_date)
            if ev.upcoming:
                feat = f.iloc[-1]   # latest close; refreshed each time prices are re-pulled
                rows.append(event_row(ticker, ev, feat, lead=1, reaction_date=pd.NaT,
                                      target_raw=np.nan, target_vs_spy=np.nan))
                continue
            if ev.report_timing == "bmo":
                react = np.searchsorted(dates, day, side="left")    # first session on or after the report
                base = react - 1
            else:
                base = np.searchsorted(dates, day, side="right") - 1  # last close before the report
                react = base + 1
            if base < max(LEADS) or react >= len(dates):
                continue
            if (dates[react] - day) / np.timedelta64(1, "D") > MAX_GAP_DAYS:
                continue
            target_raw = adj[react] / adj[base] - 1
            spy_ret = spy_adj.get(pd.Timestamp(dates[react]), np.nan) / spy_adj.get(pd.Timestamp(dates[base]), np.nan) - 1
            for lead in LEADS:
                rows.append(event_row(ticker, ev, f.iloc[base - (lead - 1)], lead, dates[react],
                                      target_raw, target_raw - spy_ret))

    df = add_past_moves(pd.DataFrame(rows))
    df = df.merge(universe[["ticker", "Sector", "Industry", "HeadquartersState"]], on="ticker", how="left")
    df = add_peer_reactions(df)
    df["quarter"] = pd.to_datetime(df["earnings_date"]).dt.quarter
    return df.sort_values(["earnings_date", "ticker", "lead"]).reset_index(drop=True)


if __name__ == "__main__":
    df = build()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT, index=False)
    events = df[(df["lead"] == 1) & ~df["upcoming"]]
    print(f"{int(df['upcoming'].sum())} upcoming reports to score")
    print(f"{len(events):,} events, {events['ticker'].nunique()} tickers, "
          f"{events['earnings_date'].min():%Y-%m-%d} to {events['earnings_date'].max():%Y-%m-%d} -> {OUT}")
    print(f"timing: {events['report_timing'].value_counts().to_dict()}")
