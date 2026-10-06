"""Daily snapshot of analyst, estimate, short-interest, insider and options data
for companies about to report.

Usage:
    uv run collect_snapshot.py                   # companies reporting within 14 days
    uv run collect_snapshot.py --within-days 7

Yahoo only serves the current value of these numbers, so there is no history
to backtest against. Saving one file per day (data/snapshots/YYYY-MM-DD.parquet,
one row per ticker) builds that history going forward, and the latest file
feeds the context panel in the calendar.
"""

import argparse
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import yfinance as yf

from collect_data import EARNINGS_DIR, RAW, load_tickers

SNAPSHOT_DIR = RAW.parent / "snapshots"
TZ = "America/New_York"
OPTIONS_WITHIN_DAYS = 10   # options chains are only pulled this close to the report

INFO_FIELDS = {
    "currentPrice": "price", "recommendationMean": "rating_mean", "numberOfAnalystOpinions": "analysts",
    "targetMeanPrice": "target_mean", "targetMedianPrice": "target_median", "targetHighPrice": "target_high",
    "targetLowPrice": "target_low", "sharesShort": "shares_short", "sharesShortPriorMonth": "shares_short_prior",
    "shortPercentOfFloat": "short_pct_float", "shortRatio": "short_days_to_cover",
    "dateShortInterest": "short_interest_date", "heldPercentInsiders": "insider_pct",
    "heldPercentInstitutions": "institution_pct",
}


def upcoming_reports(within_days: int) -> pd.DataFrame:
    """Next report per ticker, for those reporting from today through `within_days` ahead."""
    today = pd.Timestamp.now(tz=TZ).normalize()
    rows = []
    for ticker in load_tickers():
        path = EARNINGS_DIR / f"{ticker}.parquet"
        if not path.exists():
            continue
        ts = pd.to_datetime(pd.read_parquet(path)["earnings_ts"], utc=True).dt.tz_convert(TZ)
        soon = ts[(ts >= today) & (ts < today + pd.Timedelta(days=within_days + 1))]
        if len(soon):
            rows.append({"ticker": ticker, "earnings_ts": soon.min()})
    return pd.DataFrame(rows)


def quarter_row(frame: pd.DataFrame | None, prefix: str, columns: dict[str, str]) -> dict:
    """Values for the quarter about to be reported ('0q') from one of Yahoo's estimate tables."""
    if frame is None or "0q" not in frame.index:
        return {}
    row = frame.loc["0q"]
    return {f"{prefix}{new}": row.get(old) for old, new in columns.items()}


def implied_move(t: yf.Ticker, price: float, earnings_ts: pd.Timestamp) -> dict:
    """At-the-money straddle price as a share of the stock price, for the first expiry
    that covers the reaction session. It is the move the options market is pricing in
    between now and that expiry, which is mostly the earnings reaction when they are close."""
    after_close = earnings_ts.hour + earnings_ts.minute / 60 >= 9.5
    day = earnings_ts.date()
    expiries = [e for e in t.options if (pd.Timestamp(e).date() > day if after_close else pd.Timestamp(e).date() >= day)]
    if not expiries or not price:
        return {}
    chain = t.option_chain(expiries[0])
    calls, puts = chain.calls.set_index("strike"), chain.puts.set_index("strike")
    strikes = calls.index.intersection(puts.index)
    if not len(strikes):
        return {}
    strike = strikes[np.abs(strikes - price).argmin()]

    def mid(row: pd.Series) -> float:
        return (row["bid"] + row["ask"]) / 2 if row["bid"] > 0 and row["ask"] > 0 else row["lastPrice"]

    straddle = mid(calls.loc[strike]) + mid(puts.loc[strike])
    move = straddle / price
    if not np.isfinite(move) or not 0 < move < 1:
        return {}
    return {"implied_move": move, "implied_move_expiry": expiries[0], "implied_move_strike": float(strike),
            "atm_iv": float((calls.loc[strike, "impliedVolatility"] + puts.loc[strike, "impliedVolatility"]) / 2)}


def fetch(ticker: str, earnings_ts: pd.Timestamp, with_options: bool) -> dict:
    """One ticker's snapshot. Each block is best-effort: a failed block leaves its fields empty."""
    row = {"ticker": ticker, "earnings_date": earnings_ts.strftime("%Y-%m-%d")}
    t = yf.Ticker(ticker)

    def attempt(fn):
        try:
            row.update(fn() or {})
        except Exception:
            time.sleep(1)

    def info():
        i = t.info
        return {new: i.get(old) for old, new in INFO_FIELDS.items()}

    def ratings():
        r = t.recommendations
        now = r[r["period"] == "0m"].iloc[0]
        return {f"rating_{k}": int(now[c]) for k, c in [("strong_buy", "strongBuy"), ("buy", "buy"), ("hold", "hold"),
                                                         ("sell", "sell"), ("strong_sell", "strongSell")]}

    def insiders():
        p = t.insider_purchases.set_index(t.insider_purchases.columns[0])
        return {"insider_buys_6m": p.loc["Purchases", "Trans"], "insider_sells_6m": p.loc["Sales", "Trans"],
                "insider_net_shares_6m": p.loc["Net Shares Purchased (Sold)", "Shares"]}

    attempt(info)
    attempt(ratings)
    attempt(lambda: quarter_row(t.eps_trend, "eps_est_", {"current": "now", "7daysAgo": "7d_ago", "30daysAgo": "30d_ago",
                                                           "60daysAgo": "60d_ago", "90daysAgo": "90d_ago"}))
    attempt(lambda: quarter_row(t.eps_revisions, "eps_rev_", {"upLast7days": "up_7d", "upLast30days": "up_30d",
                                                               "downLast7Days": "down_7d", "downLast30days": "down_30d"}))
    attempt(lambda: quarter_row(t.earnings_estimate, "eps_est_", {"low": "low", "high": "high",
                                                                   "numberOfAnalysts": "analysts"}))
    attempt(lambda: quarter_row(t.revenue_estimate, "rev_est_", {"avg": "now", "low": "low", "high": "high",
                                                                  "yearAgoRevenue": "year_ago", "growth": "growth"}))
    attempt(insiders)
    if with_options:
        attempt(lambda: implied_move(t, row.get("price"), earnings_ts))
    return row


def collect(within_days: int) -> None:
    reports = upcoming_reports(within_days)
    now = pd.Timestamp.now(tz=TZ)
    options_cutoff = now.normalize() + pd.Timedelta(days=OPTIONS_WITHIN_DAYS + 1)
    jobs = [(r.ticker, r.earnings_ts, r.earnings_ts < options_cutoff) for r in reports.itertuples()]
    print(f"{len(jobs)} companies report within {within_days} days")
    with ThreadPoolExecutor(max_workers=3) as pool:
        rows = list(pool.map(lambda j: fetch(*j), jobs))
    df = pd.DataFrame(rows)
    df.insert(1, "snapshot_date", now.strftime("%Y-%m-%d"))
    df.insert(2, "snapshot_ts", now.strftime("%Y-%m-%d %H:%M %Z"))
    numeric = [c for c in df.columns if c not in ("ticker", "snapshot_date", "snapshot_ts", "earnings_date", "implied_move_expiry")]
    df[numeric] = df[numeric].apply(pd.to_numeric, errors="coerce")
    SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)
    out = SNAPSHOT_DIR / f"{now:%Y-%m-%d}.parquet"
    df.to_parquet(out, index=False)
    filled = df.drop(columns=["ticker", "snapshot_date", "snapshot_ts", "earnings_date"]).notna().mean().round(2)
    print(f"{len(df)} rows -> {out}")
    print("share filled:", filled[["price", "rating_mean", "eps_est_now", "eps_rev_up_30d", "short_pct_float",
                                   "insider_net_shares_6m"] + (["implied_move"] if "implied_move" in filled else [])].to_dict())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--within-days", type=int, default=14)
    collect(parser.parse_args().within_days)
