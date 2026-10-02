"""Pull raw earnings events and daily prices from Yahoo Finance.

Usage:
    uv run collect_data.py earnings   # earnings dates, EPS estimate/actual/surprise
    uv run collect_data.py prices     # daily OHLCV for universe + benchmarks
    uv run collect_data.py combine    # merge per-ticker files and print a summary

Each ticker is written to its own parquet file, so an interrupted run resumes
where it stopped. Use --refresh to re-pull tickers that already have a file.
"""

import argparse
import time
from pathlib import Path

import pandas as pd
import yfinance as yf

ROOT = Path(__file__).parent
UNIVERSE_FILE = ROOT / "reference" / "fortune1000_2024.csv"
RAW = ROOT / "data" / "raw"
EARNINGS_DIR = RAW / "earnings"
PRICES_DIR = RAW / "prices"

# Modeling window starts in 2018. Prices start a year earlier so 1-year
# lookback features exist for the first events.
EVENT_START = "2018-01-01"
PRICE_START = "2017-01-01"
EARNINGS_LIMIT = 100

# Market / sector context for market-relative features.
BENCHMARKS = [
    "SPY", "QQQ", "IWM", "^VIX",
    "XLB", "XLC", "XLE", "XLF", "XLI", "XLK", "XLP", "XLRE", "XLU", "XLV", "XLY",
]

# Symbols that changed since the 2024 list was published (old -> current Yahoo symbol).
TICKER_RENAMES = {
    "SQ": "XYZ", "GPS": "GAP", "NYCB": "FLG", "CHK": "EXE", "TPX": "SGI",
    "FI": "FISV", "BK": "BNY", "MMC": "MRSH", "PSTG": "P", "CEIX": "CNR",
}

MARKET_OPEN = pd.Timestamp("09:30").time()
MARKET_CLOSE = pd.Timestamp("16:00").time()


def load_universe() -> pd.DataFrame:
    f = pd.read_csv(UNIVERSE_FILE)
    f = f[f["CompanyType"] == "Public"]
    f["Ticker"] = f["Ticker"].replace(TICKER_RENAMES)
    f = f[["Ticker", "Company", "Sector", "Industry", "HeadquartersCity", "HeadquartersState"]]
    return f.rename(columns={"Ticker": "ticker"}).drop_duplicates("ticker").reset_index(drop=True)


def fetch_earnings(ticker: str, retries: int = 3) -> pd.DataFrame | None:
    for attempt in range(retries):
        try:
            df = yf.Ticker(ticker).get_earnings_dates(limit=EARNINGS_LIMIT)
            if df is None or df.empty:
                return None
            df = df.reset_index().rename(columns={"Earnings Date": "earnings_ts"})
            df.insert(0, "ticker", ticker)
            return df
        except Exception as e:
            wait = 20 * (attempt + 1)
            print(f"  {ticker}: {type(e).__name__}: {e} (retry in {wait}s)")
            time.sleep(wait)
    return None


def collect_earnings(refresh: bool, pause: float) -> None:
    EARNINGS_DIR.mkdir(parents=True, exist_ok=True)
    tickers = load_universe()["ticker"].tolist()
    failed = []
    for i, ticker in enumerate(tickers, 1):
        out = EARNINGS_DIR / f"{ticker}.parquet"
        if out.exists() and not refresh:
            continue
        df = fetch_earnings(ticker)
        if df is None:
            failed.append(ticker)
            print(f"[{i}/{len(tickers)}] {ticker}: no data")
        else:
            df.to_parquet(out, index=False)
            print(f"[{i}/{len(tickers)}] {ticker}: {len(df)} rows")
        time.sleep(pause)
    (RAW / "earnings_failed.txt").write_text("\n".join(failed))
    print(f"done. failed: {len(failed)}")


def collect_prices(refresh: bool, pause: float, batch_size: int = 40) -> None:
    PRICES_DIR.mkdir(parents=True, exist_ok=True)
    tickers = load_universe()["ticker"].tolist() + BENCHMARKS
    todo = [t for t in tickers if refresh or not (PRICES_DIR / f"{t}.parquet").exists()]
    failed = []
    for start in range(0, len(todo), batch_size):
        batch = todo[start : start + batch_size]
        # Unadjusted OHLC plus Adj Close, dividends and splits, so both raw and
        # adjusted series can be rebuilt later.
        data = yf.download(
            batch, start=PRICE_START, auto_adjust=False, actions=True,
            group_by="ticker", progress=False, threads=True,
        )
        for ticker in batch:
            try:
                df = data[ticker].dropna(how="all")
            except KeyError:
                df = pd.DataFrame()
            if df.empty:
                failed.append(ticker)
                continue
            df = df.reset_index().rename(columns={"Date": "date"})
            df.columns.name = None
            df.insert(0, "ticker", ticker)
            df.to_parquet(PRICES_DIR / f"{ticker}.parquet", index=False)
        print(f"[{min(start + batch_size, len(todo))}/{len(todo)}] failed so far: {len(failed)}")
        time.sleep(pause)
    (RAW / "prices_failed.txt").write_text("\n".join(failed))
    print(f"done. failed: {len(failed)} {failed[:20]}")


def report_timing(ts: pd.Series) -> pd.Series:
    """Classify the ET report time: before open, after close, or during the session."""
    t = ts.dt.time
    out = pd.Series("during", index=ts.index)
    out[t < MARKET_OPEN] = "bmo"
    out[t >= MARKET_CLOSE] = "amc"
    return out


def combine() -> None:
    earnings = pd.concat([pd.read_parquet(p) for p in sorted(EARNINGS_DIR.glob("*.parquet"))], ignore_index=True)
    earnings["earnings_ts"] = pd.to_datetime(earnings["earnings_ts"], utc=True).dt.tz_convert("America/New_York")
    earnings["earnings_date"] = earnings["earnings_ts"].dt.date
    earnings["report_timing"] = report_timing(earnings["earnings_ts"])
    earnings = earnings[earnings["earnings_ts"] >= pd.Timestamp(EVENT_START, tz="America/New_York")]
    earnings = earnings.sort_values(["ticker", "earnings_ts"]).reset_index(drop=True)
    earnings.to_parquet(RAW / "earnings.parquet", index=False)

    prices = pd.concat([pd.read_parquet(p) for p in sorted(PRICES_DIR.glob("*.parquet"))], ignore_index=True)
    prices = prices[pd.to_datetime(prices["date"]) >= PRICE_START]
    prices["date"] = pd.to_datetime(prices["date"]).dt.date
    prices = prices.sort_values(["ticker", "date"]).reset_index(drop=True)
    prices.to_parquet(RAW / "prices.parquet", index=False)

    load_universe().to_parquet(RAW / "universe.parquet", index=False)

    reported = earnings.dropna(subset=["Reported EPS"])
    print(f"earnings: {len(earnings):,} rows, {earnings['ticker'].nunique()} tickers, "
          f"{earnings['earnings_date'].min()} to {earnings['earnings_date'].max()}")
    print(f"  with reported EPS: {len(reported):,}")
    print(f"  report timing: {earnings['report_timing'].value_counts().to_dict()}")
    print(f"prices: {len(prices):,} rows, {prices['ticker'].nunique()} tickers, "
          f"{prices['date'].min()} to {prices['date'].max()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("step", choices=["earnings", "prices", "combine"])
    parser.add_argument("--refresh", action="store_true", help="re-pull tickers that already have a file")
    parser.add_argument("--pause", type=float, default=1.0, help="seconds to sleep between requests")
    args = parser.parse_args()

    if args.step == "earnings":
        collect_earnings(args.refresh, args.pause)
    elif args.step == "prices":
        collect_prices(args.refresh, args.pause)
    else:
        combine()
