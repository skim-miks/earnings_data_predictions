"""Build the upcoming-earnings data file for app/calendar.html.

Usage:
    uv run collect_data.py earnings --refresh   # re-pull so upcoming dates are current
    uv run build_calendar.py

Reads the per-ticker earnings files, keeps events from today through the end of
the current calendar quarter, and writes app/calendar_data.js.
"""

import json
from pathlib import Path

import pandas as pd

from collect_data import EARNINGS_DIR, TICKER_RENAMES, UNIVERSE_FILE

OUT = Path(__file__).parent / "app" / "calendar_data.js"
TZ = "America/New_York"


def session(ts: pd.Series) -> pd.Series:
    # Yahoo's upcoming times are placeholders (12:00 / 20:00 UTC), so bucket loosely.
    hour = ts.dt.hour + ts.dt.minute / 60
    out = pd.Series("during", index=ts.index)
    out[hour < 9.5] = "bmo"
    out[hour >= 15] = "amc"
    return out


def main() -> None:
    earnings = pd.concat([pd.read_parquet(p) for p in sorted(EARNINGS_DIR.glob("*.parquet"))], ignore_index=True)
    ts = pd.to_datetime(earnings["earnings_ts"], utc=True).dt.tz_convert(TZ)
    today = pd.Timestamp.now(tz=TZ).normalize()
    # Only the current calendar quarter: dates further out are less reliable.
    quarter = today.tz_localize(None).to_period("Q")
    quarter_end = quarter.end_time.tz_localize(TZ)
    upcoming = earnings[(ts >= today) & (ts <= quarter_end)].copy()
    ts = ts[upcoming.index]
    upcoming["date"] = ts.dt.strftime("%Y-%m-%d")
    upcoming["session"] = session(ts)

    f = pd.read_csv(UNIVERSE_FILE)
    f = f[f["CompanyType"] == "Public"].drop_duplicates("Ticker")
    f["Ticker"] = f["Ticker"].replace(TICKER_RENAMES)
    f["mcap_m"] = pd.to_numeric(f["MarketCap_Updated_M"], errors="coerce").fillna(
        pd.to_numeric(f["MarketCap_March28_M"], errors="coerce"))
    f = f.rename(columns={"Ticker": "ticker", "Company": "company", "Sector": "sector",
                          "Industry": "industry", "Rank": "rank"})
    upcoming = upcoming.merge(f[["ticker", "company", "sector", "industry", "rank", "mcap_m"]], on="ticker")
    upcoming = upcoming.rename(columns={"EPS Estimate": "eps_est"})
    upcoming = upcoming.sort_values(["date", "mcap_m"], ascending=[True, False])

    cols = ["ticker", "company", "sector", "industry", "rank", "mcap_m", "date", "session", "eps_est"]
    records = json.loads(upcoming[cols].to_json(orient="records"))
    payload = {
        "generated": pd.Timestamp.now(tz=TZ).strftime("%Y-%m-%d %H:%M %Z"),
        "quarter": f"Q{quarter.quarter} {quarter.year}",
        "quarter_start": quarter.start_time.strftime("%Y-%m-%d"),
        "quarter_end": quarter.end_time.strftime("%Y-%m-%d"),
        "universe_size": int(len(f)),
        "tickers_pulled": int(earnings["ticker"].nunique()),
        "events": records,
    }
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text("window.EARNINGS_CALENDAR = " + json.dumps(payload) + ";\n")
    print(f"{len(records)} upcoming events for {upcoming['ticker'].nunique()} companies "
          f"({upcoming['date'].min()} to {upcoming['date'].max()}) -> {OUT}")


if __name__ == "__main__":
    main()
