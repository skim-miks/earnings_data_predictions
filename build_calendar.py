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

from collect_data import EARNINGS_DIR, RAW, TICKER_RENAMES, UNIVERSE_FILE

OUT = Path(__file__).parent / "app" / "calendar_data.js"
SCORES = Path(__file__).parent / "data" / "processed" / "upcoming_scores.parquet"
TZ = "America/New_York"
TOP_N = 500   # default calendar view: the largest companies by current market cap
MARKET_CAPS = RAW / "market_caps.parquet"   # from `collect_data.py marketcap`


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
    caps_as_of = "mid-2024 (list file)"
    if MARKET_CAPS.exists():
        caps = pd.read_parquet(MARKET_CAPS)
        caps_as_of = caps["as_of"].max()
        current = f["ticker"].map(caps.set_index("ticker")["market_cap"] / 1e6)
        f["mcap_m"] = current.fillna(f["mcap_m"]).round(0)
    top = set(f.nlargest(TOP_N, "mcap_m")["ticker"])
    upcoming = upcoming.merge(f[["ticker", "company", "sector", "industry", "rank", "mcap_m"]], on="ticker")
    upcoming["top"] = upcoming["ticker"].isin(top)
    upcoming = upcoming.rename(columns={"EPS Estimate": "eps_est"})
    # Move-size model output (train_model.py): chance of a 5%+ move either way.
    upcoming["p_big_move"] = None
    upcoming["past_move"] = None
    scores_as_of = None
    if SCORES.exists():
        scores = pd.read_parquet(SCORES)
        scores_as_of = scores["feature_date"].max().strftime("%Y-%m-%d")
        scores["date"] = scores["earnings_date"].dt.strftime("%Y-%m-%d")
        scores = scores.rename(columns={"past_move_mean_8": "past_move"}).round({"p_big_move": 3, "past_move": 4})
        upcoming = upcoming.drop(columns=["p_big_move", "past_move"]).merge(
            scores[["ticker", "date", "p_big_move", "past_move"]], on=["ticker", "date"], how="left")
    upcoming = upcoming.sort_values(["date", "mcap_m"], ascending=[True, False])

    cols = ["ticker", "company", "sector", "industry", "rank", "mcap_m", "date", "session", "eps_est",
            "p_big_move", "past_move", "top"]
    records = json.loads(upcoming[cols].to_json(orient="records"))
    payload = {
        "generated": pd.Timestamp.now(tz=TZ).strftime("%Y-%m-%d %H:%M %Z"),
        "quarter": f"Q{quarter.quarter} {quarter.year}",
        "quarter_start": quarter.start_time.strftime("%Y-%m-%d"),
        "quarter_end": quarter.end_time.strftime("%Y-%m-%d"),
        "scores_as_of": scores_as_of,
        "top_n": TOP_N,
        "caps_as_of": caps_as_of,
        "events": records,
    }
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text("window.EARNINGS_CALENDAR = " + json.dumps(payload) + ";\n")
    print(f"{len(records)} upcoming events for {upcoming['ticker'].nunique()} companies "
          f"({upcoming['date'].min()} to {upcoming['date'].max()}) -> {OUT}")


if __name__ == "__main__":
    main()
