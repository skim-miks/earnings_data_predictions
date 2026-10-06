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

from collect_data import EARNINGS_DIR, load_universe

OUT = Path(__file__).parent / "app" / "calendar_data.js"
SCORES = Path(__file__).parent / "data" / "processed" / "upcoming_scores.parquet"
SNAPSHOTS = Path(__file__).parent / "data" / "snapshots"   # from collect_snapshot.py
SNAPSHOT_MAX_AGE_DAYS = 5
TZ = "America/New_York"
TOP_N = 500   # default calendar view: the largest companies by current market cap


def session(ts: pd.Series) -> pd.Series:
    # Yahoo's upcoming times are placeholders (12:00 / 20:00 UTC), so bucket loosely.
    hour = ts.dt.hour + ts.dt.minute / 60
    out = pd.Series("during", index=ts.index)
    out[hour < 9.5] = "bmo"
    out[hour >= 15] = "amc"
    return out


def pct_change(now, before):
    return (now - before) / before.abs().where(before.abs() > 0)


def load_context() -> dict[str, dict]:
    """Latest analyst / estimate / short-interest / options snapshot per ticker, shaped for the app."""
    files = sorted(SNAPSHOTS.glob("*.parquet"))[-SNAPSHOT_MAX_AGE_DAYS:]
    if not files:
        return {}
    s = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    cutoff = (pd.Timestamp.now(tz=TZ) - pd.Timedelta(days=SNAPSHOT_MAX_AGE_DAYS)).strftime("%Y-%m-%d")
    s = s[s["snapshot_date"] >= cutoff].sort_values("snapshot_ts").drop_duplicates("ticker", keep="last")
    for col in ("implied_move", "implied_move_expiry"):
        if col not in s:
            s[col] = None
    c = pd.DataFrame({
        "ticker": s["ticker"], "as_of": s["snapshot_ts"],
        "rating": s["rating_mean"].round(2), "analysts": s["analysts"],
        "buys": s["rating_strong_buy"] + s["rating_buy"], "holds": s["rating_hold"],
        "sells": s["rating_sell"] + s["rating_strong_sell"],
        "target": s["target_mean"].round(2), "target_upside": (s["target_mean"] / s["price"] - 1).round(4),
        "eps_est": s["eps_est_now"].round(2),
        "eps_chg_30d": pct_change(s["eps_est_now"], s["eps_est_30d_ago"]).round(4),
        "eps_chg_90d": pct_change(s["eps_est_now"], s["eps_est_90d_ago"]).round(4),
        "rev_up_30d": s["eps_rev_up_30d"], "rev_down_30d": s["eps_rev_down_30d"],
        "revenue_est_m": (s["rev_est_now"] / 1e6).round(0), "revenue_growth": s["rev_est_growth"].round(4),
        "short_pct": s["short_pct_float"].round(4), "short_days": s["short_days_to_cover"].round(1),
        "short_chg": (s["shares_short"] / s["shares_short_prior"] - 1).round(4),
        "insider_buys": s["insider_buys_6m"], "insider_sells": s["insider_sells_6m"],
        "implied_move": pd.to_numeric(s["implied_move"], errors="coerce").round(4),
        "implied_expiry": s["implied_move_expiry"],
    })
    records = json.loads(c.to_json(orient="records"))
    return {r.pop("ticker"): r for r in records}


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

    # Company details and size are current Yahoo values; the 2024 list only supplies tickers.
    f = load_universe().rename(columns={"Company": "company", "Sector": "sector", "Industry": "industry"})
    f["mcap_m"] = (f["market_cap"] / 1e6).round(0)
    caps_as_of = f["as_of"].max()
    top = set(f.nlargest(TOP_N, "mcap_m")["ticker"])
    upcoming = upcoming.merge(f[["ticker", "company", "sector", "industry", "mcap_m"]], on="ticker")
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

    cols = ["ticker", "company", "sector", "industry", "mcap_m", "date", "session", "eps_est",
            "p_big_move", "past_move", "top"]
    records = json.loads(upcoming[cols].to_json(orient="records"))
    context = load_context()
    for r in records:
        r["ctx"] = context.get(r["ticker"])
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
    print(f"{sum(r['ctx'] is not None for r in records)} with analyst/options context")
    print(f"{len(records)} upcoming events for {upcoming['ticker'].nunique()} companies "
          f"({upcoming['date'].min()} to {upcoming['date'].max()}) -> {OUT}")


if __name__ == "__main__":
    main()
