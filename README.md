# earnings_data_predictions

Predict whether a stock makes a large move in the session after it reports earnings.

This is a restart. The first version (notebooks, models, React app) is in git history at commit `80f5861`.

## Data collection

```bash
uv sync
uv run collect_data.py prices
uv run collect_data.py earnings
uv run collect_data.py combine
```

Outputs go to `data/raw/` (not committed):

| File | Contents |
|---|---|
| `earnings.parquet` | One row per ticker and earnings event from 2018: timestamp (US Eastern), EPS estimate, reported EPS, surprise %, and `report_timing` (`bmo` before open, `amc` after close, `during`) |
| `prices.parquet` | Daily unadjusted OHLC, adjusted close, volume, dividends and splits from 2017, for the universe plus SPY, QQQ, IWM, VIX and the SPDR sector ETFs |
| `universe.parquet` | Ticker, company, sector, industry, HQ city and state |

The universe is the public companies in `reference/fortune1000_2024.csv`.

## Known issues carried over from v1

- The v1 label was the close-to-close return on the earnings day, which is the reaction only for pre-market reporters. Use `report_timing` to pick the reaction session.
- The universe is 2024 Fortune 1000 membership applied to earlier years (survivorship bias).

## App

```bash
uv run build_calendar.py   # refresh app/calendar_data.js from the earnings pull
uv run serve.py            # http://localhost:8765/calendar.html
```

The calendar shows this quarter's upcoming reports in month and week views. Clicking a date lists the companies reporting; the Reddit button next to a company pulls the last month of posts about it from a few investing subreddits and stores them in `data/reddit/`. Reddit is only called for the ticker clicked, and the posts are context for the app, not a model feature.

The Reddit button needs credentials: copy `.env.example` to `.env` and fill it in.
