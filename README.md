# earnings_data_predictions

Predict whether a stock makes a large move in the session after it reports earnings.

This is a restart. The first version (notebooks, models, React app) is in git history at commit `80f5861`.

## Data collection

```bash
uv sync
uv run collect_data.py prices
uv run collect_data.py earnings
uv run collect_data.py profiles
uv run collect_data.py combine
```

Outputs go to `data/raw/` (not committed):

| File | Contents |
|---|---|
| `earnings.parquet` | One row per ticker and earnings event from 2018: timestamp (US Eastern), EPS estimate, reported EPS, surprise %, and `report_timing` (`bmo` before open, `amc` after close, `during`) |
| `prices.parquet` | Daily unadjusted OHLC, adjusted close, volume, dividends and splits from 2017, for the universe plus SPY, QQQ, IWM, VIX and the SPDR sector ETFs |
| `universe.parquet` | Current company profile from Yahoo: name, sector, industry, HQ city and state, market cap, revenue |

`reference/fortune1000_2024.csv` supplies the list of tickers only. Company names, sectors, market caps and everything else are pulled fresh from Yahoo.

## Known issues carried over from v1

- The v1 label was the close-to-close return on the earnings day, which is the reaction only for pre-market reporters. Use `report_timing` to pick the reaction session.
- The universe is 2024 Fortune 1000 membership applied to earlier years (survivorship bias).

## App

```bash
uv run build_calendar.py                      # refresh app/calendar_data.js
python3 -m http.server 8765 --directory app   # http://localhost:8765/calendar.html
```

The calendar shows this quarter's upcoming reports in month and week views, with the move-size model's chance of a 5%+ move for each company. Clicking a date lists the companies reporting. `app/calendar.html` also opens directly as a file.

## Hosted calendar

The `app/` folder is published to GitHub Pages at
https://skim-miks.github.io/earnings_data_predictions/ and can be added to a
phone's home screen (Safari: Share, then Add to Home Screen).

- `.github/workflows/pages.yml` publishes `app/` whenever it changes on `main`.
- `.github/workflows/refresh.yml` runs on weekdays after the US close: it re-pulls
  prices, company profiles and earnings (only companies reporting within two weeks
  or that just reported; every company on Fridays), rebuilds the dataset, retrains the
  move-size model, rebuilds the calendar data and publishes it. It can also be
  started by hand from the Actions tab.
