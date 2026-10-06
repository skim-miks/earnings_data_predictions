"""Does news sentiment before a report add anything to the move-size model?

Usage:
    uv run experiments/news_sentiment.py pull   # headlines for the last 12 months of events (resumable)
    uv run experiments/news_sentiment.py test

Headlines come from Google News search (RSS) for the company name plus a
finance term, over the 14 days before each report. Only headlines dated before
the report date are kept, so nothing written after the announcement leaks in.
Sentiment is the word-list scorer in sentiment.py.
"""

import json
import sys
import time
import xml.etree.ElementTree as ET
from email.utils import parsedate_to_datetime
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).parent.parent))
import sentiment  # noqa: E402
import train_model as tm  # noqa: E402
from collect_data import RAW  # noqa: E402

NEWS_DIR = RAW.parent / "news"
WINDOW_DAYS = 14
MONTHS_BACK = 12
FINANCE_TERMS = "(stock OR shares OR earnings OR analyst)"
NEWS_FEATURES = ["news_count", "news_count_7d", "news_sent_mean", "news_sent_mean_7d", "news_pos_share", "news_neg_share"]


def window_events() -> pd.DataFrame:
    df = pd.read_parquet(tm.EVENTS)
    df = df[(df["lead"] == 1) & ~df["upcoming"]].replace([np.inf, -np.inf], np.nan)
    df["earnings_date"] = pd.to_datetime(df["earnings_date"])
    start = df["earnings_date"].max() - pd.DateOffset(months=MONTHS_BACK)
    names = pd.read_parquet(RAW / "universe.parquet").set_index("ticker")["search_name"]
    df["company"] = df["ticker"].map(names)
    return df, start


def fetch(company: str, day: pd.Timestamp) -> list[dict]:
    after = (day - pd.Timedelta(days=WINDOW_DAYS)).strftime("%Y-%m-%d")
    q = f'"{company}" {FINANCE_TERMS} after:{after} before:{day:%Y-%m-%d}'
    for attempt in range(4):
        r = requests.get("https://news.google.com/rss/search", timeout=30,
                         params={"q": q, "hl": "en-US", "gl": "US", "ceid": "US:en"},
                         headers={"User-Agent": "Mozilla/5.0"})
        if r.status_code == 200:
            break
        time.sleep(30 * (attempt + 1))
    r.raise_for_status()
    items = []
    for item in ET.fromstring(r.content).iter("item"):
        source = item.findtext("source") or ""
        title = (item.findtext("title") or "").removesuffix(f" - {source}")
        items.append({"date": parsedate_to_datetime(item.findtext("pubDate")).strftime("%Y-%m-%d"),
                      "title": title, "source": source})
    return items


def pull(pause: float = 1.2) -> None:
    df, start = window_events()
    events = df[df["earnings_date"] > start]
    NEWS_DIR.mkdir(parents=True, exist_ok=True)
    for i, ev in enumerate(events.itertuples(), 1):
        out = NEWS_DIR / f"{ev.ticker}_{ev.earnings_date:%Y-%m-%d}.json"
        if out.exists():
            continue
        try:
            out.write_text(json.dumps(fetch(ev.company, ev.earnings_date)))
        except Exception as e:
            print(f"[{i}/{len(events)}] {ev.ticker} {ev.earnings_date:%Y-%m-%d}: {type(e).__name__} {e}")
            time.sleep(60)
            continue
        if i % 100 == 0:
            print(f"[{i}/{len(events)}]", flush=True)
        time.sleep(pause)
    print("done")


def news_features(ticker: str, day: pd.Timestamp) -> dict | None:
    path = NEWS_DIR / f"{ticker}_{day:%Y-%m-%d}.json"
    if not path.exists():
        return None
    # Strictly before the report date: the feed only gives dates, not times.
    items = [h for h in json.loads(path.read_text()) if h["date"] < f"{day:%Y-%m-%d}"]
    scores = np.array([sentiment.score(h["title"]) for h in items])
    last_week = np.array([h["date"] >= f"{day - pd.Timedelta(days=7):%Y-%m-%d}" for h in items], dtype=bool)
    return {
        "news_count": len(items),
        "news_count_7d": int(last_week.sum()),
        "news_sent_mean": scores.mean() if len(items) else 0.0,
        "news_sent_mean_7d": scores[last_week].mean() if last_week.any() else 0.0,
        "news_pos_share": (scores >= 0.05).mean() if len(items) else 0.0,
        "news_neg_share": (scores <= -0.05).mean() if len(items) else 0.0,
    }


def forward_auc(X: pd.DataFrame, y: pd.Series, period: pd.Series) -> tuple[float, np.ndarray, np.ndarray]:
    """Train on earlier quarters of the window, test on each later one; pooled test predictions."""
    periods = sorted(period.unique())
    prob = pd.Series(np.nan, index=X.index)
    for p in periods[1:]:
        model = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=1000))
        model.fit(X[period < p], y[period < p])
        prob[period == p] = model.predict_proba(X[period == p])[:, 1]
    ok = prob.notna()
    return roc_auc_score(y[ok], prob[ok]), prob[ok].values, y[ok].values


def bootstrap_gain(y: np.ndarray, base: np.ndarray, plus: np.ndarray, n: int = 2000) -> tuple[float, float]:
    rng = np.random.default_rng(0)
    gains = []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        if y[i].min() == y[i].max():
            continue
        gains.append(roc_auc_score(y[i], plus[i]) - roc_auc_score(y[i], base[i]))
    return tuple(np.percentile(gains, [2.5, 97.5]))


def test() -> None:
    df, start = window_events()
    num = tm.NUM
    train, win = df[df["earnings_date"] <= start], df[df["earnings_date"] > start].copy()
    win["base_score"] = tm.fit_predict(train, win, num)   # move-size model trained only on earlier events
    feats = [news_features(t, d) for t, d in zip(win["ticker"], win["earnings_date"])]
    have = [f is not None for f in feats]
    win = win[have].reset_index(drop=True)
    win = pd.concat([win, pd.DataFrame([f for f in feats if f is not None])], axis=1)
    win["news_count_log"] = np.log1p(win["news_count"])
    win["base_logit"] = np.log(win["base_score"] / (1 - win["base_score"]))
    ret = win["target_raw"]
    print(f"{len(win):,} events from {win['earnings_date'].min():%Y-%m-%d} to {win['earnings_date'].max():%Y-%m-%d}; "
          f"{(win['news_count'] > 0).mean():.0%} have headlines, median {win['news_count'].median():.0f} per event")

    print("\n== Rank correlation with the reaction (events with at least 3 headlines) ==")
    some = win[win["news_count"] >= 3]
    for col in ["news_sent_mean", "news_sent_mean_7d", "news_pos_share", "news_neg_share", "news_count"]:
        r1, p1 = spearmanr(some[col], some["target_raw"])
        r2, p2 = spearmanr(some[col], some["target_raw"].abs())
        print(f"{col:18} vs return: {r1:+.3f} (p={p1:.2f})   vs |return|: {r2:+.3f} (p={p2:.2f})")

    print("\n== Reaction by sentiment third (events with at least 3 headlines) ==")
    third = pd.qcut(some["news_sent_mean"], 3, labels=["most negative", "middle", "most positive"])
    print(some.groupby(third, observed=True).agg(
        events=("target_raw", "size"), mean_sentiment=("news_sent_mean", "mean"), mean_return=("target_raw", "mean"),
        share_up=("target_raw", lambda s: (s > 0).mean()), up_5pct=("target_raw", lambda s: (s >= 0.05).mean()),
        down_5pct=("target_raw", lambda s: (s <= -0.05).mean()), median_abs_move=("target_raw", lambda s: s.abs().median()),
    ).round(3).to_string())

    print("\n== Does adding news to the model help? (train on earlier quarters, test on later) ==")
    period = win["earnings_date"].dt.to_period("Q").astype(str)
    news_cols = ["news_count_log", "news_sent_mean", "news_sent_mean_7d", "news_pos_share", "news_neg_share"]
    targets = {
        "move of 5%+ either way": (ret.abs() >= 0.05).astype(int),
        "up vs down": (ret > 0).astype(int),
        "up 5%+": (ret >= 0.05).astype(int),
        "down 5%+": (ret <= -0.05).astype(int),
    }
    for name, y in targets.items():
        a_base, p_base, yt = forward_auc(win[["base_logit"]], y, period)
        a_plus, p_plus, _ = forward_auc(win[["base_logit"] + news_cols], y, period)
        a_news, _, _ = forward_auc(win[news_cols], y, period)
        lo, hi = bootstrap_gain(yt, p_base, p_plus)
        print(f"{name:24} model only {a_base:.3f} | model + news {a_plus:.3f} | news only {a_news:.3f} | "
              f"gain {a_plus - a_base:+.3f} (95% range {lo:+.3f} to {hi:+.3f})")


if __name__ == "__main__":
    {"pull": pull, "test": test}[sys.argv[1]]()
