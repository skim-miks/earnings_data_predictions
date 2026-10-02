"""Pull recent Reddit posts about one company and keep them on disk.

Posts are context for the app, not a model feature. Nothing is fetched unless a
ticker is requested explicitly, and each pull adds to data/reddit/<TICKER>.json
so history builds up from the first pull onward.

Needs a Reddit app's credentials in .env (see .env.example).
"""

import json
import os
import time
from pathlib import Path

import requests
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

ROOT = Path(__file__).parent
STORE = ROOT / "data" / "reddit"
SUBREDDITS = ["wallstreetbets", "stocks", "investing", "StockMarket", "options", "SecurityAnalysis"]
SELFTEXT_CHARS = 600

# Tickers that are ordinary words; searching them matches unrelated posts, so
# only the company name is used.
WORD_TICKERS = {
    "ALL", "ARE", "BIG", "CAN", "CAR", "COST", "DAY", "FAST", "FOR", "GAP", "HAS", "HE", "IT",
    "KEY", "LOW", "MAN", "NOW", "ON", "ONE", "OUT", "REAL", "SEE", "SO", "TWO", "WELL", "YOU",
}

# VADER is a general-purpose scorer, so it is given investing terms it does not
# know or reads wrongly. Weights are on VADER's -4 (very negative) to +4 scale.
FINANCE_LEXICON = {
    "bullish": 2.5, "bull": 1.5, "calls": 1.0, "long": 0.8, "moon": 2.5, "mooning": 2.5, "rocket": 2.0,
    "rally": 2.0, "rallies": 2.0, "breakout": 1.8, "beat": 2.0, "beats": 2.0, "upgrade": 2.0, "upgraded": 2.0,
    "outperform": 2.0, "undervalued": 2.0, "buy": 1.5, "buying": 1.2, "squeeze": 1.5, "tendies": 2.0,
    "bearish": -2.5, "bear": -1.5, "puts": -1.0, "short": -1.0, "shorting": -1.2, "dump": -2.0, "dumping": -2.0,
    "crash": -2.8, "tank": -2.2, "tanking": -2.5, "tanked": -2.5, "miss": -2.0, "missed": -2.0, "misses": -2.0,
    "downgrade": -2.0, "downgraded": -2.0, "underperform": -2.0, "overvalued": -2.0, "sell": -1.5, "selling": -1.2,
    "bagholder": -2.0, "bagholding": -2.0, "rekt": -2.5, "guh": -2.5, "selloff": -2.2, "plunge": -2.5,
    "plunges": -2.5, "layoffs": -1.5, "lawsuit": -1.5, "bankruptcy": -3.0, "dilution": -2.0,
    # Neutral in a market context even though VADER scores them.
    "share": 0.0, "shares": 0.0, "gross": 0.0, "yield": 0.0, "vice": 0.0, "tariff": -0.8,
}
SENTIMENT_CUTOFF = 0.05   # VADER's standard boundary between neutral and positive/negative

_analyzer = SentimentIntensityAnalyzer()
_analyzer.lexicon.update(FINANCE_LEXICON)

_token = {"value": None, "expires": 0.0}


class RedditNotConfigured(Exception):
    pass


def load_env() -> None:
    env = ROOT / ".env"
    if not env.exists():
        return
    for line in env.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def credentials() -> tuple[str, str, str]:
    load_env()
    client_id = os.environ.get("REDDIT_CLIENT_ID")
    secret = os.environ.get("REDDIT_CLIENT_SECRET")
    agent = os.environ.get("REDDIT_USER_AGENT")
    if not (client_id and secret and agent):
        raise RedditNotConfigured(
            "Reddit credentials are not set. Copy .env.example to .env and fill in "
            "REDDIT_CLIENT_ID, REDDIT_CLIENT_SECRET and REDDIT_USER_AGENT.")
    return client_id, secret, agent


def access_token() -> tuple[str, str]:
    client_id, secret, agent = credentials()
    if _token["value"] and time.time() < _token["expires"] - 60:
        return _token["value"], agent
    r = requests.post(
        "https://www.reddit.com/api/v1/access_token",
        auth=(client_id, secret),
        data={"grant_type": "client_credentials"},
        headers={"User-Agent": agent},
        timeout=20,
    )
    r.raise_for_status()
    body = r.json()
    _token["value"] = body["access_token"]
    _token["expires"] = time.time() + body.get("expires_in", 3600)
    return _token["value"], agent


def build_query(ticker: str, company: str) -> str:
    terms = [f'"{company}"']
    if len(ticker) >= 3 and ticker not in WORD_TICKERS:
        terms.append(ticker)
    return " OR ".join(terms)


def search(ticker: str, company: str) -> tuple[str, list[dict]]:
    token, agent = access_token()
    query = build_query(ticker, company)
    r = requests.get(
        f"https://oauth.reddit.com/r/{'+'.join(SUBREDDITS)}/search",
        params={"q": query, "restrict_sr": 1, "sort": "new", "t": "month", "limit": 100, "raw_json": 1},
        headers={"Authorization": f"bearer {token}", "User-Agent": agent},
        timeout=20,
    )
    r.raise_for_status()
    posts = []
    for child in r.json()["data"]["children"]:
        d = child["data"]
        posts.append({
            "id": d["id"],
            "subreddit": d["subreddit"],
            "title": d["title"],
            "selftext": (d.get("selftext") or "")[:SELFTEXT_CHARS],
            "score": d["score"],
            "upvote_ratio": d.get("upvote_ratio"),
            "num_comments": d["num_comments"],
            "created_utc": d["created_utc"],
            "permalink": "https://www.reddit.com" + d["permalink"],
        })
    return query, posts


def load(ticker: str) -> dict:
    path = STORE / f"{ticker}.json"
    if path.exists():
        return json.loads(path.read_text())
    return {"ticker": ticker, "pulls": [], "posts": {}}


def score(post: dict) -> float:
    """Sentiment of a post's title and body, from -1 (negative) to +1 (positive)."""
    return _analyzer.polarity_scores(f"{post['title']}. {post['selftext']}")["compound"]


def label(value: float) -> str:
    return "positive" if value >= SENTIMENT_CUTOFF else "negative" if value <= -SENTIMENT_CUTOFF else "neutral"


def sentiment_summary(posts: list[dict]) -> dict | None:
    if not posts:
        return None
    labels = [p["sentiment_label"] for p in posts]
    return {
        "mean": round(sum(p["sentiment"] for p in posts) / len(posts), 3),
        "positive": labels.count("positive"),
        "neutral": labels.count("neutral"),
        "negative": labels.count("negative"),
    }


def summarize(stored: dict) -> dict:
    """What the app shows: counts, sentiment and the most recent posts."""
    now = time.time()
    posts = sorted(stored["posts"].values(), key=lambda p: p["created_utc"], reverse=True)
    # Scored on read rather than stored, so lexicon changes apply to old pulls too.
    posts = [{**p, "sentiment": (s := score(p)), "sentiment_label": label(s)} for p in posts]
    recent = [p for p in posts if now - p["created_utc"] <= 30 * 86400]
    last_week = [p for p in recent if now - p["created_utc"] <= 7 * 86400]
    return {
        "sentiment_7d": sentiment_summary(last_week),
        "sentiment_30d": sentiment_summary(recent),
        "ticker": stored["ticker"],
        "last_pulled": stored["pulls"][-1]["at"] if stored["pulls"] else None,
        "query": stored["pulls"][-1]["query"] if stored["pulls"] else None,
        "posts_7d": sum(now - p["created_utc"] <= 7 * 86400 for p in posts),
        "posts_30d": len(recent),
        "posts_stored": len(posts),
        "posts": recent[:25],
    }


def pull(ticker: str, company: str) -> dict:
    """Fetch the last month of posts, merge them into the ticker's file, return a summary."""
    query, posts = search(ticker, company)
    stored = load(ticker)
    for p in posts:
        stored["posts"][p["id"]] = p   # newest pull wins, so scores stay current
    stored["pulls"].append({"at": time.strftime("%Y-%m-%d %H:%M:%S"), "query": query, "returned": len(posts)})
    STORE.mkdir(parents=True, exist_ok=True)
    (STORE / f"{ticker}.json").write_text(json.dumps(stored))
    return summarize(stored)
