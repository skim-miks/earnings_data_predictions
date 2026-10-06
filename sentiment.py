"""Word-list sentiment scorer for short finance text (headlines, posts): VADER plus investing terms."""

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

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

analyzer = SentimentIntensityAnalyzer()
analyzer.lexicon.update(FINANCE_LEXICON)


def score(text: str) -> float:
    """Sentiment from -1 (negative) to +1 (positive)."""
    return analyzer.polarity_scores(text)["compound"]
