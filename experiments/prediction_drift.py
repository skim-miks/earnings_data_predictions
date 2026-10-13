"""How much do predictions made weeks before a report differ from the final one, and which is right?

Usage:
    uv run experiments/prediction_drift.py

The calendar scores every upcoming report with a model trained on features from the
last close before the report, so a report three weeks away is scored on stale
inputs. This rebuilds each past report's features as they stood 5, 10, 15 and 20
trading days earlier, scores them with the same model (trained on earlier years
only), and compares those early scores with the final score and with what happened.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).parent.parent))
import train_model as tm  # noqa: E402
from build_dataset import BIG_MOVE, PEER_FEATURES, peer_groups  # noqa: E402
from collect_data import RAW  # noqa: E402

LEADS = [1, 5, 10, 15, 20]   # trading days before the reaction session; 1 = the last close before the report
VOL = ["vol_1m", "vol_3m", "vol_1y", "vol_1m_vs_3m", "vol_1m_vs_1y"]


def daily_vol(prices: pd.DataFrame) -> pd.DataFrame:
    p = prices.sort_values(["ticker", "date"]).reset_index(drop=True)
    ret = p.groupby("ticker")["Adj Close"].pct_change()
    roll = lambda n: ret.groupby(p["ticker"]).transform(lambda s: s.rolling(n).std())  # noqa: E731
    p["vol_1m"], p["vol_3m"], p["vol_1y"] = roll(21), roll(63), roll(252)
    p["vol_1m_vs_3m"], p["vol_1m_vs_1y"] = p["vol_1m"] / p["vol_3m"], p["vol_1m"] / p["vol_1y"]
    p["row"] = p.groupby("ticker").cumcount()
    return p[["ticker", "date", "row", *VOL]]


def features_at(events: pd.DataFrame, vol: pd.DataFrame, lead: int) -> pd.DataFrame:
    """The model's inputs as they stood `lead` trading days before the reaction session."""
    e = events.drop(columns=VOL + PEER_FEATURES + ["feature_date"])
    e = e.assign(row=e["base_row"] - (lead - 1))
    e = e.merge(vol.rename(columns={"date": "feature_date"}), on=["ticker", "row"], how="left")
    # Peer reactions known by that earlier date.
    one = e.reset_index(drop=True)
    move = one["target_raw"].abs().values
    vs_usual = move / one["past_move_mean_8"].values
    for level in ("Industry", "Sector"):
        tag = level.lower()
        mean, ratio, count = np.full(len(one), np.nan), np.full(len(one), np.nan), np.zeros(len(one))
        for i, peers in peer_groups(one, level):
            count[i] = len(peers)
            mean[i] = move[peers].mean()
            rel = vs_usual[peers]
            if np.isfinite(rel).any():
                ratio[i] = np.median(rel[np.isfinite(rel)])
        one[f"peer_{tag}_move"], one[f"peer_{tag}_move_vs_usual"], one[f"peer_{tag}_count"] = mean, ratio, count
    return one


if __name__ == "__main__":
    df = pd.read_parquet(tm.EVENTS)
    df = df[(df["lead"] == 1) & ~df["upcoming"]].replace([np.inf, -np.inf], np.nan)
    for col in ("earnings_date", "reaction_date", "feature_date"):
        df[col] = pd.to_datetime(df[col])
    prices = pd.read_parquet(RAW / "prices.parquet")
    prices["date"] = pd.to_datetime(prices["date"])
    vol = daily_vol(prices)
    df = df.merge(vol[["ticker", "date", "row"]].rename(columns={"date": "feature_date", "row": "base_row"}),
                  on=["ticker", "feature_date"], how="left")
    df = df[df["base_row"] >= max(LEADS) + 260].sort_values(["ticker", "earnings_date"]).reset_index(drop=True)
    sets = {lead: features_at(df, vol, lead).replace([np.inf, -np.inf], np.nan) for lead in LEADS}

    year = df["earnings_date"].dt.year
    y = (df["target_raw"].abs() >= BIG_MOVE).astype(int)
    scores = pd.DataFrame(index=df.index, columns=LEADS, dtype=float)
    contrib = []
    for test_year in tm.TEST_YEARS:
        train, test = year < test_year, year == test_year
        clip = tm.Clipper(tm.NUM).fit(sets[1][train])
        model = tm.make_model(tm.NUM).fit(clip.transform(sets[1][train]), y[train])   # trained at lead 1, as in production
        for lead in LEADS:
            scores.loc[test, lead] = model.predict_proba(clip.transform(sets[lead][test]))[:, 1]
        # Which inputs move the score between 15 days out and the final close.
        prep, coef = model.named_steps["prep"], model.named_steps["clf"].coef_[0]
        z1 = prep.transform(clip.transform(sets[1][test]))[:, :len(tm.NUM)]
        z15 = prep.transform(clip.transform(sets[15][test]))[:, :len(tm.NUM)]
        contrib.append(pd.DataFrame((z15 - z1) * coef[:len(tm.NUM)], columns=tm.NUM, index=df.index[test]))
    tested = scores[1].notna()
    s, yt = scores[tested], y[tested]
    print(f"{tested.sum():,} reports from {tm.TEST_YEARS[0]}-{tm.TEST_YEARS[-1]}; {yt.mean():.1%} moved {BIG_MOVE:.0%}+\n")

    print("== How good is the score at each distance from the report? ==")
    rows = []
    for lead in LEADS:
        rows.append({"trading days before": lead, "auc": roc_auc_score(yt, s[lead]),
                     "correlation with final score": s[lead].corr(s[1]),
                     "avg change vs final (points)": (s[lead] - s[1]).abs().mean() * 100,
                     "flagged (60%+)": (s[lead] >= 0.6).mean()})
    print(pd.DataFrame(rows).round(3).to_string(index=False))

    print("\n== When the early score (15 days out) and the final score disagree, which is right? ==")
    early, final = s[15], s[1]
    cases = {
        "flagged early and at the end": (early >= 0.6) & (final >= 0.6),
        "flagged early, not at the end": (early >= 0.6) & (final < 0.6),
        "not flagged early, flagged at the end": (early < 0.6) & (final >= 0.6),
        "never flagged": (early < 0.6) & (final < 0.6),
    }
    actual = df.loc[tested, "target_raw"].abs()
    print(pd.DataFrame([{"case": k, "reports": int(m.sum()), "moved 5%+": yt[m].mean(), "median move": actual[m].median(),
                         "avg early score": early[m].mean(), "avg final score": final[m].mean()} for k, m in cases.items()]).round(3).to_string(index=False))
    both = pd.DataFrame({"early": early, "final": final})
    from sklearn.linear_model import LogisticRegression
    lr = LogisticRegression().fit(np.log(both / (1 - both)), yt)
    print(f"weights when both scores predict the outcome together: early {lr.coef_[0][0]:+.2f}, final {lr.coef_[0][1]:+.2f}")
    print(f"AUC final only {roc_auc_score(yt, final):.3f} | average of early and final {roc_auc_score(yt, (early + final) / 2):.3f}")

    print("\n== What moves the score between 15 days out and the final close? (average absolute effect) ==")
    c = pd.concat(contrib).abs().mean().sort_values(ascending=False)
    groups = {"peer reactions": [f for f in tm.NUM if f.startswith("peer_")], "volatility": VOL,
              "own past earnings moves": [f for f in tm.NUM if f.startswith("past_")]}
    print({k: round(float(c[v].sum()), 3) for k, v in groups.items()})
    print(c.head(6).round(3).to_dict())
    flagged_drop = (early >= 0.6) & (final < 0.6)
    net = pd.concat(contrib).loc[flagged_drop[flagged_drop].index]
    print("for reports flagged early but not at the end, net effect by group (positive = pushed the early score up):")
    print({k: round(float(net[v].sum(axis=1).mean()), 3) for k, v in groups.items()})
    early_peer = sets[15].loc[tested, "peer_industry_count"]
    print(f"share with no same-industry peer reported yet: 15 days out {(early_peer == 0).mean():.0%}, "
          f"at the final close {(sets[1].loc[tested, 'peer_industry_count'] == 0).mean():.0%}")
