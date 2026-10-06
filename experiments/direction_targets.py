"""Direction, tested against different targets and with analyst activity added.

Usage:
    uv run collect_data.py analysts
    uv run experiments/direction_targets.py

Targets:
    up 3%+ vs rest      the original v1 target (big drops and flat both count as 0)
    up 5%+ vs rest
    up vs down          every report, no neutral band
    clear moves only    trained and tested only on reports that moved 5%+: up or down?

Feature sets: the move-size model's inputs, the direction groups from
experiments/direction.py, analyst rating / price-target activity before the
report, and combinations. Train on earlier years, test on each of 2022-2026.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))
import direction as dr  # noqa: E402
import train_model as tm  # noqa: E402
from build_dataset import BIG_MOVE  # noqa: E402
from collect_data import ANALYSTS_DIR  # noqa: E402

POSITIVE = {"buy", "overweight", "outperform", "strong buy", "positive", "market outperform",
            "sector outperform", "accumulate", "add", "top pick", "conviction buy"}
NEGATIVE = {"underweight", "underperform", "sell", "reduce", "negative", "strong sell",
            "market underperform", "sector underperform"}


def analyst_features(df: pd.DataFrame) -> list[str]:
    """Analyst actions dated before each event's feature date, over 30- and 90-day windows."""
    a = pd.concat([pd.read_parquet(p) for p in sorted(ANALYSTS_DIR.glob("*.parquet"))], ignore_index=True)
    a["day"] = pd.to_datetime(a["GradeDate"]).dt.normalize()
    grade = a["ToGrade"].str.lower().str.strip()
    a["grade"] = np.where(grade.isin(POSITIVE), 1.0, np.where(grade.isin(NEGATIVE), -1.0, 0.0))
    pt = a["priceTargetAction"].str.lower()
    both = (a["currentPriceTarget"] > 0) & (a["priorPriceTarget"] > 0)
    a["pt_change"] = np.where(both, (a["currentPriceTarget"] / a["priorPriceTarget"].where(both) - 1).clip(-0.5, 0.5), 0.0)
    flags = {"n": np.ones(len(a)), "up": a["Action"] == "up", "down": a["Action"] == "down",
             "pt_raise": pt == "raises", "pt_lower": pt == "lowers", "pt_change": a["pt_change"],
             "pt_n": both, "grade": a["grade"]}
    for k, v in flags.items():
        a[k] = np.asarray(v, dtype=float)
    a = a.sort_values(["ticker", "day"])

    names = ["an_actions_30", "an_actions_90", "an_net_upgrades_30", "an_net_upgrades_90", "an_downgrades_90",
             "an_upgrades_90", "an_pt_net_30", "an_pt_net_90", "an_pt_change_30", "an_pt_change_90",
             "an_grade_mean_365", "an_grade_shift"]
    out = {n: np.full(len(df), np.nan) for n in names}
    by_ticker = {t: g for t, g in a.groupby("ticker")}
    for ticker, events in df.groupby("ticker"):
        g = by_ticker.get(ticker)
        if g is None:
            continue
        days = g["day"].values
        cum = {k: np.concatenate([[0.0], g[k].values.cumsum()]) for k in flags}
        for idx, fdate in zip(events.index, events["feature_date"].values):
            hi = np.searchsorted(days, fdate, side="left")   # strictly before the feature date
            lo = {w: np.searchsorted(days, fdate - np.timedelta64(w, "D"), side="left") for w in (30, 90, 365)}
            s = lambda k, w: cum[k][hi] - cum[k][lo[w]]  # noqa: E731
            if s("n", 365) == 0:
                continue
            out["an_actions_30"][idx], out["an_actions_90"][idx] = s("n", 30), s("n", 90)
            out["an_net_upgrades_30"][idx] = s("up", 30) - s("down", 30)
            out["an_net_upgrades_90"][idx] = s("up", 90) - s("down", 90)
            out["an_upgrades_90"][idx], out["an_downgrades_90"][idx] = s("up", 90), s("down", 90)
            for w in (30, 90):
                moves = s("pt_raise", w) + s("pt_lower", w)
                out[f"an_pt_net_{w}"][idx] = (s("pt_raise", w) - s("pt_lower", w)) / moves if moves else 0.0
                out[f"an_pt_change_{w}"][idx] = s("pt_change", w) / s("pt_n", w) if s("pt_n", w) else 0.0
            out["an_grade_mean_365"][idx] = s("grade", 365) / s("n", 365)
            recent = s("grade", 90) / s("n", 90) if s("n", 90) else np.nan
            out["an_grade_shift"][idx] = recent - out["an_grade_mean_365"][idx]
    for k, v in out.items():
        df[k] = v
    return names


def evaluate(df: pd.DataFrame, y: pd.Series, cols: list[str], kind: str) -> dict:
    prob = dr.out_of_time(df, y, cols, kind)
    year = df["earnings_date"].dt.year
    tested = prob.notna().values
    lo, hi = dr.auc_range(y.values[tested], prob.values[tested], n=300)
    by_year = [roc_auc_score(y[year == t], prob[year == t]) for t in tm.TEST_YEARS]
    return {"auc": roc_auc_score(y[tested], prob[tested]), "low": lo, "high": hi,
            "years_above_0.5": sum(a > 0.5 for a in by_year), "prob": prob}


if __name__ == "__main__":
    df = pd.read_parquet(tm.EVENTS)
    df = df[(df["lead"] == 1) & ~df["upcoming"]].replace([np.inf, -np.inf], np.nan)
    df["earnings_date"] = pd.to_datetime(df["earnings_date"])
    df = df.sort_values(["ticker", "earnings_date"]).reset_index(drop=True)
    groups = dr.add_features(df)
    direction = list(dict.fromkeys(c for cols in groups.values() for c in cols))
    analyst = analyst_features(df)
    df = df.replace([np.inf, -np.inf], np.nan)
    print(f"{len(df):,} reports; analyst coverage {df['an_actions_90'].notna().mean():.0%}, "
          f"median {df['an_actions_90'].median():.0f} actions in the 90 days before a report")

    ret = df["target_raw"]
    clear = (ret.abs() >= BIG_MOVE).values
    targets = {
        "up 3%+ vs rest (original)": (df, (ret >= 0.03).astype(int)),
        "up 5%+ vs rest": (df, (ret >= 0.05).astype(int)),
        "up vs down, all reports": (df, (ret > 0).astype(int)),
        "clear moves only (5%+): up vs down": (df[clear].reset_index(drop=True), (ret[clear] > 0).astype(int).reset_index(drop=True)),
    }
    feature_sets = {
        "size inputs": tm.NUM,
        "direction groups": direction,
        "analyst activity": analyst,
        "direction + analyst": direction + analyst,
        "everything": list(dict.fromkeys(tm.NUM + direction + analyst)),
    }
    rows, keep = [], {}
    for tname, (data, y) in targets.items():
        year = data["earnings_date"].dt.year
        print(f"\n== {tname}: {int(year.isin(tm.TEST_YEARS).sum()):,} test reports, "
              f"{y[year.isin(tm.TEST_YEARS)].mean():.1%} positive ==")
        table = []
        for fname, cols in feature_sets.items():
            for kind in ("logistic", "boosting"):
                r = evaluate(data, y, cols, kind)
                keep[(tname, fname, kind)] = r.pop("prob")
                table.append({"features": fname, "model": kind, **r})
        print(pd.DataFrame(table).round(3).to_string(index=False))

    tname = "clear moves only (5%+): up vs down"
    data, y = targets[tname]
    best = max(((f, k) for (t, f, k) in keep if t == tname), key=lambda fk: roc_auc_score(
        y[keep[(tname, *fk)].notna()], keep[(tname, *fk)].dropna()))
    prob = keep[(tname, *best)]
    t = data[prob.notna()].assign(score=prob.dropna())
    t["fifth"] = pd.qcut(t["score"], 5, labels=False) + 1
    print(f"\n== Clear moves, best set ({best[0]} / {best[1]}): outcome by score fifth ==")
    print(t.groupby("fifth").agg(reports=("target_raw", "size"), share_up=("target_raw", lambda s: (s > 0).mean()),
                                  mean_return=("target_raw", "mean")).round(3).to_string())
    yr = t["earnings_date"].dt.year
    print("AUC by year:", {int(k): round(roc_auc_score((g["target_raw"] > 0), g["score"]), 3) for k, g in t.groupby(yr)})

    print("\n== Analyst features alone: rank correlation with the reaction (all reports) ==")
    from scipy.stats import spearmanr
    for col in analyst:
        ok = df[col].notna()
        r, p = spearmanr(df.loc[ok, col], ret[ok])
        print(f"{col:20} {r:+.3f} (p={p:.3f})")
