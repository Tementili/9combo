from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(r"C:\Users\temen\5 joulu\TrailingStopLoss pivot peak trading 6 march 2026")
M15_PATH = ROOT / "forex raw data" / "Dukascopy" / "weekly_monthly_analysis" / "2003_20260306_15m" / "eurjpy" / "derived_15m" / "eurjpy_15m_20030804_20260306.csv"
DAILY_PATH = ROOT / "pair_workspaces" / "eurjpy_2003_20260306_15m_v1" / "eurjpy_daily_ohlcv_ver3.csv"
OUT_DIR = Path(__file__).parent

START_DATE = pd.Timestamp("2003-01-01")
END_DATE = pd.Timestamp("2015-12-31")
ENTRY_NORM = 0.225
MIN_SIDE_RANGE_PIPS = 10.0
PIP = 0.01
BASELINE_TARGET = 0.5097


def load_data():
    d = pd.read_csv(DAILY_PATH, usecols=["Date", "High", "Low", "Close"], low_memory=False)
    d["Date"] = pd.to_datetime(d["Date"], errors="coerce")
    for c in ["High", "Low", "Close"]:
        d[c] = pd.to_numeric(d[c], errors="coerce")
    d = d.dropna().sort_values("Date").reset_index(drop=True)
    d["yh"] = d["High"].shift(1)
    d["yl"] = d["Low"].shift(1)
    d["yc"] = d["Close"].shift(1)
    d["pp"] = (d["yh"] + d["yl"] + d["yc"]) / 3.0
    d["r2"] = d["pp"] + (d["yh"] - d["yl"])
    d["s2"] = d["pp"] - (d["yh"] - d["yl"])
    d = d.dropna(subset=["pp", "r2", "s2"]).copy()
    d["dkey"] = d["Date"].dt.normalize()

    m = pd.read_csv(M15_PATH, usecols=["Datetime", "High", "Low"], low_memory=False)
    m["Datetime"] = pd.to_datetime(m["Datetime"], errors="coerce")
    for c in ["High", "Low"]:
        m[c] = pd.to_numeric(m[c], errors="coerce")
    m = m.dropna().sort_values("Datetime").reset_index(drop=True)
    m["dkey"] = m["Datetime"].dt.normalize()
    full_days = set(m.groupby("dkey").size().pipe(lambda s: s[s == 96]).index)
    m = m[m["dkey"].isin(full_days)].copy()

    bars = {}
    for dkey, g in m.groupby("dkey", sort=True):
        bars[dkey] = {"h": g["High"].to_numpy(float), "l": g["Low"].to_numpy(float)}

    info = {}
    for _, r in d.iterrows():
        k = r["dkey"]
        if START_DATE <= k <= END_DATE and k in bars:
            info[k] = {"pp": float(r["pp"]), "r2": float(r["r2"]), "s2": float(r["s2"])}
    return sorted(info.keys()), info, bars


def build_rows(days, info, bars) -> pd.DataFrame:
    rows = []
    min_px = MIN_SIDE_RANGE_PIPS * PIP
    for d in days:
        pp = info[d]["pp"]
        r2 = info[d]["r2"]
        s2 = info[d]["s2"]
        up_range = r2 - pp
        dn_range = pp - s2
        if up_range < min_px or dn_range < min_px:
            continue

        upper_touch = pp + ENTRY_NORM * up_range
        lower_touch = pp - ENTRY_NORM * dn_range
        h = bars[d]["h"]
        l = bars[d]["l"]

        side = None
        trig_i = None
        ambiguous = False
        for i in range(96):
            hit_lo = l[i] <= lower_touch <= h[i]
            hit_hi = l[i] <= upper_touch <= h[i]
            if hit_lo and hit_hi:
                ambiguous = True
                break
            if hit_lo:
                side = "BUY"
                trig_i = i
                break
            if hit_hi:
                side = "SELL"
                trig_i = i
                break
        if ambiguous or trig_i is None:
            continue

        if side == "BUY":
            denom = dn_range
            # away from pivot on buy side
            away_series = np.maximum((pp - l) / denom, 0.0)
            # toward pivot/through it after trigger
            best_toward = np.max((h[trig_i:] - pp) / denom)
        else:
            denom = up_range
            # away from pivot on sell side
            away_series = np.maximum((h - pp) / denom, 0.0)
            # toward pivot/through it after trigger (mirrored)
            best_toward = np.max((pp - l[trig_i:]) / denom)

        before_max_away = float(np.max(away_series[: trig_i + 1]))
        after_max_away = float(np.max(away_series[trig_i:]))
        full_max_away = float(np.max(away_series))
        rows.append(
            {
                "date": d,
                "side": side,
                "trigger_idx": trig_i,
                "best_toward_pivot_coord": float(best_toward),
                "before_max_away_from_pivot": before_max_away,
                "after_max_away_from_pivot": after_max_away,
                "full_day_max_away_from_pivot": full_max_away,
            }
        )
    return pd.DataFrame(rows)


def threshold_from_bins(df: pd.DataFrame, col: str) -> dict:
    bins = np.arange(0.2, 2.05, 0.05)
    x = df.copy()
    x["bin"] = pd.cut(x[col], bins=bins, right=False)
    gb = x.groupby("bin", observed=False).agg(
        n=("best_toward_pivot_coord", "size"),
        mean_best=("best_toward_pivot_coord", "mean"),
    ).reset_index()
    gb = gb[gb["n"] >= 40].copy()
    if gb.empty:
        return {"threshold_bin": None, "table": []}
    gb["bin"] = gb["bin"].astype(str)
    under = gb[gb["mean_best"] < BASELINE_TARGET].copy()
    if under.empty:
        tbin = None
    else:
        tbin = under.iloc[0].to_dict()
    return {"threshold_bin": tbin, "table": gb.to_dict(orient="records")}


def main():
    days, info, bars = load_data()
    df = build_rows(days, info, bars)

    out = {
        "pair": "EURJPY",
        "period": [str(START_DATE.date()), str(END_DATE.date())],
        "entry_norm": ENTRY_NORM,
        "baseline_mean_best_toward_pivot": BASELINE_TARGET,
        "triggers": int(len(df)),
        "actual_mean_best_toward_pivot": round(float(df["best_toward_pivot_coord"].mean()), 4) if not df.empty else None,
        "threshold_by_before_max_away": threshold_from_bins(df, "before_max_away_from_pivot"),
        "threshold_by_after_max_away": threshold_from_bins(df, "after_max_away_from_pivot"),
        "threshold_by_full_day_max_away": threshold_from_bins(df, "full_day_max_away_from_pivot"),
    }

    out_json = OUT_DIR / "pivot_norm_touch_reversion_threshold_2003_2015_eurjpy.json"
    out_csv = OUT_DIR / "pivot_norm_touch_reversion_threshold_2003_2015_eurjpy_rows.csv"
    out_json.write_text(json.dumps(out, indent=2), encoding="utf-8")
    df.to_csv(out_csv, index=False)

    print("RESULT")
    print(json.dumps(out, indent=2))
    print(f"saved={out_json}")
    print(f"saved={out_csv}")


if __name__ == "__main__":
    main()

