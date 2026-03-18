from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

ROOT = Path(r"C:\Users\temen\5 joulu\TrailingStopLoss pivot peak trading 6 march 2026")
M15_PATH = ROOT / "forex raw data" / "Dukascopy" / "weekly_monthly_analysis" / "2003_20260306_15m" / "eurjpy" / "derived_15m" / "eurjpy_15m_20030804_20260306.csv"
DAILY_PATH = ROOT / "pair_workspaces" / "eurjpy_2003_20260306_15m_v1" / "eurjpy_daily_ohlcv_ver3.csv"
OUT_DIR = Path(__file__).parent

START_DATE = pd.Timestamp("2003-01-01")
END_DATE = pd.Timestamp("2015-12-31")
ENTRY_NORM = 0.225
THRESH_NORM = 0.55
MIN_SIDE_RANGE_PIPS = 10.0
PIP = 0.01


def load():
    d = pd.read_csv(DAILY_PATH, usecols=["Date", "High", "Low", "Close"], low_memory=False)
    d["Date"] = pd.to_datetime(d["Date"], errors="coerce")
    for c in ["High", "Low", "Close"]:
        d[c] = pd.to_numeric(d[c], errors="coerce")
    d = d.dropna().sort_values("Date").reset_index(drop=True)
    d["yh"] = d["High"].shift(1)
    d["yl"] = d["Low"].shift(1)
    d["yc"] = d["Close"].shift(1)
    d["pp"] = (d["yh"] + d["yl"] + d["yc"]) / 3.0
    d["s2"] = d["pp"] - (d["yh"] - d["yl"])
    d = d.dropna(subset=["pp", "s2"]).copy()
    d["dkey"] = d["Date"].dt.normalize()

    m = pd.read_csv(M15_PATH, usecols=["Datetime", "High", "Low"], low_memory=False)
    m["Datetime"] = pd.to_datetime(m["Datetime"], errors="coerce")
    for c in ["High", "Low"]:
        m[c] = pd.to_numeric(m[c], errors="coerce")
    m = m.dropna().sort_values("Datetime").reset_index(drop=True)
    m["dkey"] = m["Datetime"].dt.normalize()
    full = set(m.groupby("dkey").size().pipe(lambda s: s[s == 96]).index)
    m = m[m["dkey"].isin(full)].copy()

    bars = {k: {"h": g["High"].to_numpy(float), "l": g["Low"].to_numpy(float)} for k, g in m.groupby("dkey", sort=True)}
    info = {}
    for _, r in d.iterrows():
        k = r["dkey"]
        if START_DATE <= k <= END_DATE and k in bars:
            info[k] = {"pp": float(r["pp"]), "s2": float(r["s2"])}
    return sorted(info.keys()), info, bars


def main():
    days, info, bars = load()
    min_px = MIN_SIDE_RANGE_PIPS * PIP

    total_buy_22 = 0
    hit_55_before_pivot = 0
    hit_pivot_before_55 = 0
    neither = 0
    pivot_after_55_same_day = 0

    for d in days:
        pp = info[d]["pp"]
        dn = pp - info[d]["s2"]
        if dn < min_px:
            continue

        entry_22 = pp - ENTRY_NORM * dn
        lvl_55 = pp - THRESH_NORM * dn
        h = bars[d]["h"]
        l = bars[d]["l"]

        # first touch of buy 22.5%
        trig = None
        for i in range(96):
            if l[i] <= entry_22 <= h[i]:
                trig = i
                break
        if trig is None:
            continue
        total_buy_22 += 1

        state = None
        first_55_idx = None
        for j in range(trig, 96):
            see_55 = l[j] <= lvl_55 <= h[j]
            see_pivot = l[j] <= pp <= h[j]
            if see_55 and see_pivot:
                # conservative: count as 55 first for risk view
                state = "55_before_pivot"
                first_55_idx = j
                break
            if see_55:
                state = "55_before_pivot"
                first_55_idx = j
                break
            if see_pivot:
                state = "pivot_before_55"
                break
        if state == "55_before_pivot":
            hit_55_before_pivot += 1
            for k in range(first_55_idx + 1, 96):
                if l[k] <= pp <= h[k]:
                    pivot_after_55_same_day += 1
                    break
        elif state == "pivot_before_55":
            hit_pivot_before_55 += 1
        else:
            neither += 1

    out = {
        "pair": "EURJPY",
        "period": [str(START_DATE.date()), str(END_DATE.date())],
        "condition": "after BUY 22.5% touch, does 55% happen before pivot?",
        "buy_22p5_triggers": total_buy_22,
        "hit_55_before_pivot": hit_55_before_pivot,
        "hit_pivot_before_55": hit_pivot_before_55,
        "neither_until_day_end": neither,
        "rate_55_before_pivot": round(hit_55_before_pivot / max(total_buy_22, 1), 6),
        "rate_pivot_before_55": round(hit_pivot_before_55 / max(total_buy_22, 1), 6),
        "pivot_after_55_same_day": pivot_after_55_same_day,
        "rate_pivot_after_55_given_55first": round(pivot_after_55_same_day / max(hit_55_before_pivot, 1), 6),
    }

    out_json = OUT_DIR / "buy_22p5_then_55_before_pivot_diag_2003_2015.json"
    out_json.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=2))
    print(f"saved={out_json}")


if __name__ == "__main__":
    main()

