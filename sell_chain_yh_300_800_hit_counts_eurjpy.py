from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

ROOT = Path(r"C:\Users\temen\5 joulu\TrailingStopLoss pivot peak trading 6 march 2026")
M15_PATH = ROOT / "forex raw data" / "Dukascopy" / "weekly_monthly_analysis" / "2003_20260306_15m" / "eurjpy" / "derived_15m" / "eurjpy_15m_20030804_20260306.csv"
DAILY_PATH = ROOT / "pair_workspaces" / "eurjpy_2003_20260306_15m_v1" / "eurjpy_daily_ohlcv_ver3.csv"
OUT_DIR = Path(__file__).parent

PIP = 0.01
SL1_ABOVE_YH_PIPS = 300
SL2_ABOVE_YH_PIPS = 800


def load_data():
    d = pd.read_csv(DAILY_PATH, usecols=["Date", "High", "Low"], low_memory=False)
    d["Date"] = pd.to_datetime(d["Date"], errors="coerce")
    d["High"] = pd.to_numeric(d["High"], errors="coerce")
    d["Low"] = pd.to_numeric(d["Low"], errors="coerce")
    d = d.dropna().sort_values("Date").reset_index(drop=True)
    d["yh"] = d["High"].shift(1)
    d["yl"] = d["Low"].shift(1)
    d = d.dropna(subset=["yh", "yl"]).copy()
    d["dkey"] = d["Date"].dt.normalize()

    m = pd.read_csv(M15_PATH, usecols=["Datetime", "High", "Low"], low_memory=False)
    m["Datetime"] = pd.to_datetime(m["Datetime"], errors="coerce")
    m["High"] = pd.to_numeric(m["High"], errors="coerce")
    m["Low"] = pd.to_numeric(m["Low"], errors="coerce")
    m = m.dropna().sort_values("Datetime").reset_index(drop=True)
    m["dkey"] = m["Datetime"].dt.normalize()
    return d, m


def first_touch_index(day_df: pd.DataFrame, price: float) -> int | None:
    for i, (hi, lo) in enumerate(zip(day_df["High"].to_numpy(float), day_df["Low"].to_numpy(float))):
        if lo <= price <= hi:
            return i
    return None


def scan_leg(highs, lows, start_i: int, tp: float, sl: float):
    # SELL leg: TP if Low <= tp, SL if High >= sl
    for j in range(start_i, len(highs)):
        hi = highs[j]
        lo = lows[j]
        hit_sl = hi >= sl
        hit_tp = lo <= tp
        if hit_sl and hit_tp:
            return "SL", j  # conservative
        if hit_sl:
            return "SL", j
        if hit_tp:
            return "TP", j
    return "OPEN", None


def main():
    daily, m15 = load_data()

    highs = m15["High"].to_numpy(float)
    lows = m15["Low"].to_numpy(float)

    # Global index map to jump from day-local index to global index.
    day_first = m15.groupby("dkey").head(1).reset_index()[["dkey", "index"]]
    day_to_first_idx = {r["dkey"]: int(r["index"]) for _, r in day_first.iterrows()}

    leg1_tp = 0
    leg1_sl = 0
    leg1_open = 0
    leg2_tp = 0
    leg2_sl = 0
    leg2_open = 0
    leg2_started = 0
    days_with_entry = 0

    for _, r in daily.iterrows():
        dkey = r["dkey"]
        if dkey not in day_to_first_idx:
            continue

        yh = float(r["yh"])
        yl = float(r["yl"])
        sl1 = yh + SL1_ABOVE_YH_PIPS * PIP
        sl2 = yh + SL2_ABOVE_YH_PIPS * PIP

        day_df = m15[m15["dkey"] == dkey]
        t0_local = first_touch_index(day_df, yh)
        if t0_local is None:
            continue
        days_with_entry += 1

        # Leg1 opens at YH touch and activates next bar.
        g0 = day_to_first_idx[dkey] + t0_local + 1
        if g0 >= len(highs):
            leg1_open += 1
            continue

        res1, idx1 = scan_leg(highs, lows, g0, tp=yl, sl=sl1)
        if res1 == "TP":
            leg1_tp += 1
        elif res1 == "SL":
            leg1_sl += 1
        else:
            leg1_open += 1

        # Leg2 only after Leg1 SL hit.
        if res1 == "SL":
            leg2_started += 1
            g1 = (idx1 + 1) if idx1 is not None else len(highs)
            if g1 >= len(highs):
                leg2_open += 1
            else:
                res2, _ = scan_leg(highs, lows, g1, tp=yh, sl=sl2)
                if res2 == "TP":
                    leg2_tp += 1
                elif res2 == "SL":
                    leg2_sl += 1
                else:
                    leg2_open += 1

    out = {
        "pair": "EURJPY",
        "rule": "SELL at YH touch; TP=YL; SL=YH+300. If SL hit -> new SELL at YH+300 with TP=YH and SL=YH+800.",
        "entry_fill": "real touch at YH",
        "exit_policy": "no forced close; positions may remain open to dataset end",
        "days_with_leg1_entry": days_with_entry,
        "leg1": {
            "tp_hits": leg1_tp,
            "sl_hits": leg1_sl,
            "still_open_at_dataset_end": leg1_open,
        },
        "leg2": {
            "started_after_leg1_sl": leg2_started,
            "tp_hits": leg2_tp,
            "sl_hits": leg2_sl,
            "still_open_at_dataset_end": leg2_open,
        },
    }

    out_json = OUT_DIR / "sell_chain_yh_300_800_hit_counts_eurjpy.json"
    out_json.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=2))
    print(f"saved={out_json}")


if __name__ == "__main__":
    main()

