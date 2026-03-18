from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

ROOT = Path(r"C:\Users\temen\5 joulu\TrailingStopLoss pivot peak trading 6 march 2026")
M15_PATH = ROOT / "forex raw data" / "Dukascopy" / "weekly_monthly_analysis" / "2003_20260306_15m" / "eurjpy" / "derived_15m" / "eurjpy_15m_20030804_20260306.csv"
DAILY_PATH = ROOT / "pair_workspaces" / "eurjpy_2003_20260306_15m_v1" / "eurjpy_daily_ohlcv_ver3.csv"
OUT_DIR = Path(__file__).parent

PIP = 0.01
SPREAD_PIPS = 1.0


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


def scan_sell(highs, lows, start_i: int, tp: float, sl: float):
    for j in range(start_i, len(highs)):
        hi = highs[j]
        lo = lows[j]
        hit_sl = hi >= sl
        hit_tp = lo <= tp
        if hit_sl and hit_tp:
            return "SL", j
        if hit_sl:
            return "SL", j
        if hit_tp:
            return "TP", j
    return "OPEN", None


def scan_buy(highs, lows, start_i: int, tp: float, sl: float):
    for j in range(start_i, len(highs)):
        hi = highs[j]
        lo = lows[j]
        hit_sl = lo <= sl
        hit_tp = hi >= tp
        if hit_sl and hit_tp:
            return "SL", j
        if hit_sl:
            return "SL", j
        if hit_tp:
            return "TP", j
    return "OPEN", None


def main():
    daily, m15 = load_data()
    day_first_idx = m15.groupby("dkey").head(1).reset_index()[["dkey", "index"]]
    day_to_first_idx = {r["dkey"]: int(r["index"]) for _, r in day_first_idx.iterrows()}

    highs = m15["High"].to_numpy(float)
    lows = m15["Low"].to_numpy(float)

    leg1_tp = leg1_sl = leg1_open = 0
    leg2_tp = leg2_sl = leg2_open = 0
    leg2_started = 0
    days_with_leg1_entry = 0

    for _, r in daily.iterrows():
        dkey = r["dkey"]
        if dkey not in day_to_first_idx:
            continue
        yh = float(r["yh"])
        yl = float(r["yl"])
        day_df = m15[m15["dkey"] == dkey]
        t0_local = first_touch_index(day_df, yh)
        if t0_local is None:
            continue
        days_with_leg1_entry += 1

        # Leg1: SELL @YH, TP=YL, SL=YH+300
        sl1 = yh + 300 * PIP
        g0 = day_to_first_idx[dkey] + t0_local + 1
        if g0 >= len(highs):
            leg1_open += 1
            continue
        res1, idx1 = scan_sell(highs, lows, g0, tp=yl, sl=sl1)
        if res1 == "TP":
            leg1_tp += 1
        elif res1 == "SL":
            leg1_sl += 1
        else:
            leg1_open += 1

        # Flipped leg2 (other way around):
        # BUY @YH+300, TP=YH+800, SL=YH
        if res1 == "SL":
            leg2_started += 1
            entry2 = yh + 300 * PIP
            tp2 = yh + 800 * PIP
            sl2 = yh
            g1 = (idx1 + 1) if idx1 is not None else len(highs)
            if g1 >= len(highs):
                leg2_open += 1
            else:
                res2, _ = scan_buy(highs, lows, g1, tp=tp2, sl=sl2)
                if res2 == "TP":
                    leg2_tp += 1
                elif res2 == "SL":
                    leg2_sl += 1
                else:
                    leg2_open += 1

    # Expectancy for leg2 flipped
    # BUY TP move: +500 pips minus spread
    # BUY SL move: -300 pips minus spread
    win_pips = 500 - SPREAD_PIPS
    loss_pips = -300 - SPREAD_PIPS
    closed2 = leg2_tp + leg2_sl
    wr2 = (leg2_tp / closed2) if closed2 > 0 else 0.0
    be_wr2 = abs(loss_pips) / (win_pips + abs(loss_pips))
    exp2 = wr2 * win_pips + (1.0 - wr2) * loss_pips if closed2 > 0 else None

    out = {
        "pair": "EURJPY",
        "rule": "Leg1 SELL@YH TP=YL SL=YH+300; after Leg1 SL, flipped Leg2 BUY@YH+300 TP=YH+800 SL=YH",
        "entry_fill": "real touch",
        "exit_policy": "no forced close",
        "leg1": {
            "tp_hits": leg1_tp,
            "sl_hits": leg1_sl,
            "still_open_at_dataset_end": leg1_open,
        },
        "leg2_flipped_buy": {
            "started_after_leg1_sl": leg2_started,
            "tp_hits": leg2_tp,
            "sl_hits": leg2_sl,
            "still_open_at_dataset_end": leg2_open,
            "win_rate_closed_only": round(wr2, 6),
            "break_even_win_rate": round(be_wr2, 6),
            "expectancy_pips_per_closed_trade": round(exp2, 4) if exp2 is not None else None,
        },
    }

    out_json = OUT_DIR / "sell_chain_leg2_flipped_buy_counts_eurjpy.json"
    out_json.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=2))
    print(f"saved={out_json}")


if __name__ == "__main__":
    main()

