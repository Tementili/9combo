from __future__ import annotations

import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(r"C:\Users\temen\5 joulu\TrailingStopLoss pivot peak trading 6 march 2026")
M15_PATH = ROOT / "forex raw data" / "Dukascopy" / "weekly_monthly_analysis" / "2003_20260306_15m" / "eurjpy" / "derived_15m" / "eurjpy_15m_20030804_20260306.csv"
DAILY_PATH = ROOT / "pair_workspaces" / "eurjpy_2003_20260306_15m_v1" / "eurjpy_daily_ohlcv_ver3.csv"
OUT_DIR = Path(__file__).parent

PAIR = "eurjpy"
PIP = 0.01
SPREAD_PIPS = 1.0

# Normalized where |PP->R2| or |PP->S2| = 1.00
ENTRY_PCT_GRID = [0.15, 0.30, 0.45, 0.60, 0.75, 0.90]
TP_PCT_GRID = [0.20, 0.30, 0.40, 0.50, 0.65, 0.80, 1.00]
SL_PCT_GRID = [0.20, 0.30, 0.40, 0.50, 0.65, 0.80, 1.00]


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

    m = pd.read_csv(M15_PATH, usecols=["Datetime", "Open", "High", "Low", "Close"], low_memory=False)
    m["Datetime"] = pd.to_datetime(m["Datetime"], errors="coerce")
    for c in ["Open", "High", "Low", "Close"]:
        m[c] = pd.to_numeric(m[c], errors="coerce")
    m = m.dropna().sort_values("Datetime").reset_index(drop=True)
    m["dkey"] = m["Datetime"].dt.normalize()

    # Full 96-bar days only.
    full_days = m.groupby("dkey").size()
    full_days = set(full_days[full_days == 96].index)
    m = m[m["dkey"].isin(full_days)].copy()

    bars = {}
    for dkey, g in m.groupby("dkey", sort=True):
        bars[dkey] = {
            "open": float(g["Open"].iloc[0]),
            "h": g["High"].to_numpy(float),
            "l": g["Low"].to_numpy(float),
            "last_c": float(g["Close"].iloc[-1]),
        }

    info = {}
    for _, r in d.iterrows():
        k = r["dkey"]
        if k in bars:
            pp = float(r["pp"])
            r2 = float(r["r2"])
            s2 = float(r["s2"])
            unit_buy = pp - s2
            unit_sell = r2 - pp
            if unit_buy > 0 and unit_sell > 0:
                info[k] = {"pp": pp, "unit_buy": unit_buy, "unit_sell": unit_sell}

    days = sorted(info.keys())
    return days, info, bars


def simulate_config(days, info, bars, entries_pct, tp_pct, sl_pct):
    total_pips = 0.0
    closed_legs = 0
    filled_legs = 0
    days_traded = 0
    wins = 0
    losses = 0
    time_exits = 0

    for d in days:
        lv = info[d]
        b = bars[d]
        o = b["open"]
        pp = lv["pp"]

        if o < pp:
            side = "BUY"
            unit = lv["unit_buy"]
            sign = 1.0
            entry_prices = [pp - p * unit for p in entries_pct]
        elif o > pp:
            side = "SELL"
            unit = lv["unit_sell"]
            sign = -1.0
            entry_prices = [pp + p * unit for p in entries_pct]
        else:
            continue

        days_traded += 1
        h = b["h"]
        l = b["l"]
        last_c = b["last_c"]

        # each leg: [filled, active_from_bar, entry, tp, sl, closed]
        legs = []
        for ep in entry_prices:
            tp = ep + sign * tp_pct * unit
            sl = ep - sign * sl_pct * unit
            legs.append([False, -1, ep, tp, sl, False])

        for i in range(len(h)):
            hi = h[i]
            lo = l[i]

            # Fill pending entries by touch; activate next bar.
            for leg in legs:
                if leg[0]:
                    continue
                ep = leg[2]
                if lo <= ep <= hi:
                    leg[0] = True
                    leg[1] = i + 1
                    filled_legs += 1

            # Check exits for filled+active legs.
            for leg in legs:
                if (not leg[0]) or leg[5] or i < leg[1]:
                    continue
                entry, tp, sl = leg[2], leg[3], leg[4]
                if side == "BUY":
                    hit_sl = lo <= sl
                    hit_tp = hi >= tp
                else:
                    hit_sl = hi >= sl
                    hit_tp = lo <= tp

                if hit_sl and hit_tp:
                    px = sl
                    losses += 1
                elif hit_sl:
                    px = sl
                    losses += 1
                elif hit_tp:
                    px = tp
                    wins += 1
                else:
                    continue

                leg[5] = True
                if side == "BUY":
                    pips = ((px - entry) / PIP) - SPREAD_PIPS
                else:
                    pips = ((entry - px) / PIP) - SPREAD_PIPS
                total_pips += pips
                closed_legs += 1

        # EOD close for filled but still open legs.
        for leg in legs:
            if leg[0] and (not leg[5]):
                entry = leg[2]
                leg[5] = True
                time_exits += 1
                if side == "BUY":
                    pips = ((last_c - entry) / PIP) - SPREAD_PIPS
                else:
                    pips = ((entry - last_c) / PIP) - SPREAD_PIPS
                total_pips += pips
                closed_legs += 1

    cal_days = len(days)
    return {
        "entry_leg1_pct": entries_pct[0],
        "entry_leg2_pct": entries_pct[1],
        "entry_leg3_pct": entries_pct[2],
        "tp_pct_of_unit": tp_pct,
        "sl_pct_of_unit": sl_pct,
        "days_traded": days_traded,
        "filled_legs": filled_legs,
        "closed_legs": closed_legs,
        "wins": wins,
        "losses": losses,
        "time_exits": time_exits,
        "win_rate_excl_time_exit": round(wins / max(wins + losses, 1), 6),
        "total_pips": round(total_pips, 4),
        "cal_ppd": round(total_pips / max(cal_days, 1), 4),
    }


def main():
    days, info, bars = load_data()
    entry_triplets = list(itertools.combinations(ENTRY_PCT_GRID, 3))
    rows = []

    total_cfg = len(entry_triplets) * len(TP_PCT_GRID) * len(SL_PCT_GRID)
    done = 0
    for e in entry_triplets:
        for tp in TP_PCT_GRID:
            for sl in SL_PCT_GRID:
                r = simulate_config(days, info, bars, e, tp, sl)
                rows.append(r)
                done += 1
                if done % 100 == 0:
                    print(f"{done}/{total_cfg} cfg ... best_so_far={max(x['cal_ppd'] for x in rows):+.4f}")

    df = pd.DataFrame(rows).sort_values("cal_ppd", ascending=False).reset_index(drop=True)
    best = df.iloc[0].to_dict()
    out = {
        "pair": PAIR.upper(),
        "period": [str(days[0].date()), str(days[-1].date())],
        "normalization": "|PP->R2| or |PP->S2| = 1.00",
        "direction_rule": "open<PP buy side, open>PP sell side",
        "entry_rule": "3 ladder legs at percentages from PP toward S2/R2",
        "cost_pips_per_leg": SPREAD_PIPS,
        "entry_pct_grid": ENTRY_PCT_GRID,
        "tp_pct_grid": TP_PCT_GRID,
        "sl_pct_grid": SL_PCT_GRID,
        "best": best,
        "top20": df.head(20).to_dict(orient="records"),
    }

    out_json = OUT_DIR / "three_leg_normalized_r2s2_optimize_eurjpy_full_days.json"
    out_csv = OUT_DIR / "three_leg_normalized_r2s2_optimize_eurjpy_full_days.csv"
    out_json.write_text(json.dumps(out, indent=2), encoding="utf-8")
    df.to_csv(out_csv, index=False)

    print("\nBEST:")
    print(best)
    print(f"saved={out_json}")
    print(f"saved={out_csv}")


if __name__ == "__main__":
    main()

