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
MAX_HOLD_HOURS = 432

# Requested: TP uses 1.3 * normalized range.
TP_MULT = 1.3
# Assumption: protective SL uses 1.0 * normalized range.
SL_MULT = 1.0


def load_data():
    d = pd.read_csv(DAILY_PATH, usecols=["Date", "High", "Low", "Close"], low_memory=False)
    d["Date"] = pd.to_datetime(d["Date"], errors="coerce")
    for c in ["High", "Low", "Close"]:
        d[c] = pd.to_numeric(d[c], errors="coerce")
    d = d.dropna().sort_values("Date").reset_index(drop=True)

    # Previous-day pivot ladder for today's trade decision.
    d["yh"] = d["High"].shift(1)
    d["yl"] = d["Low"].shift(1)
    d["yc"] = d["Close"].shift(1)
    d["pp"] = (d["yh"] + d["yl"] + d["yc"]) / 3.0
    d["r1"] = 2.0 * d["pp"] - d["yl"]
    d["s1"] = 2.0 * d["pp"] - d["yh"]
    d = d.dropna(subset=["pp", "r1", "s1"]).copy()
    d["dkey"] = d["Date"].dt.normalize()

    m = pd.read_csv(M15_PATH, usecols=["Datetime", "Open", "High", "Low", "Close"], low_memory=False)
    m["Datetime"] = pd.to_datetime(m["Datetime"], errors="coerce")
    for c in ["Open", "High", "Low", "Close"]:
        m[c] = pd.to_numeric(m[c], errors="coerce")
    m = m.dropna().sort_values("Datetime").reset_index(drop=True)
    m["dkey"] = m["Datetime"].dt.normalize()

    # Day open index map.
    day_first = m.groupby("dkey").head(1).reset_index()[["dkey", "index"]]
    day_to_i0 = {r["dkey"]: int(r["index"]) for _, r in day_first.iterrows()}

    info = {}
    for _, r in d.iterrows():
        k = r["dkey"]
        if k in day_to_i0:
            info[k] = {
                "pp": float(r["pp"]),
                "r1": float(r["r1"]),
                "s1": float(r["s1"]),
                "i0": day_to_i0[k],
            }
    return m, info


def main():
    m, info = load_data()
    times = m["Datetime"].to_numpy()
    opens = m["Open"].to_numpy(float)
    highs = m["High"].to_numpy(float)
    lows = m["Low"].to_numpy(float)
    closes = m["Close"].to_numpy(float)

    total_pips = 0.0
    wins = 0
    losses = 0
    time_exits = 0
    buys = 0
    sells = 0
    skipped_equal = 0
    skipped_bad_unit = 0

    max_bars = MAX_HOLD_HOURS * 4
    days = sorted(info.keys())

    for d in days:
        lv = info[d]
        i0 = lv["i0"]
        entry = opens[i0]
        pp = lv["pp"]

        if entry < pp:
            side = "BUY"
            buys += 1
            unit = pp - lv["s1"]  # normalized 100% for buy side
        elif entry > pp:
            side = "SELL"
            sells += 1
            unit = lv["r1"] - pp  # normalized 100% for sell side
        else:
            skipped_equal += 1
            continue

        if unit <= 0:
            skipped_bad_unit += 1
            continue

        tp_dist = TP_MULT * unit
        sl_dist = SL_MULT * unit

        if side == "BUY":
            tp = entry + tp_dist
            sl = entry - sl_dist
        else:
            tp = entry - tp_dist
            sl = entry + sl_dist

        end_i = min(i0 + max_bars, len(highs) - 1)
        done = False
        for j in range(i0 + 1, end_i + 1):
            hi = highs[j]
            lo = lows[j]
            if side == "BUY":
                hit_sl = lo <= sl
                hit_tp = hi >= tp
            else:
                hit_sl = hi >= sl
                hit_tp = lo <= tp

            if hit_sl and hit_tp:
                px = sl
                losses += 1
                done = True
            elif hit_sl:
                px = sl
                losses += 1
                done = True
            elif hit_tp:
                px = tp
                wins += 1
                done = True

            if done:
                if side == "BUY":
                    pips = ((px - entry) / PIP) - SPREAD_PIPS
                else:
                    pips = ((entry - px) / PIP) - SPREAD_PIPS
                total_pips += pips
                break

        if not done:
            px = closes[end_i]
            time_exits += 1
            if side == "BUY":
                pips = ((px - entry) / PIP) - SPREAD_PIPS
            else:
                pips = ((entry - px) / PIP) - SPREAD_PIPS
            total_pips += pips

    cal_days = len(days)
    out = {
        "pair": "EURJPY",
        "period": [str(days[0].date()), str(days[-1].date())],
        "entry": "day open",
        "direction_rule": "open<PP buy, open>PP sell",
        "normalization": "buy unit=PP-S1 (100%), sell unit=R1-PP (100%)",
        "tp_mult": TP_MULT,
        "sl_mult": SL_MULT,
        "max_hold_hours": MAX_HOLD_HOURS,
        "spread_pips": SPREAD_PIPS,
        "days": cal_days,
        "buys": buys,
        "sells": sells,
        "skipped_equal_open_pp": skipped_equal,
        "skipped_bad_unit": skipped_bad_unit,
        "wins": wins,
        "losses": losses,
        "time_exits": time_exits,
        "win_rate_excl_time_exit": round(wins / max(wins + losses, 1), 6),
        "total_pips": round(total_pips, 4),
        "cal_ppd": round(total_pips / max(cal_days, 1), 4),
    }

    out_json = OUT_DIR / "pivot_norm_r1s1_tp13_432h_eurjpy_full_days.json"
    out_json.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=2))
    print(f"saved={out_json}")


if __name__ == "__main__":
    main()

