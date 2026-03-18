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
MAX_HOLD_HOURS = 720
MAX_HOLD_BARS = MAX_HOLD_HOURS * 4


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
    d = d.dropna(subset=["pp"]).copy()
    d["dkey"] = d["Date"].dt.normalize()

    m = pd.read_csv(M15_PATH, usecols=["Datetime", "Open", "High", "Low", "Close"], low_memory=False)
    m["Datetime"] = pd.to_datetime(m["Datetime"], errors="coerce")
    for c in ["Open", "High", "Low", "Close"]:
        m[c] = pd.to_numeric(m[c], errors="coerce")
    m = m.dropna().sort_values("Datetime").reset_index(drop=True)
    m["dkey"] = m["Datetime"].dt.normalize()

    full = m.groupby("dkey").size()
    full_days = set(full[full == 96].index)
    m = m[m["dkey"].isin(full_days)].copy().reset_index(drop=True)

    day_first = m.groupby("dkey").head(1).reset_index()[["dkey", "index"]]
    day_to_i0 = {r["dkey"]: int(r["index"]) for _, r in day_first.iterrows()}

    info = {}
    for _, r in d.iterrows():
        k = r["dkey"]
        if k in day_to_i0:
            info[k] = {"pp": float(r["pp"]), "i0": day_to_i0[k]}

    return m, info


def close_pips(side: str, entry: float, px: float) -> float:
    if side == "BUY":
        return ((px - entry) / PIP) - SPREAD_PIPS
    return ((entry - px) / PIP) - SPREAD_PIPS


def run_one(m: pd.DataFrame, info: dict):
    opens = m["Open"].to_numpy(float)
    highs = m["High"].to_numpy(float)
    lows = m["Low"].to_numpy(float)
    closes = m["Close"].to_numpy(float)

    total_pips = 0.0
    buy_pips = 0.0
    sell_pips = 0.0
    wins = 0
    losses = 0
    time_exits = 0
    buys = 0
    sells = 0
    skipped_equal = 0

    days = sorted(info.keys())
    for d in days:
        pp = info[d]["pp"]
        i0 = info[d]["i0"]
        entry = opens[i0]

        if entry < pp:
            side = "BUY"
            buys += 1
            tp = pp
            sl = entry - (pp - entry)  # equal distance opposite side
        elif entry > pp:
            side = "SELL"
            sells += 1
            tp = pp
            sl = entry + (entry - pp)  # equal distance opposite side
        else:
            skipped_equal += 1
            continue

        start_i = i0 + 1
        if start_i >= len(opens):
            continue
        end_i = min(start_i + MAX_HOLD_BARS, len(opens) - 1)
        done = False

        for j in range(start_i, end_i + 1):
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
                done = True
                losses += 1
            elif hit_sl:
                px = sl
                done = True
                losses += 1
            elif hit_tp:
                px = tp
                done = True
                wins += 1
            else:
                continue

            pips = close_pips(side, entry, px)
            total_pips += pips
            if side == "BUY":
                buy_pips += pips
            else:
                sell_pips += pips
            break

        if not done:
            px = closes[end_i]
            pips = close_pips(side, entry, px)
            total_pips += pips
            if side == "BUY":
                buy_pips += pips
            else:
                sell_pips += pips
            time_exits += 1
            if pips >= 0:
                wins += 1
            else:
                losses += 1

    cal_days = len(days)
    return {
        "pair": "EURJPY",
        "days": cal_days,
        "max_hold_hours": MAX_HOLD_HOURS,
        "entry_rule": "enter at day open",
        "direction_rule": "open<pivot BUY, open>pivot SELL",
        "tp_rule": "TP at pivot",
        "sl_rule": "SL equal distance opposite side from entry",
        "spread_pips": SPREAD_PIPS,
        "wins": wins,
        "losses": losses,
        "time_exits": time_exits,
        "win_rate": round(wins / max(wins + losses, 1), 6),
        "buys": buys,
        "sells": sells,
        "skipped_equal_open_pivot": skipped_equal,
        "buy_pips": round(buy_pips, 4),
        "sell_pips": round(sell_pips, 4),
        "total_pips": round(total_pips, 4),
        "cal_ppd": round(total_pips / max(cal_days, 1), 4),
    }


def main():
    m, info = load_data()
    out = run_one(m, info)
    out_json = OUT_DIR / "dayopen_to_pivot_equal_sl_eurjpy.json"
    out_json.write_text(json.dumps(out, indent=2), encoding="utf-8")

    print("RESULT:")
    print(json.dumps(out, indent=2))
    print(f"saved={out_json}")


if __name__ == "__main__":
    main()

