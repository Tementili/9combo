from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(r"C:\Users\temen\5 joulu\TrailingStopLoss pivot peak trading 6 march 2026")
M15_PATH = ROOT / "forex raw data" / "Dukascopy" / "weekly_monthly_analysis" / "2003_20260306_15m" / "eurjpy" / "derived_15m" / "eurjpy_15m_20030804_20260306.csv"
DAILY_PATH = ROOT / "pair_workspaces" / "eurjpy_2003_20260306_15m_v1" / "eurjpy_daily_ohlcv_ver3.csv"

PIP = 0.01
SPREAD_PIPS = 1.0
MAX_HOLD_HOURS = 432
ATR_PERIOD = 14


def atr_wilder(h: np.ndarray, l: np.ndarray, c: np.ndarray, period: int = 14) -> np.ndarray:
    n = len(h)
    tr = np.empty(n, dtype=float)
    tr[0] = h[0] - l[0]
    for i in range(1, n):
        tr[i] = max(h[i] - l[i], abs(h[i] - c[i - 1]), abs(l[i] - c[i - 1]))
    atr = np.full(n, np.nan)
    if n >= period:
        atr[period - 1] = np.mean(tr[:period])
        k = 1.0 / period
        for i in range(period, n):
            atr[i] = atr[i - 1] * (1.0 - k) + tr[i] * k
    return atr


def load():
    m = pd.read_csv(M15_PATH, usecols=["Datetime", "Open", "High", "Low", "Close"], low_memory=False)
    m["Datetime"] = pd.to_datetime(m["Datetime"], errors="coerce")
    for c in ["Open", "High", "Low", "Close"]:
        m[c] = pd.to_numeric(m[c], errors="coerce")
    m = m.dropna().sort_values("Datetime").reset_index(drop=True)
    m["dkey"] = m["Datetime"].dt.normalize()
    full = m.groupby("dkey").size()
    full_days = set(full[full == 96].index)
    m = m[m["dkey"].isin(full_days)].copy()
    m = m.reset_index(drop=True)

    # First bar index per day
    day_first_idx = m.groupby("dkey").head(1).reset_index()[["dkey", "index"]]
    day_to_i0 = {r["dkey"]: int(r["index"]) for _, r in day_first_idx.iterrows()}

    d = pd.read_csv(DAILY_PATH, usecols=["Date", "High", "Low", "Close"], low_memory=False)
    d["Date"] = pd.to_datetime(d["Date"], errors="coerce")
    for c in ["High", "Low", "Close"]:
        d[c] = pd.to_numeric(d[c], errors="coerce")
    d = d.dropna().sort_values("Date").reset_index(drop=True)
    atr = atr_wilder(d["High"].to_numpy(float), d["Low"].to_numpy(float), d["Close"].to_numpy(float), ATR_PERIOD)
    d["atr_prev"] = pd.Series(atr).shift(1)
    d["yh"] = d["High"].shift(1)
    d["yl"] = d["Low"].shift(1)
    d["yc"] = d["Close"].shift(1)
    d["pp"] = (d["yh"] + d["yl"] + d["yc"]) / 3.0
    d["dkey"] = d["Date"].dt.normalize()
    d = d.dropna(subset=["atr_prev", "pp"])

    day_info = {}
    for _, r in d.iterrows():
        k = r["dkey"]
        if k in day_to_i0:
            day_info[k] = {
                "pp": float(r["pp"]),
                "atr_prev": float(r["atr_prev"]),
                "i0": day_to_i0[k],
            }

    return m, day_info


def side_for_day(pp: float, open_px: float, mode: str, rng: np.random.Generator) -> str | None:
    if mode == "pivot_side":
        if open_px < pp:
            return "BUY"
        if open_px > pp:
            return "SELL"
        return None
    if mode == "random":
        return "BUY" if rng.random() < 0.5 else "SELL"
    raise ValueError(mode)


def run(mode: str, ratio: float, scale: float, seed: int = 0):
    m, day_info = load()
    times = m["Datetime"].to_numpy()
    opens = m["Open"].to_numpy(float)
    highs = m["High"].to_numpy(float)
    lows = m["Low"].to_numpy(float)
    closes = m["Close"].to_numpy(float)

    rng = np.random.default_rng(seed)

    tp_atr = 1.0 * scale
    sl_atr = (1.0 / ratio) * scale
    max_bars = int(MAX_HOLD_HOURS * 4)

    days = sorted(day_info.keys())
    total = 0.0
    wins = 0
    losses = 0
    time_exits = 0
    skipped = 0

    for d in days:
        info = day_info[d]
        i0 = info["i0"]
        entry = opens[i0]
        pp = info["pp"]
        atr_prev = info["atr_prev"]
        side = side_for_day(pp, entry, mode, rng)
        if side is None:
            skipped += 1
            continue

        tp_dist = tp_atr * atr_prev
        sl_dist = sl_atr * atr_prev
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
                total += pips
                break

        if not done:
            px = closes[end_i]
            time_exits += 1
            if side == "BUY":
                pips = ((px - entry) / PIP) - SPREAD_PIPS
            else:
                pips = ((entry - px) / PIP) - SPREAD_PIPS
            total += pips

    cal_days = len(days)
    return {
        "mode": mode,
        "ratio_tp_to_sl": ratio,
        "scale": scale,
        "tp_atr": tp_atr,
        "sl_atr": sl_atr,
        "max_hold_hours": MAX_HOLD_HOURS,
        "days": cal_days,
        "skipped_days": skipped,
        "wins": wins,
        "losses": losses,
        "time_exits": time_exits,
        "win_rate_excl_time_exit": round(wins / max(wins + losses, 1), 6),
        "total_pips": round(total, 4),
        "cal_ppd": round(total / max(cal_days, 1), 4),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["pivot_side", "random"], required=True)
    ap.add_argument("--ratio", type=float, required=True)
    ap.add_argument("--scale", type=float, required=True)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    out = run(args.mode, args.ratio, args.scale, args.seed)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()

