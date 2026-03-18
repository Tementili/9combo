from __future__ import annotations

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
ATR_PERIOD = 14

# Grow TP:SL ratio from 1.27 up to 1.50.
RATIO_GRID = [1.27, 1.30, 1.35, 1.40, 1.45, 1.50]
# Shrink ATR size in 5% steps.
SCALE_GRID = [1.00, 0.95, 0.90, 0.85, 0.80, 0.75, 0.70, 0.65, 0.60, 0.55]


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


def load_data():
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
    d = d.dropna(subset=["atr_prev", "pp"]).copy()
    d["dkey"] = d["Date"].dt.normalize()

    m = pd.read_csv(M15_PATH, usecols=["Datetime", "Open", "High", "Low", "Close"], low_memory=False)
    m["Datetime"] = pd.to_datetime(m["Datetime"], errors="coerce")
    for c in ["Open", "High", "Low", "Close"]:
        m[c] = pd.to_numeric(m[c], errors="coerce")
    m = m.dropna().sort_values("Datetime").reset_index(drop=True)
    m["dkey"] = m["Datetime"].dt.normalize()

    full_days = m.groupby("dkey").size()
    full_days = set(full_days[full_days == 96].index)
    m = m[m["dkey"].isin(full_days)].copy()

    day_bars = {}
    for dkey, g in m.groupby("dkey", sort=True):
        day_bars[dkey] = {
            "open": float(g["Open"].iloc[0]),
            "h": g["High"].to_numpy(float),
            "l": g["Low"].to_numpy(float),
            "last_c": float(g["Close"].iloc[-1]),
        }

    day_info = {}
    for _, r in d.iterrows():
        k = r["dkey"]
        if k in day_bars:
            day_info[k] = {"pp": float(r["pp"]), "atr_prev": float(r["atr_prev"])}
    days = sorted(day_info.keys())
    return days, day_info, day_bars


def run_one(days, day_info, day_bars, ratio: float, scale: float) -> dict:
    tp_atr = 1.0 * scale
    sl_atr = (1.0 / ratio) * scale

    total_pips = 0.0
    wins = 0
    losses = 0
    time_exits = 0
    buys = 0
    sells = 0
    skipped_equal = 0

    for d in days:
        info = day_info[d]
        bars = day_bars[d]
        entry = bars["open"]  # Requested: start from daily open.
        pp = info["pp"]
        atr = info["atr_prev"]
        tp_dist = tp_atr * atr
        sl_dist = sl_atr * atr

        if entry < pp:
            side = "BUY"
            buys += 1
        elif entry > pp:
            side = "SELL"
            sells += 1
        else:
            skipped_equal += 1
            continue

        if side == "BUY":
            tp = entry + tp_dist
            sl = entry - sl_dist
        else:
            tp = entry - tp_dist
            sl = entry + sl_dist

        done = False
        for h, l in zip(bars["h"], bars["l"]):
            if side == "BUY":
                hit_sl = l <= sl
                hit_tp = h >= tp
            else:
                hit_sl = h >= sl
                hit_tp = l <= tp

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
            px = bars["last_c"]
            time_exits += 1
            if side == "BUY":
                pips = ((px - entry) / PIP) - SPREAD_PIPS
            else:
                pips = ((entry - px) / PIP) - SPREAD_PIPS
            total_pips += pips

    cal_days = len(days)
    return {
        "ratio_tp_to_sl": round(ratio, 4),
        "scale": round(scale, 4),
        "tp_atr": round(tp_atr, 6),
        "sl_atr": round(sl_atr, 6),
        "wins": wins,
        "losses": losses,
        "time_exits": time_exits,
        "buys": buys,
        "sells": sells,
        "skipped_equal_open_pp": skipped_equal,
        "win_rate_excl_time_exit": round(wins / max(wins + losses, 1), 6),
        "total_pips": round(total_pips, 4),
        "cal_ppd": round(total_pips / max(cal_days, 1), 4),
    }


def main():
    days, day_info, day_bars = load_data()
    rows = []

    for ratio in RATIO_GRID:
        for scale in SCALE_GRID:
            r = run_one(days, day_info, day_bars, ratio, scale)
            rows.append(r)
            print(
                f"ratio={ratio:.2f} scale={scale:.2f} tp={r['tp_atr']:.4f} sl={r['sl_atr']:.4f} "
                f"cal_ppd={r['cal_ppd']:+.4f}"
            )

    df = pd.DataFrame(rows).sort_values("cal_ppd", ascending=False).reset_index(drop=True)
    out = {
        "pair": PAIR.upper(),
        "period": [str(days[0].date()), str(days[-1].date())],
        "rule": "day open < daily pivot -> BUY, day open > daily pivot -> SELL",
        "entry": "daily open",
        "spread_pips": SPREAD_PIPS,
        "ratio_grid": RATIO_GRID,
        "scale_grid": SCALE_GRID,
        "best": df.iloc[0].to_dict(),
        "top15": df.head(15).to_dict(orient="records"),
    }

    out_json = OUT_DIR / "dayopen_pivot_side_ratio127_to150_scale5pct_sweep_eurjpy_full_days.json"
    out_csv = OUT_DIR / "dayopen_pivot_side_ratio127_to150_scale5pct_sweep_eurjpy_full_days.csv"
    out_json.write_text(json.dumps(out, indent=2), encoding="utf-8")
    df.to_csv(out_csv, index=False)

    print("\nBEST:")
    print(df.iloc[0].to_dict())
    print(f"saved={out_json}")
    print(f"saved={out_csv}")


if __name__ == "__main__":
    main()

