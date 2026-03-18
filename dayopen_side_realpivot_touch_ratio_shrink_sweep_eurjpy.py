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
TP_ATR_BASE = 1.0
SL_ATR_BASE = 1.0 / 1.27
SHRINK_STEP = 0.03
N_SWEEPS = 16  # 1.00 .. 0.55


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

    # Full 96-bar days only.
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


def run_one(days, day_info, day_bars, scale: float) -> dict:
    tp_atr = TP_ATR_BASE * scale
    sl_atr = SL_ATR_BASE * scale

    total_pips = 0.0
    wins = 0
    losses = 0
    time_exits = 0
    buys = 0
    sells = 0
    skipped_equal = 0
    skipped_no_touch = 0

    for d in days:
        info = day_info[d]
        bars = day_bars[d]
        day_open = bars["open"]
        pp = info["pp"]
        atr = info["atr_prev"]
        tp_dist = tp_atr * atr
        sl_dist = sl_atr * atr

        # Direction from day open vs pivot (same as previous test).
        if day_open < pp:
            side = "BUY"
            buys += 1
        elif day_open > pp:
            side = "SELL"
            sells += 1
        else:
            skipped_equal += 1
            continue

        # REAL pivot start: wait until price first touches daily pivot.
        h_arr = bars["h"]
        l_arr = bars["l"]
        touch_i = None
        for i in range(len(h_arr)):
            if l_arr[i] <= pp <= h_arr[i]:
                touch_i = i
                break
        if touch_i is None:
            skipped_no_touch += 1
            continue

        # Fill at pivot on touch, then activate next bar to avoid intrabar ambiguity.
        entry = pp
        start_i = touch_i + 1
        if start_i >= len(h_arr):
            continue

        if side == "BUY":
            tp = entry + tp_dist
            sl = entry - sl_dist
        else:
            tp = entry - tp_dist
            sl = entry + sl_dist

        done = False
        for h, l in zip(h_arr[start_i:], l_arr[start_i:]):
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
        "scale": round(scale, 4),
        "tp_atr": round(tp_atr, 6),
        "sl_atr": round(sl_atr, 6),
        "wins": wins,
        "losses": losses,
        "time_exits": time_exits,
        "buys": buys,
        "sells": sells,
        "skipped_equal_open_pp": skipped_equal,
        "skipped_no_real_pivot_touch": skipped_no_touch,
        "win_rate_excl_time_exit": round(wins / max(wins + losses, 1), 6),
        "total_pips": round(total_pips, 4),
        "cal_ppd": round(total_pips / max(cal_days, 1), 4),
    }


def main():
    days, day_info, day_bars = load_data()
    rows = []
    for i in range(N_SWEEPS):
        scale = 1.0 - i * SHRINK_STEP
        if scale <= 0:
            break
        r = run_one(days, day_info, day_bars, scale)
        rows.append(r)
        print(
            f"scale={r['scale']:.2f} tp={r['tp_atr']:.4f} sl={r['sl_atr']:.4f} "
            f"cal_ppd={r['cal_ppd']:+.4f} wins={r['wins']} losses={r['losses']} no_touch={r['skipped_no_real_pivot_touch']}"
        )

    df = pd.DataFrame(rows).sort_values("cal_ppd", ascending=False).reset_index(drop=True)
    out = {
        "pair": PAIR.upper(),
        "period": [str(days[0].date()), str(days[-1].date())],
        "rule": "direction from day open vs daily pivot; entry at FIRST REAL daily pivot touch",
        "spread_pips": SPREAD_PIPS,
        "ratio_tp_to_sl": 1.27,
        "base_tp_atr": TP_ATR_BASE,
        "base_sl_atr": SL_ATR_BASE,
        "shrink_step": SHRINK_STEP,
        "sweeps": rows,
        "best": df.iloc[0].to_dict(),
    }

    out_json = OUT_DIR / "dayopen_side_realpivot_touch_ratio_shrink_sweep_eurjpy_full_days.json"
    out_csv = OUT_DIR / "dayopen_side_realpivot_touch_ratio_shrink_sweep_eurjpy_full_days.csv"
    out_json.write_text(json.dumps(out, indent=2), encoding="utf-8")
    df.to_csv(out_csv, index=False)

    print("\nBEST:")
    print(df.iloc[0].to_dict())
    print(f"saved={out_json}")
    print(f"saved={out_csv}")


if __name__ == "__main__":
    main()

