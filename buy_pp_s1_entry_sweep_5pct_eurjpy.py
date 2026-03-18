# buy_pp_s1_entry_sweep_5pct_eurjpy.py
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
TP_ATR = 1.0
SL_ATR = 1.0 / 1.27
MAX_HOLD_BARS = 96  # 96 x 15m bars = 1 full day

N_BUCKETS = 20
BUCKET_STEP = 0.05  # 5% steps


def _bucket_name(idx: int) -> str:
    lo = idx * 5
    hi = lo + 5
    return f"{lo}to{hi}"


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
    # ── Daily data ─────────────────────────────────────────────────────────────
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
    d["s1"] = 2.0 * d["pp"] - d["yh"]
    d = d.dropna(subset=["atr_prev", "pp", "s1"]).copy()
    d["dkey"] = d["Date"].dt.normalize()

    # ── 15m data ───────────────────────────────────────────────────────────────
    m = pd.read_csv(M15_PATH, usecols=["Datetime", "Open", "High", "Low", "Close"], low_memory=False)
    m["Datetime"] = pd.to_datetime(m["Datetime"], errors="coerce")
    for c in ["Open", "High", "Low", "Close"]:
        m[c] = pd.to_numeric(m[c], errors="coerce")
    m = m.dropna().sort_values("Datetime").reset_index(drop=True)
    m["dkey"] = m["Datetime"].dt.normalize()

    # Keep full 96-bar days only
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
            day_info[k] = {
                "pp": float(r["pp"]),
                "s1": float(r["s1"]),
                "atr_prev": float(r["atr_prev"]),
            }

    days = sorted(day_info.keys())
    return days, day_info, day_bars


def main():
    days, day_info, day_bars = load_data()
    print(f"Processing {len(days)} days...")

    # bucket_idx -> list of ledger rows
    buckets: dict[int, list] = {i: [] for i in range(N_BUCKETS)}

    skipped_above_pp = 0
    skipped_below_s1 = 0
    skipped_zero_range = 0

    for d in days:
        info = day_info[d]
        bars = day_bars[d]

        pp = info["pp"]
        s1 = info["s1"]
        atr = info["atr_prev"]
        entry = bars["open"]

        pp_s1_range = pp - s1
        if pp_s1_range <= 0.0:
            skipped_zero_range += 1
            continue

        # Normalised position: 0 = at PP, 1 = at S1
        norm = (pp - entry) / pp_s1_range

        if norm < 0.0:
            skipped_above_pp += 1
            continue
        if norm > 1.0:
            skipped_below_s1 += 1
            continue

        bucket_idx = int(norm / BUCKET_STEP)
        if bucket_idx >= N_BUCKETS:
            bucket_idx = N_BUCKETS - 1

        tp = entry + TP_ATR * atr
        sl = entry - SL_ATR * atr

        # Simulate through 15m bars (up to MAX_HOLD_BARS)
        h_arr = bars["h"]
        l_arr = bars["l"]
        n_bars = min(len(h_arr), MAX_HOLD_BARS)

        exit_px = bars["last_c"]
        exit_reason = "TIME_EXIT"
        done = False

        for i in range(n_bars):
            hit_sl = l_arr[i] <= sl
            hit_tp = h_arr[i] >= tp

            if hit_sl and hit_tp:
                exit_px = sl
                exit_reason = "BOTH_SL"
                done = True
            elif hit_sl:
                exit_px = sl
                exit_reason = "SL"
                done = True
            elif hit_tp:
                exit_px = tp
                exit_reason = "TP"
                done = True

            if done:
                break

        net_pips = ((exit_px - entry) / PIP) - SPREAD_PIPS

        buckets[bucket_idx].append({
            "date": str(d.date()),
            "entry_px": round(entry, 6),
            "exit_px": round(exit_px, 6),
            "net_pips": round(net_pips, 4),
            "exit_reason": exit_reason,
            "bucket": _bucket_name(bucket_idx),
        })

    print(f"Skipped above PP: {skipped_above_pp}, below S1: {skipped_below_s1}, zero range: {skipped_zero_range}")

    # ── Write per-bucket ledger CSVs ───────────────────────────────────────────
    cal_days = len(days)
    summary = {}

    for idx in range(N_BUCKETS):
        name = _bucket_name(idx)
        rows = buckets[idx]
        if not rows:
            continue

        df = pd.DataFrame(rows)
        csv_path = OUT_DIR / f"ledger_bucket_{name}.csv"
        df.to_csv(csv_path, index=False)
        print(f"  bucket {name}: {len(rows)} trades  -> {csv_path.name}")

        wins = sum(1 for r in rows if r["net_pips"] > 0)
        losses = sum(1 for r in rows if r["net_pips"] <= 0)
        time_exits = sum(1 for r in rows if r["exit_reason"] == "TIME_EXIT")
        total_pips = sum(r["net_pips"] for r in rows)
        n = len(rows)

        summary[f"bucket_{name}"] = {
            "bucket": name,
            "n_trades": n,
            "total_pips": round(total_pips, 4),
            "avg_pips": round(total_pips / n, 4),
            "win_rate": round(wins / n, 4),
            "wins": wins,
            "losses": losses,
            "time_exits": time_exits,
            "cal_ppd": round(total_pips / max(cal_days, 1), 6),
        }

    # ── Meta ───────────────────────────────────────────────────────────────────
    summary["_meta"] = {
        "pair": PAIR.upper(),
        "period": [str(days[0].date()), str(days[-1].date())],
        "tp_atr": TP_ATR,
        "sl_atr": round(SL_ATR, 6),
        "spread_pips": SPREAD_PIPS,
        "atr_period": ATR_PERIOD,
        "max_hold_bars": MAX_HOLD_BARS,
        "n_buckets": N_BUCKETS,
        "bucket_step_pct": 5,
    }

    json_path = OUT_DIR / "sweep_5pct_summary.json"
    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(json.dumps(summary, indent=2))
    print(f"\nsaved={json_path}")


if __name__ == "__main__":
    main()