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

PAIR        = "eurjpy"
PIP         = 0.01
SPREAD_PIPS = 1.0
ATR_PERIOD  = 14
SL_ATR      = 1.0 / 1.27          # LOCKED – never changes
MAX_HOLD_BARS = 96                 # 96 x 15m = 1 full day

N_BUCKETS   = 40                   # 2.5% steps  -> covers 0-100% of PP..S1
BUCKET_STEP = 0.025

TP_SWEEP = [1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8]

def _bucket_name(idx: int) -> str:
    lo = round(idx * 2.5, 1)
    hi = round(lo + 2.5, 1)
    return f"{lo}to{hi}"

def atr_wilder(h, l, c, period=14):
    n = len(h)
    tr = np.empty(n, dtype=float)
    tr[0] = h[0] - l[0]
    for i in range(1, n):
        tr[i] = max(h[i] - l[i], abs(h[i] - c[i-1]), abs(l[i] - c[i-1]))
    atr = np.full(n, np.nan)
    if n >= period:
        atr[period-1] = np.mean(tr[:period])
        k = 1.0 / period
        for i in range(period, n):
            atr[i] = atr[i-1] * (1-k) + tr[i] * k
    return atr

def load_data():
    d = pd.read_csv(DAILY_PATH, usecols=["Date","High","Low","Close"], low_memory=False)
    d["Date"] = pd.to_datetime(d["Date"], errors="coerce")
    for c in ["High","Low","Close"]:
        d[c] = pd.to_numeric(d[c], errors="coerce")
    d = d.dropna().sort_values("Date").reset_index(drop=True)

    atr = atr_wilder(d["High"].to_numpy(float), d["Low"].to_numpy(float), d["Close"].to_numpy(float))
    d["atr_prev"] = pd.Series(atr).shift(1)
    d["yh"] = d["High"].shift(1)
    d["yl"] = d["Low"].shift(1)
    d["yc"] = d["Close"].shift(1)
    d["pp"] = (d["yh"] + d["yl"] + d["yc"]) / 3.0
    d["s1"] = 2.0 * d["pp"] - d["yh"]
    d = d.dropna(subset=["atr_prev","pp","s1"]).copy()
    d["dkey"] = d["Date"].dt.normalize()

    m = pd.read_csv(M15_PATH, usecols=["Datetime","Open","High","Low","Close"], low_memory=False)
    m["Datetime"] = pd.to_datetime(m["Datetime"], errors="coerce")
    for c in ["Open","High","Low","Close"]:
        m[c] = pd.to_numeric(m[c], errors="coerce")
    m = m.dropna().sort_values("Datetime").reset_index(drop=True)
    m["dkey"] = m["Datetime"].dt.normalize()

    full_days = m.groupby("dkey").size()
    full_days = set(full_days[full_days == 96].index)
    m = m[m["dkey"].isin(full_days)].copy()

    day_bars = {}
    for dkey, g in m.groupby("dkey", sort=True):
        day_bars[dkey] = {
            "open":   float(g["Open"].iloc[0]),
            "h":      g["High"].to_numpy(float),
            "l":      g["Low"].to_numpy(float),
            "last_c": float(g["Close"].iloc[-1]),
        }

    day_info = {}
    for _, r in d.iterrows():
        k = r["dkey"]
        if k in day_bars:
            day_info[k] = {
                "pp":       float(r["pp"]),
                "s1":       float(r["s1"]),
                "atr_prev": float(r["atr_prev"]),
            }

    days = sorted(day_info.keys())
    return days, day_info, day_bars

def simulate_one(entry, tp, sl, h_arr, l_arr, last_c, n_bars):
    exit_px     = last_c
    exit_reason = "TIME_EXIT"
    for i in range(n_bars):
        hit_sl = l_arr[i] <= sl
        hit_tp = h_arr[i] >= tp
        if hit_sl and hit_tp:
            return sl, "BOTH_SL"
        if hit_sl:
            return sl, "SL"
        if hit_tp:
            return tp, "TP"
    return exit_px, exit_reason

def main():
    days, day_info, day_bars = load_data()
    print(f"Processing {len(days)} days | buckets=2.5% (n={{N_BUCKETS}}) | SL_ATR={{round(SL_ATR,4)}}")
    print(f"TP sweep: {{TP_SWEEP}}\n")

    # Pre-classify every day into a bucket + cache entry/atr
    day_cache = []
    skipped = {"above_pp": 0, "below_s1": 0, "zero_range": 0}

    for d in days:
        info  = day_info[d]
        bars  = day_bars[d]
        pp    = info["pp"]
        s1    = info["s1"]
        atr   = info["atr_prev"]
        entry = bars["open"]

        rng = pp - s1
        if rng <= 0.0:
            skipped["zero_range"] += 1
            continue
        norm = (pp - entry) / rng
        if norm < 0.0:
            skipped["above_pp"] += 1
            continue
        if norm > 1.0:
            skipped["below_s1"] += 1
            continue

        bucket_idx = int(norm / BUCKET_STEP)
        if bucket_idx >= N_BUCKETS:
            bucket_idx = N_BUCKETS - 1

        n_bars = min(len(bars["h"]), MAX_HOLD_BARS)
        day_cache.append({
            "date":       str(d.date()),
            "bucket_idx": bucket_idx,
            "entry":      entry,
            "atr":        atr,
            "h_arr":      bars["h"],
            "l_arr":      bars["l"],
            "last_c":     bars["last_c"],
            "n_bars":     n_bars,
        })

    print(f"Skipped: above_pp={{skipped['above_pp']}}  below_s1={{skipped['below_s1']}}  zero_range={{skipped['zero_range']}}")
    print(f"Tradeable days: {{len(day_cache)}}\n")

    cal_days = len(days)

    # Grid: tp_val -> bucket_idx -> list of net_pips
    grid = {tp: {i: [] for i in range(N_BUCKETS)} for tp in TP_SWEEP}

    for rec in day_cache:
        entry     = rec["entry"]
        atr       = rec["atr"]
        sl        = entry - SL_ATR * atr
        bidx      = rec["bucket_idx"]
        h_arr     = rec["h_arr"]
        l_arr     = rec["l_arr"]
        last_c    = rec["last_c"]
        n_bars    = rec["n_bars"]

        for tp_val in TP_SWEEP:
            tp = entry + tp_val * atr
            exit_px, _ = simulate_one(entry, tp, sl, h_arr, l_arr, last_c, n_bars)
            net = ((exit_px - entry) / PIP) - SPREAD_PIPS
            grid[tp_val][bidx].append(net)

    # ── Per-TP JSON output ────────────────────────────────────────────────────
    for tp_val in TP_SWEEP:
        tag     = f"tp{{str(tp_val).replace('.', 'p')}}"
        summary = {}
        for idx in range(N_BUCKETS):
            pips_list = grid[tp_val][idx]
            if not pips_list:
                continue
            name = _bucket_name(idx)
            n    = len(pips_list)
            tot  = sum(pips_list)
            wins = sum(1 for x in pips_list if x > 0)
            summary[f"bucket_{{name}}"] = {
                "bucket":      name,
                "n_trades":    n,
                "total_pips":  round(tot, 4),
                "avg_pips":    round(tot / n, 4),
                "win_rate":    round(wins / n, 4),
                "cal_ppd":     round(tot / max(cal_days, 1), 6),
            }
        summary["_meta"] = {
            "pair": PAIR.upper(),
            "period": [str(days[0].date()), str(days[-1].date())],
            "tp_atr": tp_val, "sl_atr": round(SL_ATR, 6),
            "spread_pips": SPREAD_PIPS, "bucket_step_pct": 2.5,
        }
        jpath = OUT_DIR / f"sweep_2p5pct_{{tag}}_summary.json"
        jpath.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"Saved {{jpath.name}}")

    # ── Combined comparison grid (total_pips) ─────────────────────────────────
    print("\n=== TOTAL PIPS GRID  (bucket rows x TP cols) ===")
    header = f"{{'Bucket':>10}}" + "".join(f"  TP{{tp:>4}}" for tp in TP_SWEEP)
    print(header)
    print("-" * len(header))

    for idx in range(N_BUCKETS):
        # Only print rows that have trades in at least one TP column
        if all(len(grid[tp][idx]) == 0 for tp in TP_SWEEP):
            continue
        name = _bucket_name(idx)
        row  = f"{{name:>10}}"
        for tp_val in TP_SWEEP:
            pips_list = grid[tp_val][idx]
            if pips_list:
                row += f"  {{round(sum(pips_list),1):>6}}"
            else:
                row += f"  {{'---':>6}}"
        print(row)

    print("\nDone.")


if __name__ == "__main__":
    main()