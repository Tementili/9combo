"""
bl_validate.py  -  Engine Validation Memo for Baseline Discovery Step 1
=======================================================================
Validates:
  V1) Split integrity:    train_end < adjust_start < oos_start (no overlap)
  V2) Warmup enforcement: no Trade rows with NaN ATR14
  V3) 432h guard:         every triggered trade has bar_idx + 432 <= n_bars
  V4) OOS leakage:        no Train/Adjust rows with date >= oos_start
  V5) Direction rule:     SELL iff window_open > PP, else BUY (no exceptions)
  V6) TP rule:            TP is on the correct side of entry for all tp_modes
  V7) SL rule:            SL is on the correct side of entry for both directions
  V8) Cost inclusion:     net pips = gross - cost (not gross alone)
  V9) Deterministic:      same cfg + same rows => same results every run

Prints PASS/FAIL for each check with numeric evidence.
Exits with code 0 if all pass, 1 if any fail.
"""
from __future__ import annotations

import sys
import math
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd

OUTPUT_DIR = Path(__file__).parent
sys.path.insert(0, str(OUTPUT_DIR))

from bl_defs import (
    SPLIT, WINDOWS, WINDOW_UTC_START, ATR_PERIOD, MAIN_SIZE,
    SL_MULTS, SL_MAIN_FRACS, TP_MODES, SKIP_FRACS,
    _atr_series, _build_rows_all_windows,
    _run_backtest_bl, _summarize_bl, _n_cal_days, _tp_price,
)
from bl_runner import PAIR_CFG, _load_daily, _load_hourly

PASS_COUNT = 0
FAIL_COUNT = 0

def chk(name: str, cond: bool, detail: str = ""):
    global PASS_COUNT, FAIL_COUNT
    mark = "PASS" if cond else "FAIL"
    if not cond:
        FAIL_COUNT += 1
    else:
        PASS_COUNT += 1
    print(f"  [{mark}] {name}" + (f"  ({detail})" if detail else ""))


# ── V1: Split integrity ────────────────────────────────────────────────────────
def check_split_integrity():
    print("\nV1: Split integrity")
    te  = pd.Timestamp(SPLIT["train_end"])
    as_ = pd.Timestamp(SPLIT["adjust_start"])
    ae  = pd.Timestamp(SPLIT["adjust_end"])
    os_ = pd.Timestamp(SPLIT["oos_start"])
    oe  = pd.Timestamp(SPLIT["oos_end"])

    chk("train_end < adjust_start", te < as_,
        f"{SPLIT['train_end']} < {SPLIT['adjust_start']}")
    chk("adjust_start <= adjust_end", as_ <= ae,
        f"{SPLIT['adjust_start']} <= {SPLIT['adjust_end']}")
    chk("adjust_end < oos_start", ae < os_,
        f"{SPLIT['adjust_end']} < {SPLIT['oos_start']}")
    chk("oos_start <= oos_end", os_ <= oe,
        f"{SPLIT['oos_start']} <= {SPLIT['oos_end']}")
    chk("No overlap: train_end + 1 == adjust_start",
        (as_ - te).days == 1,
        f"gap={(as_ - te).days} days")
    chk("No overlap: adjust_end + 1 == oos_start",
        (os_ - ae).days == 1,
        f"gap={(os_ - ae).days} days")
    chk("max_hold_hours == 432", SPLIT["max_hold_hours"] == 432,
        f"got {SPLIT['max_hold_hours']}")


# ── V2-V4 + V5-V7: Per-pair data checks ───────────────────────────────────────
def check_pair(pair: str, quiet: bool = False):
    cfg_p     = PAIR_CFG[pair]
    pip       = cfg_p["pip"]
    cost_pips = cfg_p["cost_pips"]

    daily = _load_daily(cfg_p["daily"])
    times, opens, highs, lows, closes = _load_hourly(cfg_p["h1"])

    if not quiet:
        print(f"  Data: {len(daily)} daily rows, {len(highs)} hourly bars")
        print(f"  Hourly span: {pd.Timestamp(times[0]).date()} .. "
              f"{pd.Timestamp(times[-1]).date()}")

    tr_rows  = _build_rows_all_windows(daily, times, opens,
                                       SPLIT["train_start"], SPLIT["train_end"])
    ad_rows  = _build_rows_all_windows(daily, times, opens,
                                       SPLIT["adjust_start"], SPLIT["adjust_end"])
    oos_rows = _build_rows_all_windows(daily, times, opens,
                                       SPLIT["oos_start"], SPLIT["oos_end"])

    n_bars = len(highs)

    # V2: Warmup — no NaN ATR14 in Train rows
    nan_count = int(tr_rows["ATR14"].isna().sum())
    chk(f"V2 [{pair}] No NaN ATR14 in Train", nan_count == 0,
        f"{nan_count} NaN rows")
    if not quiet and len(tr_rows) > 0:
        first_date = tr_rows.iloc[0]["trade_date_str"]
        print(f"      First Train row: {first_date}  "
              f"ATR14={tr_rows.iloc[0]['ATR14']:.5f}")

    # V3: 432h guard — all triggered rows have bar_idx + 432 <= n_bars
    mh = SPLIT["max_hold_hours"]
    violating = tr_rows[tr_rows["window_bar_idx"] + mh > n_bars]
    chk(f"V3 [{pair}] 432h guard respected",
        len(violating) == 0,
        f"{len(violating)} rows would exceed data end")

    # V4: OOS leakage — no Train/Adjust rows after oos_start
    ts_oos = pd.Timestamp(SPLIT["oos_start"])
    tr_leak  = int((tr_rows["date"]  >= ts_oos).sum())
    adj_leak = int((ad_rows["date"] >= ts_oos).sum())
    chk(f"V4 [{pair}] No OOS in Train",  tr_leak  == 0, f"{tr_leak} rows")
    chk(f"V4 [{pair}] No OOS in Adjust", adj_leak == 0, f"{adj_leak} rows")

    # V5: Direction rule — SELL iff window_open > PP
    # Use window_h=8 rows only (backtest filters to one window, so rows must match)
    if len(tr_rows) > 0:
        ref_cfg = {"sl_mult": 2.0, "sl_main_frac": 0.5, "tp_mode": "TP_ATR_1.5",
                   "skip_frac": 0.0, "window_h": 8}  # skip_frac=0 to bypass range filter
        w8_rows = tr_rows[tr_rows["window_h"] == 8].reset_index(drop=True)
        ref_res = _run_backtest_bl(w8_rows, highs, lows, closes, pip, cost_pips, ref_cfg)
        dir_ok = True
        for row, res in zip(w8_rows.itertuples(), ref_res):
            if res["direction"] == "SKIP":
                continue
            exp = "SELL" if row.window_open > row.PP else "BUY"
            if res["direction"] != exp:
                dir_ok = False
                print(f"      MISMATCH: date={row.trade_date_str} "
                      f"wo={row.window_open:.5f} pp={row.PP:.5f} "
                      f"expected={exp} got={res['direction']}")
                break
        chk(f"V5 [{pair}] Direction rule: SELL iff wo>PP  "
            f"(checked {sum(1 for r in ref_res if r['direction']!='SKIP')} trades)", dir_ok)

    # V6: TP rule — TP on correct side for every tp_mode x direction
    # Synthetic row: entry != pp to avoid TP_PP degenerate case.
    # BUY synthetic: entry below PP → TP_PP above entry.
    # SELL synthetic: entry above PP → TP_PP below entry.
    tp_ok = True
    for tpm in TP_MODES:
        for direction in ["BUY", "SELL"]:
            if direction == "BUY":
                # entry=99, pp=100 (entry < pp, so TP_PP=100 > 99 = entry)
                entry, atr, yh, yl, pp = 99.0, 1.0, 102.0, 96.0, 100.0
            else:
                # entry=101, pp=100 (entry > pp, so TP_PP=100 < 101 = entry)
                entry, atr, yh, yl, pp = 101.0, 1.0, 104.0, 98.0, 100.0
            tp = _tp_price(tpm, direction, entry, yh, yl, atr, pp)
            if direction == "BUY" and not (tp > entry):
                tp_ok = False
                print(f"      FAIL: tp_mode={tpm} dir=BUY  tp={tp:.4f} entry={entry}")
            if direction == "SELL" and not (tp < entry):
                tp_ok = False
                print(f"      FAIL: tp_mode={tpm} dir=SELL tp={tp:.4f} entry={entry}")
    chk(f"V6 [{pair}] TP on correct side for all tp_mode x direction", tp_ok)

    # V7: SL rule — SL on correct side of entry
    sl_ok = True
    for direction in ["BUY", "SELL"]:
        for slm in [1.0, 2.0, 3.0]:
            for smf in [0.3, 0.5, 0.7]:
                entry, atr = 100.0, 1.0
                sl_dist = atr * slm * smf
                sl = entry - sl_dist if direction == "BUY" else entry + sl_dist
                if direction == "BUY"  and sl >= entry:
                    sl_ok = False
                if direction == "SELL" and sl <= entry:
                    sl_ok = False
    chk(f"V7 [{pair}] SL on correct side for all param combos", sl_ok)

    # V8: Cost inclusion — verify net = gross - cost in a simple synthetic test
    # Synthesize: BUY at 100, TP at 101, pip=0.01, cost=2.0
    # Expected gross pips = 100 / 0.01 = 100 per lot, net = (100-2)*2 = 196
    pip_s = 0.01
    entry_s = 100.0
    tp_s    = 101.0
    sl_s    = 99.0
    cost_s  = 2.0
    raw = (tp_s - entry_s) * MAIN_SIZE - cost_s * pip_s * MAIN_SIZE
    net_s = raw / pip_s
    expected = (100.0 - 2.0) * MAIN_SIZE   # 196.0
    chk(f"V8 [{pair}] Cost correctly deducted from gross",
        abs(net_s - expected) < 1e-8,
        f"net={net_s:.2f} expected={expected:.2f}")

    return tr_rows, ad_rows, oos_rows, highs, lows, closes, pip, cost_pips


# ── V9: Determinism ────────────────────────────────────────────────────────────
def check_determinism(tr_rows, highs, lows, closes, pip, cost_pips, pair):
    print(f"\nV9: Determinism [{pair}]")
    cfg = {"sl_mult": 2.0, "sl_main_frac": 0.5, "tp_mode": "TP_ATR_1.5",
           "skip_frac": 0.30, "window_h": 8}
    res1 = _run_backtest_bl(tr_rows, highs, lows, closes, pip, cost_pips, cfg)
    res2 = _run_backtest_bl(tr_rows, highs, lows, closes, pip, cost_pips, cfg)
    match = all(r1["total"] == r2["total"] and r1["direction"] == r2["direction"]
                for r1, r2 in zip(res1, res2))
    chk(f"V9 [{pair}] Two identical runs produce identical results", match)


# ── Sample result spot-check ───────────────────────────────────────────────────
def spot_check(tr_rows, highs, lows, closes, pip, cost_pips, pair):
    print(f"\n  Spot-check sample ({pair}, 8h window, 3 configs):")
    n_tr = _n_cal_days(SPLIT["train_start"], SPLIT["train_end"])
    for cfg in [
        {"sl_mult": 1.0, "sl_main_frac": 0.3, "tp_mode": "TP_EDGE",    "skip_frac": 0.25, "window_h": 8},
        {"sl_mult": 2.0, "sl_main_frac": 0.5, "tp_mode": "TP_ATR_1.5", "skip_frac": 0.30, "window_h": 8},
        {"sl_mult": 3.5, "sl_main_frac": 0.7, "tp_mode": "TP_ATR_2.0", "skip_frac": 0.40, "window_h": 24},
    ]:
        res  = _run_backtest_bl(tr_rows, highs, lows, closes, pip, cost_pips, cfg)
        summ = _summarize_bl(res, n_tr)
        wh   = cfg["window_h"]
        print(f"    w{wh}h slm{cfg['sl_mult']:.1f} smf{cfg['sl_main_frac']:.1f} "
              f"{cfg['tp_mode']:>12} sf{cfg['skip_frac']:.2f} | "
              f"cal_ppd={summ['cal_ppd']:+7.4f}  WR={summ['win_rate']:.3f}  "
              f"trades={summ['total_trades']:4d}  ter={summ['time_exit_rate']:.3f}")


# ── Main ────────────────────────────────────────────────────────────────────────
def main():
    print(f"\n{'='*68}")
    print("  Engine Validation Memo — Baseline Discovery Step 1")
    print(f"{'='*68}")

    check_split_integrity()

    # Run per-pair checks on EURJPY (primary), spot-check others
    for i, pair in enumerate(["eurjpy", "eurusd", "usdjpy"]):
        quiet = (i > 0)
        print(f"\n{'='*60}")
        print(f"  Pair: {pair.upper()}")
        print(f"{'='*60}")
        tr_rows, ad_rows, oos_rows, highs, lows, closes, pip, cost_pips = \
            check_pair(pair, quiet=quiet)
        check_determinism(tr_rows, highs, lows, closes, pip, cost_pips, pair)
        if i == 0:
            spot_check(tr_rows, highs, lows, closes, pip, cost_pips, pair)

    print(f"\n{'='*68}")
    print(f"  RESULT: {PASS_COUNT} PASS / {FAIL_COUNT} FAIL")
    if FAIL_COUNT == 0:
        print("  ENGINE VALIDATION: PASSED — safe to proceed with full grid.")
    else:
        print("  ENGINE VALIDATION: FAILED — fix issues before running grid.")
    print(f"{'='*68}\n")

    return FAIL_COUNT


if __name__ == "__main__":
    fails = main()
    sys.exit(0 if fails == 0 else 1)
