"""
bl_layer4_defs.py  -  Layer 4: Main + Addon, BE_OFF vs BE_ON
=============================================================
Extends Layer 1 (Main only) with an addon leg.

Addon design:
  ao_frac scales sl_dist (NOT ATR) so ao_entry is always between entry and SL.
  BUY  trade: ao_entry = window_open - ao_frac * sl_dist  (fills on price dip)
  SELL trade: ao_entry = window_open + ao_frac * sl_dist  (fills on price rally)
  ao_sl  = main_sl   (shared protective stop)
  ao_tp  = main_tp   (shared take-profit)
  ao_size = AO_SIZE = 0.8 lots
  AO_FRACS = {0.30, 0.50, 0.70} — always between entry and SL (ao_frac < 1 guaranteed)

BE_OFF: no BE law -- main SL never moves, no BE-TP.
BE_ON : corrected BE law -- on addon fill (while main open):
    arm  main.be_tp_px     = main.entry  (TP-side target at entry price)
         main.be_tp_active = True
         main.be_tp_from_bar = fill_bar + 1  (next-bar activation)
    On subsequent bars: if H >= be_tp_px (BUY) or L <= be_tp_px (SELL)
      and no original SL/TP hit -> exit main at be_tp_px, reason=BE_TP.
    Conflict rule: if SL/TP also fires same bar, original exit wins.
    SAR: not active in Layer 4.

Per-leg tracking:
  main_net  -- net pips for main leg (MAIN_SIZE lots, after cost)
  addon_net -- net pips for addon leg (AO_SIZE lots, after cost) or 0 if unfilled
  total     -- main_net + addon_net

Exit reason counters (for main leg):
  sl_count, tp_count, be_tp_count, time_exit_count

Both variants are run inside the same pool task to share data loads.
"""
from __future__ import annotations

import math
import numpy as np
import pandas as pd
from pathlib import Path
import sys

_HERE = Path(__file__).parent
sys.path.insert(0, str(_HERE))

from bl_defs import (
    SPLIT, MAIN_SIZE, WORKERS, ATR_PERIOD, WINDOWS,
    SL_MULTS, SL_MAIN_FRACS, TP_MODES, SKIP_FRACS,
    HONING_STOP_DELTA, MAX_ROUNDS, MIN_ROUNDS,
    _tp_price, _n_cal_days,
)

# ── Layer 4 constants ──────────────────────────────────────────────────────────
AO_SIZE   = 0.8                   # addon lots (fixed)
# ao_frac = fraction of sl_dist (NOT ATR) for addon entry offset.
# This guarantees ao_entry is always between main_entry and main_sl (ao_frac < 1).
# BUY:  ao_entry = window_open - ao_frac * sl_dist  (fills on price dip)
# SELL: ao_entry = window_open + ao_frac * sl_dist  (fills on price rally)
AO_FRACS  = [0.30, 0.50, 0.70]   # 30/50/70% of sl_dist → ao_entry always between entry and SL


# ── Core backtest (one trade per row) ─────────────────────────────────────────
def _run_one_trade_layer4(
    wo: float,          # window_open (main entry price)
    yh: float, yl: float, yc: float, pp: float,
    atr: float, yr: float,
    bar_idx: int,
    highs: np.ndarray, lows: np.ndarray, closes: np.ndarray,
    pip: float, cost_pips: float,
    sf: float, slm: float, smf: float, tpm: str,
    ao_frac: float,
    be_on: bool,
) -> dict | None:
    """
    Simulate one trade (main + optional addon).
    Returns result dict or None if trade is skipped.
    """
    n_bars = len(highs)
    mh     = SPLIT["max_hold_hours"]

    # ── Eligibility guards ─────────────────────────────────────────────────
    if atr < pip * 5:           return None   # degenerate ATR
    if yr  < sf * atr:          return None   # range too compressed (skip_frac)
    if bar_idx + mh > n_bars:   return None   # 432h guard

    # ── Direction (LOCKED) ─────────────────────────────────────────────────
    direction = "SELL" if wo > pp else "BUY"

    # ── Main SL / TP ───────────────────────────────────────────────────────
    sl_dist = atr * slm * smf
    if direction == "BUY":
        main_sl = wo - sl_dist
    else:
        main_sl = wo + sl_dist

    main_tp = _tp_price(tpm, direction, wo, yh, yl, atr, pp)

    if direction == "BUY"  and main_tp <= wo:  return None
    if direction == "SELL" and main_tp >= wo:  return None

    # ── Addon entry ────────────────────────────────────────────────────────
    # ao_frac scales sl_dist → ao_entry is always between entry and SL.
    # For ao_frac in (0, 1): ao_entry strictly between entry and SL → always valid.
    if direction == "BUY":
        ao_entry = wo - ao_frac * sl_dist
        ao_valid = ao_entry > main_sl          # sanity (always true for ao_frac < 1)
    else:
        ao_entry = wo + ao_frac * sl_dist
        ao_valid = ao_entry < main_sl

    # Addon uses shared SL and TP
    ao_sl = main_sl
    ao_tp = main_tp

    # ── Simulation state ───────────────────────────────────────────────────
    main_open    = True
    main_exit_px = closes[min(bar_idx + mh - 1, n_bars - 1)]
    main_reason  = "TIME_EXIT"

    be_tp_px      = float("nan")
    be_tp_active  = False
    be_tp_from_bar = -1

    addon_filled  = False
    addon_open    = False
    addon_exit_px = float("nan")
    addon_reason  = "NOT_FILLED"

    end_bar = min(bar_idx + mh, n_bars)

    for j in range(bar_idx, end_bar):
        h_ = highs[j]
        l_ = lows[j]

        # ── Addon fill ─────────────────────────────────────────────────────
        if ao_valid and not addon_filled:
            ao_cond = (l_ <= ao_entry) if direction == "BUY" else (h_ >= ao_entry)
            if ao_cond:
                addon_filled  = True
                addon_open    = True
                addon_exit_px = closes[end_bar - 1]  # time-exit default
                addon_reason  = "TIME_EXIT"

                # Arm BE-TP (BE_ON only, and only while main is still open)
                if be_on and main_open:
                    be_tp_px       = wo         # TP-side target = main entry
                    be_tp_active   = True
                    be_tp_from_bar = j + 1      # next-bar activation

        # ── Main exit ──────────────────────────────────────────────────────
        if main_open:
            hsl_m = (l_ <= main_sl) if direction == "BUY" else (h_ >= main_sl)
            htp_m = (h_ >= main_tp) if direction == "BUY" else (l_ <= main_tp)

            be_hit = (be_tp_active
                      and j >= be_tp_from_bar
                      and ((direction == "BUY"  and h_ >= be_tp_px)
                           or (direction == "SELL" and l_ <= be_tp_px)))

            # Conflict: original SL/TP wins over BE-TP (same bar)
            if hsl_m and htp_m:
                main_exit_px = main_sl; main_reason = "BOTH_SL"; main_open = False
            elif hsl_m:
                main_exit_px = main_sl; main_reason = "SL";      main_open = False
            elif htp_m:
                main_exit_px = main_tp; main_reason = "TP";      main_open = False
            elif be_hit:
                main_exit_px = be_tp_px; main_reason = "BE_TP";  main_open = False

        # ── Addon exit ─────────────────────────────────────────────────────
        if addon_open:
            hsl_a = (l_ <= ao_sl) if direction == "BUY" else (h_ >= ao_sl)
            htp_a = (h_ >= ao_tp) if direction == "BUY" else (l_ <= ao_tp)

            if hsl_a and htp_a:
                addon_exit_px = ao_sl; addon_reason = "BOTH_SL"; addon_open = False
            elif hsl_a:
                addon_exit_px = ao_sl; addon_reason = "SL";      addon_open = False
            elif htp_a:
                addon_exit_px = ao_tp; addon_reason = "TP";      addon_open = False

        # ── Early exit when both positions are done ────────────────────────
        # (if addon never filled and main closed: no point continuing)
        if not main_open and (not ao_valid or not addon_filled or not addon_open):
            break
        if not main_open and not addon_open:
            break

    # ── P&L ────────────────────────────────────────────────────────────────
    if direction == "BUY":
        main_raw  = (main_exit_px - wo) * MAIN_SIZE - cost_pips * pip * MAIN_SIZE
        addon_raw = ((addon_exit_px - ao_entry) * AO_SIZE
                     - cost_pips * pip * AO_SIZE) if addon_filled else 0.0
    else:
        main_raw  = (wo - main_exit_px) * MAIN_SIZE - cost_pips * pip * MAIN_SIZE
        addon_raw = ((ao_entry - addon_exit_px) * AO_SIZE
                     - cost_pips * pip * AO_SIZE) if addon_filled else 0.0

    main_net  = main_raw  / pip
    addon_net = addon_raw / pip if addon_filled else 0.0

    return {
        "direction":    direction,
        "main_net":     main_net,
        "addon_net":    addon_net,
        "total":        main_net + addon_net,
        "main_reason":  main_reason,
        "addon_reason": addon_reason,
        "addon_filled": addon_filled,
    }


def _run_layer4(
    rows: pd.DataFrame,
    highs:   np.ndarray,
    lows:    np.ndarray,
    closes:  np.ndarray,
    pip:     float,
    cost_pips: float,
    cfg:     dict,
    be_on:   bool,
) -> list:
    """
    Run Layer 4 backtest for one cfg × one BE variant.
    Returns list of dicts (SKIP or result per row).
    """
    sf   = float(cfg["skip_frac"])
    slm  = float(cfg["sl_mult"])
    smf  = float(cfg["sl_main_frac"])
    tpm  = str(cfg["tp_mode"])
    wh   = int(cfg["window_h"])
    aof  = float(cfg["ao_frac"])

    w_rows = rows[rows["window_h"] == wh] if "window_h" in rows.columns else rows
    results = []

    for _, row in w_rows.iterrows():
        date_s  = str(row["trade_date_str"])
        bar_idx = int(row["window_bar_idx"])

        res = _run_one_trade_layer4(
            wo=float(row["window_open"]),
            yh=float(row["YH"]), yl=float(row["YL"]),
            yc=float(row["YC"]), pp=float(row["PP"]),
            atr=float(row["ATR14"]), yr=float(row["YR"]),
            bar_idx=bar_idx,
            highs=highs, lows=lows, closes=closes,
            pip=pip, cost_pips=cost_pips,
            sf=sf, slm=slm, smf=smf, tpm=tpm,
            ao_frac=aof, be_on=be_on,
        )
        if res is None:
            results.append({"date": date_s, "direction": "SKIP",
                             "total": 0.0, "main_net": 0.0, "addon_net": 0.0,
                             "main_reason": "SKIP", "addon_reason": "SKIP",
                             "addon_filled": False})
        else:
            res["date"] = date_s
            results.append(res)

    return results


# ── Summary statistics (per-leg) ───────────────────────────────────────────────
def _summarize_layer4(results: list, n_cal_days: int) -> dict:
    triggered = [r for r in results if r.get("direction") != "SKIP"]
    if not triggered:
        return {
            "cal_ppd": 0.0, "main_cal_ppd": 0.0, "addon_cal_ppd": 0.0,
            "win_rate": 0.0, "total_trades": 0, "n_cal_days": n_cal_days,
            "addon_fire_rate": 0.0, "best_day": 0.0, "worst_day": 0.0,
            "be_tp_count": 0, "sl_count": 0, "tp_count": 0, "time_exit_count": 0,
        }

    n          = len(triggered)
    main_nets  = [r["main_net"]  for r in triggered]
    addon_nets = [r["addon_net"] for r in triggered]
    total_nets = [r["total"]     for r in triggered]
    reasons    = [r["main_reason"] for r in triggered]

    return {
        "cal_ppd":         round(sum(total_nets) / max(n_cal_days, 1), 4),
        "main_cal_ppd":    round(sum(main_nets)  / max(n_cal_days, 1), 4),
        "addon_cal_ppd":   round(sum(addon_nets) / max(n_cal_days, 1), 4),
        "win_rate":        round(sum(1 for x in total_nets if x > 0) / n, 4),
        "total_trades":    n,
        "n_cal_days":      n_cal_days,
        "addon_fire_rate": round(sum(1 for r in triggered if r["addon_filled"]) / n, 4),
        "best_day":        round(max(total_nets), 2),
        "worst_day":       round(min(total_nets), 2),
        "be_tp_count":     sum(1 for r in reasons if r == "BE_TP"),
        "sl_count":        sum(1 for r in reasons if r in ("SL", "BOTH_SL")),
        "tp_count":        sum(1 for r in reasons if r == "TP"),
        "time_exit_count": sum(1 for r in reasons if r == "TIME_EXIT"),
    }


# ── Pool worker (returns BOTH BE_OFF and BE_ON summaries) ─────────────────────
_g4_highs  = None
_g4_lows   = None
_g4_closes = None


def _pool_init_l4(highs, lows, closes):
    global _g4_highs, _g4_lows, _g4_closes
    _g4_highs  = highs
    _g4_lows   = lows
    _g4_closes = closes


def _task_l4(task):
    """
    task = (rows_records, n_cal_days, pip, cost_pips, cfg)
    Returns: (cfg, summ_be_off, summ_be_on)
    Both variants computed per task to share data overhead.
    """
    rows_records, n_cal, pip, cost_pips, cfg = task
    df = pd.DataFrame(rows_records)
    res_off = _run_layer4(df, _g4_highs, _g4_lows, _g4_closes, pip, cost_pips, cfg, be_on=False)
    res_on  = _run_layer4(df, _g4_highs, _g4_lows, _g4_closes, pip, cost_pips, cfg, be_on=True)
    return (cfg,
            _summarize_layer4(res_off, n_cal),
            _summarize_layer4(res_on,  n_cal))
