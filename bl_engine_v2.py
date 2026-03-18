"""
bl_engine_v2.py  -  Unified per-leg backtest engine for L1-L4
=============================================================
Each layer is a strict mode with no mixing:
  L1  main-only      : main_on=T  addon_on=F  sar_on=F
  L2  addon-only     : main_on=F  addon_on=T  sar_on=F  (addon as standalone)
  L3  main+SAR       : main_on=T  addon_on=F  sar_on=T
  L4  main+addon     : main_on=T  addon_on=T  sar_on=F  (BE_OFF + BE_ON)

Per-day output: list of leg records (one per filled leg + SKIP/NO_FILL markers).
Each leg record is a dict -- the raw material for ledger CSVs and summaries.

Leg record schema:
  date, window_h, layer, be_mode,
  leg_tag        : "main" | "addon" | "sar"
  side           : "BUY" | "SELL" | "N/A"
  direction      : overall trade direction
  filled         : bool  (True = leg actually executed)
  skip_reason    : why skipped / unfilled (or "")
  entry_bar_idx  : int (-1 if unfilled)
  exit_bar_idx   : int (-1 if unfilled)
  entry_px       : float (nan if unfilled)
  exit_px        : float (nan if unfilled)
  exit_reason    : "SL"|"BOTH_SL"|"TP"|"BE_TP"|"TIME_EXIT"|"NO_FILL"|"SKIP"
  pips_net       : float (0 if unfilled)
  lots           : float (0 if unfilled)
  config_id      : str

Intrabar convention (all layers):
  On same-bar SL+TP: SL wins (BOTH_SL).
  On same-bar SL+BE_TP: SL wins (original exit beats BE).
  On same-bar TP+BE_TP: TP wins (original exit beats BE).
"""
from __future__ import annotations
import math
import numpy as np
import pandas as pd
from pathlib import Path
import sys

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))

from bl_defs import (
    SPLIT, WINDOW_UTC_START, ATR_PERIOD,
    MAIN_SIZE, WORKERS,
    SL_MULTS, SL_MAIN_FRACS, TP_MODES, SKIP_FRACS, WINDOWS,
    _atr_series, _tp_price, _n_cal_days,
    _build_rows_all_windows,
)

# ── Layer-specific size constants ─────────────────────────────────────────────
AO_SIZE   = 0.8    # addon lots
SAR_SIZE  = 0.7    # SAR lots

# ── Addon entry offsets (fraction of sl_dist — always between entry and SL) ───
AO_FRACS  = [0.30, 0.50, 0.70]   # 30/50/70 % of sl_dist

# ── SAR TP/SL grid (fraction of YR = yesterday range) ────────────────────────
SAR_TP_FRACS = [0.50, 0.70, 1.00]   # SAR TP distance = frac * YR
SAR_SL_FRACS = [0.25, 0.35, 0.50]   # SAR SL distance = frac * YR

NAN = float("nan")


# ── Config ID builder ─────────────────────────────────────────────────────────
def cfg_id(cfg: dict) -> str:
    parts = [f"w{cfg['window_h']}",
             f"slm{cfg['sl_mult']:.2f}",
             f"smf{cfg['sl_main_frac']:.2f}",
             cfg["tp_mode"],
             f"sf{cfg['skip_frac']:.2f}"]
    if "ao_frac"     in cfg: parts.append(f"ao{cfg['ao_frac']:.2f}")
    if "sar_tp_frac" in cfg: parts.append(f"stp{cfg['sar_tp_frac']:.2f}")
    if "sar_sl_frac" in cfg: parts.append(f"ssl{cfg['sar_sl_frac']:.2f}")
    return "|".join(parts)


# ── Base skeleton for a leg record ────────────────────────────────────────────
def _skel(date_s: str, wh: int, layer: str, be_mode: str,
          leg: str, direction: str, cid: str) -> dict:
    return {
        "date": date_s, "window_h": wh, "layer": layer, "be_mode": be_mode,
        "leg_tag": leg, "side": "N/A", "direction": direction,
        "filled": False, "skip_reason": "",
        "entry_bar_idx": -1, "exit_bar_idx": -1,
        "entry_px": NAN, "exit_px": NAN,
        "exit_reason": "NO_FILL", "pips_net": 0.0, "lots": 0.0,
        "config_id": cid,
    }


# ── P&L helper ────────────────────────────────────────────────────────────────
def _pnl(side: str, entry: float, exit_px: float, size: float,
          pip: float, cost_pips: float) -> float:
    if side == "BUY":
        raw = (exit_px - entry) * size - cost_pips * pip * size
    else:
        raw = (entry - exit_px) * size - cost_pips * pip * size
    return raw / pip


# ── Single-leg exit simulator (shared by all layers) ─────────────────────────
def _sim_exit(side: str, entry: float, sl: float, tp: float | None,
              bar_start: int, highs: np.ndarray, lows: np.ndarray,
              closes: np.ndarray, max_hold: int,
              be_tp_px: float | None = None, be_from_bar: int = -1) -> tuple:
    """
    Simulate exit for one leg starting at bar_start.
    Returns: (exit_bar_idx, exit_px, exit_reason)
    Intrabar convention: SL wins over TP on BOTH_SL; original wins over BE_TP.
    """
    n = len(highs)
    end_bar = min(bar_start + max_hold, n)
    last_close = closes[end_bar - 1]

    for j in range(bar_start, end_bar):
        h_ = highs[j]; l_ = lows[j]

        hsl = (l_ <= sl)  if side == "BUY"  else (h_ >= sl)
        htp = (h_ >= tp)  if (side == "BUY"  and tp is not None) else \
              (l_ <= tp)  if (side == "SELL" and tp is not None) else False
        be_hit = (be_tp_px is not None
                  and j >= be_from_bar
                  and ((side == "BUY"  and h_ >= be_tp_px)
                       or (side == "SELL" and l_ <= be_tp_px)))

        # Original exit beats BE_TP on same-bar conflict
        if hsl and htp: return j, sl, "BOTH_SL"
        if hsl:         return j, sl, "SL"
        if htp:         return j, float(tp), "TP"
        if be_hit:      return j, float(be_tp_px), "BE_TP"

    return end_bar - 1, last_close, "TIME_EXIT"


# ═══════════════════════════════════════════════════════════════════════════════
# L1 — main-only
# ═══════════════════════════════════════════════════════════════════════════════
def _run_day_L1(date_s, wh, bar_idx, wo, yh, yl, pp, atr, yr,
                highs, lows, closes, pip, cost_pips, cfg) -> list:
    layer = "L1"; cid = cfg_id(cfg)
    m = _skel(date_s, wh, layer, "N/A", "main", "?", cid)
    sf = cfg["skip_frac"]; slm = cfg["sl_mult"]; smf = cfg["sl_main_frac"]
    tpm = cfg["tp_mode"]; mh = SPLIT["max_hold_hours"]

    if atr < pip * 5:
        m["skip_reason"] = "degenerate_atr"; m["exit_reason"] = "SKIP"; return [m]
    if yr < sf * atr:
        m["skip_reason"] = "skip_frac";     m["exit_reason"] = "SKIP"; return [m]
    if bar_idx + mh > len(highs):
        m["skip_reason"] = "432h_guard";    m["exit_reason"] = "SKIP"; return [m]

    side = "SELL" if wo > pp else "BUY"
    m["direction"] = side; m["side"] = side

    sl_dist = atr * slm * smf
    sl  = wo - sl_dist if side == "BUY" else wo + sl_dist
    tp  = _tp_price(tpm, side, wo, yh, yl, atr, pp)
    if (side == "BUY" and tp <= wo) or (side == "SELL" and tp >= wo):
        m["skip_reason"] = "tp_wrong_side"; m["exit_reason"] = "SKIP"; return [m]

    ebar, epx, reason = _sim_exit(side, wo, sl, tp, bar_idx, highs, lows, closes, mh)
    m.update({"filled": True, "entry_bar_idx": bar_idx, "exit_bar_idx": ebar,
              "entry_px": wo, "exit_px": epx, "exit_reason": reason,
              "pips_net": _pnl(side, wo, epx, MAIN_SIZE, pip, cost_pips),
              "lots": MAIN_SIZE})
    return [m]


# ═══════════════════════════════════════════════════════════════════════════════
# L2 — addon-only (addon as standalone, fills at ao_entry)
# ═══════════════════════════════════════════════════════════════════════════════
def _run_day_L2(date_s, wh, bar_idx, wo, yh, yl, pp, atr, yr,
                highs, lows, closes, pip, cost_pips, cfg) -> list:
    layer = "L2"; cid = cfg_id(cfg)
    a = _skel(date_s, wh, layer, "N/A", "addon", "?", cid)
    sf = cfg["skip_frac"]; slm = cfg["sl_mult"]; smf = cfg["sl_main_frac"]
    tpm = cfg["tp_mode"]; aof = cfg["ao_frac"]; mh = SPLIT["max_hold_hours"]
    n = len(highs)

    if atr < pip * 5:
        a["skip_reason"] = "degenerate_atr"; a["exit_reason"] = "SKIP"; return [a]
    if yr < sf * atr:
        a["skip_reason"] = "skip_frac";     a["exit_reason"] = "SKIP"; return [a]
    if bar_idx + mh > n:
        a["skip_reason"] = "432h_guard";    a["exit_reason"] = "SKIP"; return [a]

    side = "SELL" if wo > pp else "BUY"
    a["direction"] = side; a["side"] = side

    sl_dist  = atr * slm * smf
    ao_entry = wo - aof * sl_dist if side == "BUY" else wo + aof * sl_dist
    ao_sl    = wo - sl_dist       if side == "BUY" else wo + sl_dist
    ao_tp    = _tp_price(tpm, side, wo, yh, yl, atr, pp)  # TP relative to wo

    if (side == "BUY" and ao_tp <= wo) or (side == "SELL" and ao_tp >= wo):
        a["skip_reason"] = "tp_wrong_side"; a["exit_reason"] = "SKIP"; return [a]

    # Find fill bar: first bar in [bar_idx, bar_idx+mh) where price touches ao_entry
    end_bar = min(bar_idx + mh, n)
    fill_bar = -1
    for j in range(bar_idx, end_bar):
        if (side == "BUY"  and lows[j]  <= ao_entry) or \
           (side == "SELL" and highs[j] >= ao_entry):
            fill_bar = j; break

    if fill_bar == -1:
        a["exit_reason"] = "NO_FILL"; return [a]  # addon never filled

    ebar, epx, reason = _sim_exit(side, ao_entry, ao_sl, ao_tp,
                                   fill_bar, highs, lows, closes,
                                   mh - (fill_bar - bar_idx))
    a.update({"filled": True, "entry_bar_idx": fill_bar, "exit_bar_idx": ebar,
              "entry_px": ao_entry, "exit_px": epx, "exit_reason": reason,
              "pips_net": _pnl(side, ao_entry, epx, AO_SIZE, pip, cost_pips),
              "lots": AO_SIZE})
    return [a]


# ═══════════════════════════════════════════════════════════════════════════════
# L3 — main + SAR (SAR fires only on SL/BOTH_SL exit of main)
# ═══════════════════════════════════════════════════════════════════════════════
def _run_day_L3(date_s, wh, bar_idx, wo, yh, yl, pp, atr, yr,
                highs, lows, closes, pip, cost_pips, cfg) -> list:
    layer = "L3"; cid = cfg_id(cfg)
    sf = cfg["skip_frac"]; slm = cfg["sl_mult"]; smf = cfg["sl_main_frac"]
    tpm = cfg["tp_mode"]; mh = SPLIT["max_hold_hours"]
    stp = cfg["sar_tp_frac"]; ssl = cfg["sar_sl_frac"]
    n = len(highs)

    m = _skel(date_s, wh, layer, "N/A", "main", "?", cid)
    s = _skel(date_s, wh, layer, "N/A", "sar",  "?", cid)

    if atr < pip * 5:
        r = "degenerate_atr"; m["skip_reason"] = r; m["exit_reason"] = "SKIP"
        s["skip_reason"] = r; s["exit_reason"] = "SKIP"; return [m, s]
    if yr < sf * atr:
        r = "skip_frac"; m["skip_reason"] = r; m["exit_reason"] = "SKIP"
        s["skip_reason"] = r; s["exit_reason"] = "SKIP"; return [m, s]
    if bar_idx + mh > n:
        r = "432h_guard"; m["skip_reason"] = r; m["exit_reason"] = "SKIP"
        s["skip_reason"] = r; s["exit_reason"] = "SKIP"; return [m, s]

    side = "SELL" if wo > pp else "BUY"
    sar_side = "BUY" if side == "SELL" else "SELL"
    m["direction"] = side; m["side"] = side
    s["direction"] = side  # overall trade direction (main direction)

    sl_dist = atr * slm * smf
    main_sl = wo - sl_dist if side == "BUY" else wo + sl_dist
    main_tp = _tp_price(tpm, side, wo, yh, yl, atr, pp)

    if (side == "BUY" and main_tp <= wo) or (side == "SELL" and main_tp >= wo):
        r = "tp_wrong_side"; m["skip_reason"] = r; m["exit_reason"] = "SKIP"
        s["skip_reason"] = r; s["exit_reason"] = "SKIP"; return [m, s]

    # Main leg
    mebar, mepx, mreason = _sim_exit(side, wo, main_sl, main_tp,
                                      bar_idx, highs, lows, closes, mh)
    m.update({"filled": True, "entry_bar_idx": bar_idx, "exit_bar_idx": mebar,
              "entry_px": wo, "exit_px": mepx, "exit_reason": mreason,
              "pips_net": _pnl(side, wo, mepx, MAIN_SIZE, pip, cost_pips),
              "lots": MAIN_SIZE})

    # SAR: fires only on main SL exits
    s["direction"] = side; s["side"] = sar_side
    if mreason in ("SL", "BOTH_SL"):
        sar_entry = mepx   # SAR enters at main exit price
        sar_tp  = (sar_entry - stp * yr) if sar_side == "SELL" else (sar_entry + stp * yr)
        sar_sl  = (sar_entry + ssl * yr) if sar_side == "SELL" else (sar_entry - ssl * yr)
        # SAR starts from bar AFTER main exit
        sar_start = min(mebar + 1, n - 1)
        remaining = mh - (sar_start - bar_idx)
        if remaining > 0 and sar_start < n:
            sebar, sepx, sreason = _sim_exit(sar_side, sar_entry, sar_sl, sar_tp,
                                              sar_start, highs, lows, closes, remaining)
            s.update({"filled": True, "entry_bar_idx": sar_start, "exit_bar_idx": sebar,
                      "entry_px": sar_entry, "exit_px": sepx, "exit_reason": sreason,
                      "pips_net": _pnl(sar_side, sar_entry, sepx, SAR_SIZE, pip, cost_pips),
                      "lots": SAR_SIZE})
        else:
            s["skip_reason"] = "no_bars_left"

    return [m, s]


# ═══════════════════════════════════════════════════════════════════════════════
# L4 — main + addon, BE_OFF and BE_ON (returns records for BOTH variants)
# ═══════════════════════════════════════════════════════════════════════════════
def _run_day_L4(date_s, wh, bar_idx, wo, yh, yl, pp, atr, yr,
                highs, lows, closes, pip, cost_pips, cfg) -> tuple:
    """Returns (records_be_off, records_be_on) each as list of leg dicts."""
    cid = cfg_id(cfg)
    sf = cfg["skip_frac"]; slm = cfg["sl_mult"]; smf = cfg["sl_main_frac"]
    tpm = cfg["tp_mode"]; aof = cfg["ao_frac"]; mh = SPLIT["max_hold_hours"]
    n = len(highs)

    def skips(reason):
        off = [_skel(date_s, wh, "L4", "BE_OFF", lg, "?", cid) for lg in ("main","addon")]
        on  = [_skel(date_s, wh, "L4", "BE_ON",  lg, "?", cid) for lg in ("main","addon")]
        for r in off + on:
            r["skip_reason"] = reason; r["exit_reason"] = "SKIP"
        return off, on

    if atr < pip * 5:        return skips("degenerate_atr")
    if yr < sf * atr:        return skips("skip_frac")
    if bar_idx + mh > n:     return skips("432h_guard")

    side = "SELL" if wo > pp else "BUY"
    sl_dist  = atr * slm * smf
    main_sl  = wo - sl_dist if side == "BUY" else wo + sl_dist
    main_tp  = _tp_price(tpm, side, wo, yh, yl, atr, pp)

    def tp_skip():
        return skips("tp_wrong_side")

    if (side == "BUY" and main_tp <= wo) or (side == "SELL" and main_tp >= wo):
        return tp_skip()

    # Addon entry (fraction of sl_dist from main entry — always between entry and SL)
    ao_entry = wo - aof * sl_dist if side == "BUY" else wo + aof * sl_dist
    ao_sl    = main_sl
    ao_tp    = main_tp

    # ── Simulate both BE variants in one pass ─────────────────────────────────
    def simulate(be_on: bool) -> list:
        be_mode = "BE_ON" if be_on else "BE_OFF"

        m = _skel(date_s, wh, "L4", be_mode, "main",  side, cid)
        a = _skel(date_s, wh, "L4", be_mode, "addon", side, cid)
        m["direction"] = side; a["direction"] = side

        # State
        main_open  = True
        main_exit  = closes[min(bar_idx + mh - 1, n - 1)]
        main_ebar  = min(bar_idx + mh - 1, n - 1)
        main_rsn   = "TIME_EXIT"

        be_px      = NAN
        be_active  = False
        be_from    = -1

        ao_filled  = False
        ao_open    = False
        ao_exit    = closes[min(bar_idx + mh - 1, n - 1)]
        ao_ebar    = min(bar_idx + mh - 1, n - 1)
        ao_rsn     = "NO_FILL"

        end_bar = min(bar_idx + mh, n)

        for j in range(bar_idx, end_bar):
            h_ = highs[j]; l_ = lows[j]

            # Addon fill
            if not ao_filled:
                af = (l_ <= ao_entry) if side == "BUY" else (h_ >= ao_entry)
                if af:
                    ao_filled = True; ao_open = True
                    ao_exit = closes[end_bar - 1]; ao_ebar = end_bar - 1; ao_rsn = "TIME_EXIT"
                    if be_on and main_open:
                        be_px = wo; be_active = True; be_from = j + 1

            # Main exit
            if main_open:
                hsl = (l_ <= main_sl) if side == "BUY" else (h_ >= main_sl)
                htp = (h_ >= main_tp) if side == "BUY" else (l_ <= main_tp)
                bh  = (be_active and j >= be_from
                       and ((side == "BUY"  and h_ >= be_px)
                            or (side == "SELL" and l_ <= be_px)))
                if hsl and htp: main_exit=main_sl; main_ebar=j; main_rsn="BOTH_SL"; main_open=False
                elif hsl:       main_exit=main_sl; main_ebar=j; main_rsn="SL";       main_open=False
                elif htp:       main_exit=float(main_tp); main_ebar=j; main_rsn="TP";  main_open=False
                elif bh:        main_exit=be_px;   main_ebar=j; main_rsn="BE_TP";    main_open=False

            # Addon exit
            if ao_open:
                hsl_a = (l_ <= ao_sl) if side == "BUY" else (h_ >= ao_sl)
                htp_a = (h_ >= ao_tp) if side == "BUY" else (l_ <= ao_tp)
                if hsl_a and htp_a: ao_exit=ao_sl;          ao_ebar=j; ao_rsn="BOTH_SL"; ao_open=False
                elif hsl_a:          ao_exit=ao_sl;          ao_ebar=j; ao_rsn="SL";      ao_open=False
                elif htp_a:          ao_exit=float(ao_tp);   ao_ebar=j; ao_rsn="TP";      ao_open=False

            # Early exit: both closed, or main closed and addon never filled
            if not main_open and (not ao_filled or not ao_open):
                break
            if not main_open and not ao_open:
                break

        # Finalise main record
        m.update({"filled": True, "entry_bar_idx": bar_idx, "exit_bar_idx": main_ebar,
                  "entry_px": wo, "exit_px": main_exit, "exit_reason": main_rsn,
                  "pips_net": _pnl(side, wo, main_exit, MAIN_SIZE, pip, cost_pips),
                  "lots": MAIN_SIZE})

        # Finalise addon record
        if ao_filled:
            a.update({"filled": True, "entry_bar_idx": bar_idx, "exit_bar_idx": ao_ebar,
                      "entry_px": ao_entry, "exit_px": ao_exit, "exit_reason": ao_rsn,
                      "pips_net": _pnl(side, ao_entry, ao_exit, AO_SIZE, pip, cost_pips),
                      "lots": AO_SIZE})
        else:
            a["exit_reason"] = "NO_FILL"

        return [m, a]

    return simulate(be_on=False), simulate(be_on=True)


# ═══════════════════════════════════════════════════════════════════════════════
# Row-level dispatcher (called per row from pool worker)
# ═══════════════════════════════════════════════════════════════════════════════
def _dispatch(row: dict, highs, lows, closes,
              pip: float, cost_pips: float, cfg: dict,
              layer: str) -> object:
    """
    Dispatch one row to the correct layer handler.
    L4 returns (records_off, records_on); all others return records list.
    """
    kw = dict(date_s=row["trade_date_str"], wh=int(row["window_h"]),
               bar_idx=int(row["window_bar_idx"]),
               wo=float(row["window_open"]),
               yh=float(row["YH"]), yl=float(row["YL"]),
               pp=float(row["PP"]),
               atr=float(row["ATR14"]), yr=float(row["YR"]),
               highs=highs, lows=lows, closes=closes,
               pip=pip, cost_pips=cost_pips, cfg=cfg)
    if layer == "L1": return _run_day_L1(**kw)
    if layer == "L2": return _run_day_L2(**kw)
    if layer == "L3": return _run_day_L3(**kw)
    if layer == "L4": return _run_day_L4(**kw)
    raise ValueError(f"Unknown layer: {layer}")


# ═══════════════════════════════════════════════════════════════════════════════
# Pool worker infrastructure
# ═══════════════════════════════════════════════════════════════════════════════
_g_highs  = None
_g_lows   = None
_g_closes = None
_g_times  = None


def pool_init(highs, lows, closes, times=None):
    global _g_highs, _g_lows, _g_closes, _g_times
    _g_highs = highs; _g_lows = lows; _g_closes = closes; _g_times = times


def pool_task(task):
    """
    SWEEP TASK (memory-safe): computes per-leg summary, does NOT return records.
    task = (rows_records, pip, cost_pips, cfg, layer, n_cal)
    Returns: (cfg, layer, ppd_select, summary_off, summary_on)
      ppd_select = BE_ON cal_ppd for L4, else total cal_ppd
    """
    rows_records, pip, cost_pips, cfg, layer, n_cal = task
    # Accumulate pips per leg (avoids storing all records)
    leg_totals_off = {"main": 0.0, "addon": 0.0, "sar": 0.0}
    leg_totals_on  = {"main": 0.0, "addon": 0.0, "sar": 0.0}
    counts_off     = {"n": 0, "sl": 0, "tp": 0, "be_tp": 0, "te": 0, "ao_fill": 0}
    counts_on      = {"n": 0, "sl": 0, "tp": 0, "be_tp": 0, "te": 0, "ao_fill": 0}

    for row in rows_records:
        res = _dispatch(row, _g_highs, _g_lows, _g_closes, pip, cost_pips, cfg, layer)
        def accum(recs, totals, cnts):
            # Find if main triggered (not SKIP)
            main = next((r for r in recs if r["leg_tag"] == "main"), None)
            if main and main.get("filled"):
                cnts["n"] += 1
                rsn = main.get("exit_reason", "")
                if rsn in ("SL","BOTH_SL"): cnts["sl"] += 1
                elif rsn == "TP":           cnts["tp"] += 1
                elif rsn == "BE_TP":        cnts["be_tp"] += 1
                elif rsn == "TIME_EXIT":    cnts["te"] += 1
            for r in recs:
                if r.get("filled"):
                    totals[r["leg_tag"]] = totals.get(r["leg_tag"], 0.0) + r["pips_net"]
                    if r["leg_tag"] == "addon": cnts["ao_fill"] += 1
        if layer == "L4":
            off_recs, on_recs = res
            accum(off_recs, leg_totals_off, counts_off)
            accum(on_recs,  leg_totals_on,  counts_on)
        else:
            accum(res, leg_totals_off, counts_off)

    def mk_summ(totals, cnts, n_cal):
        total = sum(totals.values())
        n = max(cnts["n"], 1)
        return {
            "cal_ppd":       round(total / max(n_cal, 1), 4),
            "main_cal_ppd":  round(totals.get("main",0) / max(n_cal,1), 4),
            "addon_cal_ppd": round(totals.get("addon",0) / max(n_cal,1), 4),
            "sar_cal_ppd":   round(totals.get("sar",0) / max(n_cal,1), 4),
            "total_pips":    round(total, 2),
            "n_triggered":   cnts["n"],
            "sl_count":      cnts["sl"], "tp_count": cnts["tp"],
            "be_tp_count":   cnts["be_tp"], "time_exit_count": cnts["te"],
            "addon_fire_count": cnts["ao_fill"],
        }

    s_off = mk_summ(leg_totals_off, counts_off, n_cal)
    s_on  = mk_summ(leg_totals_on,  counts_on,  n_cal)
    ppd_select = s_on["cal_ppd"] if layer == "L4" else s_off["cal_ppd"]
    return cfg, layer, ppd_select, s_off, s_on


def replay_config(cfg: dict, rows_df, pip: float, cost_pips: float, layer: str) -> tuple:
    """
    REPLAY TASK: run one frozen config to get full per-trade records for ledger.
    Returns (records_off, records_on).  records_on == records_off for non-L4.
    """
    all_off = []; all_on = []
    for row in rows_df.to_dict("records"):
        res = _dispatch(row, _g_highs, _g_lows, _g_closes, pip, cost_pips, cfg, layer)
        if layer == "L4":
            off_recs, on_recs = res
            all_off.extend(off_recs); all_on.extend(on_recs)
        else:
            all_off.extend(res)
    if layer != "L4":
        all_on = all_off
    return all_off, all_on


# ═══════════════════════════════════════════════════════════════════════════════
# Grid builders
# ═══════════════════════════════════════════════════════════════════════════════
def build_grid(layer: str) -> list[dict]:
    base = [{"window_h": wh, "sl_mult": slm, "sl_main_frac": smf,
              "tp_mode": tpm, "skip_frac": sf}
            for wh in WINDOWS for slm in SL_MULTS
            for smf in SL_MAIN_FRACS for tpm in TP_MODES
            for sf in SKIP_FRACS]   # 864 configs

    if layer == "L1":
        return base
    if layer in ("L2", "L4"):
        return [{**b, "ao_frac": aof} for b in base for aof in AO_FRACS]
    if layer == "L3":
        return [{**b, "sar_tp_frac": stp, "sar_sl_frac": ssl}
                for b in base for stp in SAR_TP_FRACS for ssl in SAR_SL_FRACS]
    raise ValueError(f"Unknown layer: {layer}")


def build_neighbor_grid(winner: dict, layer: str, rnd: int) -> list[dict]:
    def nbrs(lst, val, s):
        try:    idx = lst.index(val)
        except ValueError:
            diffs = [abs(v - val) for v in lst]; idx = diffs.index(min(diffs))
        return lst[max(0, idx - s): idx + s + 1]

    s = max(1, 3 - rnd // 3)
    wh = winner["window_h"]
    cfgs = []
    for slm in nbrs(SL_MULTS,      winner["sl_mult"],      s):
        for smf in nbrs(SL_MAIN_FRACS, winner["sl_main_frac"], s):
            for tpm in TP_MODES:
                for sf in nbrs(SKIP_FRACS, winner["skip_frac"], s):
                    base = {"window_h": wh, "sl_mult": slm, "sl_main_frac": smf,
                            "tp_mode": tpm, "skip_frac": sf}
                    if layer == "L1":
                        cfgs.append(base)
                    elif layer in ("L2", "L4"):
                        for aof in nbrs(AO_FRACS, winner.get("ao_frac", 0.5), s):
                            cfgs.append({**base, "ao_frac": aof})
                    elif layer == "L3":
                        for stp in nbrs(SAR_TP_FRACS, winner.get("sar_tp_frac", 0.7), s):
                            for ssl in nbrs(SAR_SL_FRACS, winner.get("sar_sl_frac", 0.35), s):
                                cfgs.append({**base, "sar_tp_frac": stp, "sar_sl_frac": ssl})
    # Deduplicate
    seen = set(); unique = []
    for c in cfgs:
        k = tuple(sorted(c.items()))
        if k not in seen: seen.add(k); unique.append(c)
    return unique
