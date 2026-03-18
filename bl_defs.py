"""
bl_defs.py  -  Baseline Discovery Step 1: Main Leg Only, ATR14
==============================================================
NO addon. NO SAR. NO BE law. NO extra indicators.
ATR14 (Wilder smoothed) drives SL sizing. Main leg only.

Window UTC starts:
  24h -> 00:00 UTC (midnight, full-day signal)
  12h -> 12:00 UTC (afternoon session open)
   8h -> 08:00 UTC (London session open)

Split (LOCKED):
  Train:  2010-01-01 .. 2016-12-31
  Adjust: 2017-01-01 .. 2021-12-31
  OOS:    2022-01-01 .. 2026-03-06
  max_hold_hours: 432 (hard cap)
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path

RUNNER_VERSION = "v1.0.0-bl-step1"
MODULE_TAG     = "baseline_discovery_v1"

SPLIT = {
    "train_start":    "2010-01-01",
    "train_end":      "2016-12-31",
    "adjust_start":   "2017-01-01",
    "adjust_end":     "2021-12-31",
    "oos_start":      "2022-01-01",
    "oos_end":        "2026-03-06",
    "max_hold_hours": 432,
}

MAIN_SIZE  = 2.0   # lots, fixed
WORKERS    = 19
ATR_PERIOD = 14

# Fixed UTC start hour for each window type (LOCKED)
WINDOW_UTC_START = {24: 0, 12: 12, 8: 8}

# Full parameter grid (LOCKED per spec)
SL_MULTS      = [1.0, 1.5, 2.0, 2.5, 3.0, 3.5]
SL_MAIN_FRACS = [0.3, 0.5, 0.7]
TP_MODES      = ["TP_EDGE", "TP_ATR_1.5", "TP_ATR_2.0", "TP_PP"]
SKIP_FRACS    = [0.25, 0.30, 0.35, 0.40]
WINDOWS       = [24, 12, 8]

HONING_STOP_DELTA = 0.20   # stop when Adjust improvement < this over 2 consecutive rounds
MAX_ROUNDS        = 10
MIN_ROUNDS        = 3


# ── ATR (Wilder smoothed, identical formula to all prior phases) ───────────────
def _atr_series(h: np.ndarray, l: np.ndarray, c: np.ndarray,
                period: int = ATR_PERIOD) -> np.ndarray:
    n = len(h)
    tr = np.empty(n, dtype=float)
    tr[0] = h[0] - l[0]
    for i in range(1, n):
        tr[i] = max(h[i] - l[i], abs(h[i] - c[i-1]), abs(l[i] - c[i-1]))
    atr = np.full(n, np.nan)
    if period <= n:
        atr[period - 1] = float(np.mean(tr[:period]))
        k = 1.0 / period
        for i in range(period, n):
            atr[i] = atr[i-1] * (1.0 - k) + tr[i] * k
    return atr


# ── Row builder: all windows, causal features only ────────────────────────────
def _build_rows_all_windows(
    daily: pd.DataFrame,
    times: np.ndarray,        # 60m datetime64 array
    opens: np.ndarray,        # 60m open prices
    start: str,
    end: str,
    windows: list = WINDOWS,
) -> pd.DataFrame:
    """
    Build one row per (trading-day, window) with strictly causal features.

    Columns:
      date, trade_date_str, window_h, window_open, window_bar_idx,
      YH, YL, YC, ATR14, PP, YR
    All from prior calendar day (causal). ATR14 from prior daily close.
    """
    h_ = daily["High"].values.astype(float)
    l_ = daily["Low"].values.astype(float)
    c_ = daily["Close"].values.astype(float)

    atr14 = _atr_series(h_, l_, c_, ATR_PERIOD)

    n = len(daily)
    # Strictly causal shift: yesterday's values for trade on day i
    yh = np.concatenate([[np.nan], h_[:-1]])
    yl = np.concatenate([[np.nan], l_[:-1]])
    yc = np.concatenate([[np.nan], c_[:-1]])
    # ATR14 prior day (causal: we know atr14[i-1] before day i opens)
    atr_prior = np.concatenate([[np.nan], atr14[:-1]])

    dates   = pd.to_datetime(daily["Date"].values)
    ts_start = pd.Timestamp(start)
    ts_end   = pd.Timestamp(end)

    rows = []
    for i in range(n):
        d = dates[i]
        if d < ts_start or d > ts_end:
            continue

        atr = atr_prior[i]
        if (np.isnan(atr) or np.isnan(yh[i])
                or np.isnan(yl[i]) or np.isnan(yc[i])):
            continue

        for wh in windows:
            utc_h = WINDOW_UTC_START[wh]
            # Find first 60m bar at/after d + utc_h hours (UTC)
            t0 = np.datetime64(d + pd.Timedelta(hours=utc_h))
            idx = int(np.searchsorted(times, t0))
            if idx >= len(times):
                continue
            # Verify bar is on the expected calendar day
            bar_day = pd.Timestamp(times[idx]).date()
            if bar_day != d.date():
                continue   # no data for this window on this day

            rows.append({
                "date":           d,
                "trade_date_str": str(d.date()),
                "window_h":       wh,
                "window_open":    float(opens[idx]),
                "window_bar_idx": idx,
                "YH":    float(yh[i]),
                "YL":    float(yl[i]),
                "YC":    float(yc[i]),
                "ATR14": float(atr),
                "PP":    (float(yh[i]) + float(yl[i]) + float(yc[i])) / 3.0,
                "YR":    float(yh[i]) - float(yl[i]),
            })

    return pd.DataFrame(rows).reset_index(drop=True)


# ── TP price helper (strictly causal) ────────────────────────────────────────
def _tp_price(tp_mode: str, direction: str,
              entry: float, yh: float, yl: float,
              atr: float, pp: float) -> float:
    if direction == "BUY":
        if tp_mode == "TP_EDGE":    return float(yh)
        if tp_mode == "TP_ATR_1.5": return entry + 1.5 * atr
        if tp_mode == "TP_ATR_2.0": return entry + 2.0 * atr
        if tp_mode == "TP_PP":      return float(pp)
    else:  # SELL
        if tp_mode == "TP_EDGE":    return float(yl)
        if tp_mode == "TP_ATR_1.5": return entry - 1.5 * atr
        if tp_mode == "TP_ATR_2.0": return entry - 2.0 * atr
        if tp_mode == "TP_PP":      return float(pp)
    return float("nan")


# ── Core backtest engine ───────────────────────────────────────────────────────
def _run_backtest_bl(
    rows: pd.DataFrame,
    highs:   np.ndarray,
    lows:    np.ndarray,
    closes:  np.ndarray,
    pip:     float,
    cost_pips: float,
    cfg:     dict,
) -> list:
    """
    Main-only backtest. No addon, no SAR, no BE law.

    cfg: {sl_mult, sl_main_frac, tp_mode, skip_frac, window_h}

    Intrabar convention (inherited from all prior phases):
      On same-bar SL+TP: SL wins (BOTH_SL).
      No intrabar order knowledge beyond this.

    Returns list of dicts: {date, direction, total, exit_reason}
    total = net pips for the full position (MAIN_SIZE lots).
    """
    sf    = float(cfg["skip_frac"])
    slm   = float(cfg["sl_mult"])
    smf   = float(cfg["sl_main_frac"])
    tpm   = str(cfg["tp_mode"])
    wh    = int(cfg["window_h"])
    mh    = int(SPLIT["max_hold_hours"])
    size  = MAIN_SIZE
    n_bars = len(highs)

    # Filter rows for this window_h
    w_rows = rows[rows["window_h"] == wh] if "window_h" in rows.columns else rows

    results = []

    for _, row in w_rows.iterrows():
        atr     = float(row["ATR14"])
        yr      = float(row["YR"])
        wo      = float(row["window_open"])
        yh      = float(row["YH"])
        yl      = float(row["YL"])
        pp      = float(row["PP"])
        bar_idx = int(row["window_bar_idx"])
        date_s  = str(row["trade_date_str"])

        skip_rec = {"date": date_s, "direction": "SKIP",
                    "total": 0.0, "exit_reason": "SKIP"}

        # Degenerate ATR guard
        if atr < pip * 5:
            results.append(dict(skip_rec, reason="degenerate_atr"))
            continue

        # Skip-frac guard: range too compressed
        if yr < sf * atr:
            results.append(dict(skip_rec, reason="skip_frac"))
            continue

        # 432h guard: need mh bars from entry (data-end safety)
        if bar_idx + mh > n_bars:
            results.append(dict(skip_rec, reason="432h_guard"))
            continue

        # Direction rule (LOCKED)
        direction = "SELL" if wo > pp else "BUY"

        # Entry and SL
        entry   = wo
        sl_dist = atr * slm * smf

        if direction == "BUY":
            sl = entry - sl_dist
        else:
            sl = entry + sl_dist

        # TP
        tp = _tp_price(tpm, direction, entry, yh, yl, atr, pp)

        # TP must be on the correct side of entry — else skip
        if direction == "BUY"  and tp <= entry:
            results.append(dict(skip_rec, reason="tp_wrong_side"))
            continue
        if direction == "SELL" and tp >= entry:
            results.append(dict(skip_rec, reason="tp_wrong_side"))
            continue

        # Simulate through 60m bars (intrabar: check H/L each bar)
        end_bar     = min(bar_idx + mh, n_bars)
        exit_px     = closes[end_bar - 1]
        exit_reason = "TIME_EXIT"

        for j in range(bar_idx, end_bar):
            h_ = highs[j]
            l_ = lows[j]
            hsl = (l_ <= sl) if direction == "BUY" else (h_ >= sl)
            htp = (h_ >= tp) if direction == "BUY" else (l_ <= tp)

            if hsl and htp:
                exit_px = sl; exit_reason = "BOTH_SL"; break
            if hsl:
                exit_px = sl; exit_reason = "SL";      break
            if htp:
                exit_px = tp; exit_reason = "TP";      break

        # P&L
        if direction == "BUY":
            raw = (exit_px - entry) * size - cost_pips * pip * size
        else:
            raw = (entry - exit_px) * size - cost_pips * pip * size
        net = raw / pip   # total pips for position

        results.append({
            "date":        date_s,
            "direction":   direction,
            "total":       net,
            "exit_reason": exit_reason,
        })

    return results


# ── Summary statistics ─────────────────────────────────────────────────────────
def _summarize_bl(results: list, n_cal_days: int) -> dict:
    triggered = [r for r in results if r.get("direction") != "SKIP"]
    if not triggered:
        return {
            "cal_ppd": 0.0, "win_rate": 0.0, "avg_pips_per_trade": 0.0,
            "time_exit_rate": 0.0, "total_trades": 0, "n_cal_days": n_cal_days,
            "best_day_pips": 0.0, "worst_day_pips": 0.0,
        }
    tots = [r["total"] for r in triggered]
    n = len(triggered)
    wins  = sum(1 for x in tots if x > 0)
    te    = sum(1 for r in triggered if r.get("exit_reason") == "TIME_EXIT")
    return {
        "cal_ppd":            round(sum(tots) / max(n_cal_days, 1), 4),
        "win_rate":           round(wins / n, 4),
        "avg_pips_per_trade": round(sum(tots) / n, 4),
        "time_exit_rate":     round(te / n, 4),
        "total_trades":       n,
        "n_cal_days":         n_cal_days,
        "best_day_pips":      round(max(tots), 2),
        "worst_day_pips":     round(min(tots), 2),
    }


# ── Pool worker (Windows-spawn safe) ──────────────────────────────────────────
_g_highs  = None
_g_lows   = None
_g_closes = None


def _pool_init_bl(highs, lows, closes):
    global _g_highs, _g_lows, _g_closes
    _g_highs  = highs
    _g_lows   = lows
    _g_closes = closes


def _task_bl(task):
    """task = (rows_records, n_cal_days, pip, cost_pips, cfg)"""
    rows_records, n_cal_days, pip, cost_pips, cfg = task
    df = pd.DataFrame(rows_records)
    results = _run_backtest_bl(df, _g_highs, _g_lows, _g_closes, pip, cost_pips, cfg)
    return cfg, _summarize_bl(results, n_cal_days)


# ── Calendar day counter ───────────────────────────────────────────────────────
def _n_cal_days(start: str, end: str) -> int:
    return (pd.Timestamp(end) - pd.Timestamp(start)).days + 1
