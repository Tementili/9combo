from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


PAIR = "usdjpy"
START_DATE = "2014-01-01"
END_DATE = "2021-12-31"
SL_MULT = 0.10
TP_MULT = 0.50
ATR_PERIOD = 14
MAX_HOLD_HOURS = 432
WINDOW_STARTS = [0, 8, 16]

DATA_ROOT = Path(r"C:\Users\temen\5 joulu\TrailingStopLoss pivot peak trading 6 march 2026")
DAILY_PATH = DATA_ROOT / "NEW_AGENT_HOURLY_33PIPS_PACK" / "pair_datasets" / "usdjpy" / "processed_usdjpy_data_daily.csv"
M15_PATH = DATA_ROOT / "NEW_AGENT_HOURLY_33PIPS_PACK" / "pair_datasets" / "usdjpy" / "processed_usdjpy_data_15m.csv"
OUTPUT_DIR = Path(__file__).parent


@dataclass
class Pos:
    side: str  # BUY/SELL
    entry: float
    tp: float
    sl: float
    best_extreme: float
    open_bar: int
    active_from_bar: int
    open: bool = True
    exit_bar: int = -1
    exit_px: float = np.nan
    reason: str = ""


def load_daily(path: Path) -> pd.DataFrame:
    d = pd.read_csv(path, low_memory=False)
    d["Date"] = pd.to_datetime(d["Date"], errors="coerce")
    for c in ["Open", "High", "Low", "Close"]:
        d[c] = pd.to_numeric(d[c], errors="coerce")
    return d.dropna(subset=["Date", "Open", "High", "Low", "Close"]).sort_values("Date").reset_index(drop=True)


def load_m15(path: Path):
    m = pd.read_csv(path, usecols=["Datetime", "Open", "High", "Low", "Close"], low_memory=False)
    m["Datetime"] = pd.to_datetime(m["Datetime"], errors="coerce")
    for c in ["Open", "High", "Low", "Close"]:
        m[c] = pd.to_numeric(m[c], errors="coerce")
    m = m.dropna(subset=["Datetime", "Open", "High", "Low", "Close"]).sort_values("Datetime").reset_index(drop=True)
    return (
        m["Datetime"].to_numpy(),
        m["Open"].to_numpy(float),
        m["High"].to_numpy(float),
        m["Low"].to_numpy(float),
        m["Close"].to_numpy(float),
    )


def pip_size(pair: str) -> float:
    return 0.01 if pair.endswith("jpy") else 0.0001


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


def new_buy(entry: float, sl_dist: float, tp_dist: float, bar: int, active_from: int) -> Pos:
    return Pos(
        side="BUY",
        entry=entry,
        tp=entry + tp_dist,
        sl=entry - sl_dist,
        best_extreme=entry,
        open_bar=bar,
        active_from_bar=active_from,
    )


def new_sell(entry: float, sl_dist: float, tp_dist: float, bar: int, active_from: int) -> Pos:
    return Pos(
        side="SELL",
        entry=entry,
        tp=entry - tp_dist,
        sl=entry + sl_dist,
        best_extreme=entry,
        open_bar=bar,
        active_from_bar=active_from,
    )


def close_pos(p: Pos, bar: int, px: float, reason: str) -> None:
    p.open = False
    p.exit_bar = bar
    p.exit_px = float(px)
    p.reason = reason


def run() -> dict:
    daily = load_daily(DAILY_PATH)
    times, opens, highs, lows, closes = load_m15(M15_PATH)
    pip = pip_size(PAIR)
    cost_pips = 2.0
    atr = atr_wilder(
        daily["High"].to_numpy(float),
        daily["Low"].to_numpy(float),
        daily["Close"].to_numpy(float),
        ATR_PERIOD,
    )

    start = pd.Timestamp(START_DATE)
    end = pd.Timestamp(END_DATE)
    cal_days = (end - start).days + 1

    total = 0.0
    total_buy = 0.0
    total_sell = 0.0
    entries_initial = 0
    reentry_pairs = 0
    all_closed: list[Pos] = []
    exit_counts = {"SL": 0, "TP": 0, "BOTH_SL": 0, "TIME_EXIT": 0}

    for i in range(1, len(daily)):
        d = pd.Timestamp(daily.loc[i, "Date"])
        if d < start or d > end:
            continue
        yh = float(daily.loc[i - 1, "High"])
        yl = float(daily.loc[i - 1, "Low"])
        yc = float(daily.loc[i - 1, "Close"])
        atr_prev = float(atr[i - 1]) if np.isfinite(atr[i - 1]) else np.nan
        if not np.isfinite(atr_prev) or atr_prev <= 0:
            continue

        pp = (yh + yl + yc) / 3.0
        sl_dist = SL_MULT * atr_prev
        tp_dist = TP_MULT * atr_prev

        for wh in WINDOW_STARTS:
            wstart = np.datetime64(d + pd.Timedelta(hours=wh))
            i0 = int(np.searchsorted(times, wstart, side="left"))
            if i0 >= len(times):
                continue
            if pd.Timestamp(times[i0]).date() != d.date():
                continue

            end_bar = min(i0 + (MAX_HOLD_HOURS * 4), len(highs))
            if end_bar <= i0:
                continue

            # Initial straddle at pivot
            open_positions: list[Pos] = [
                new_buy(pp, sl_dist, tp_dist, i0, i0),
                new_sell(pp, sl_dist, tp_dist, i0, i0),
            ]
            entries_initial += 1

            for j in range(i0, end_bar):
                h_ = float(highs[j]); l_ = float(lows[j])
                tp_hits_this_bar: list[float] = []

                for p in open_positions:
                    if not p.open or j < p.active_from_bar:
                        continue

                    # trailing SL
                    if p.side == "BUY":
                        p.best_extreme = max(p.best_extreme, h_)
                        p.sl = max(p.sl, p.best_extreme - sl_dist)
                        hit_sl = l_ <= p.sl
                        hit_tp = h_ >= p.tp
                    else:
                        p.best_extreme = min(p.best_extreme, l_)
                        p.sl = min(p.sl, p.best_extreme + sl_dist)
                        hit_sl = h_ >= p.sl
                        hit_tp = l_ <= p.tp

                    if hit_sl and hit_tp:
                        close_pos(p, j, p.sl, "BOTH_SL")
                    elif hit_sl:
                        close_pos(p, j, p.sl, "SL")
                    elif hit_tp:
                        close_pos(p, j, p.tp, "TP")
                        tp_hits_this_bar.append(p.tp)

                # Re-entry rule: for each TP hit, open new pair at same spot.
                # Activate from next bar to avoid same-bar infinite recursion under OHLC ambiguity.
                for tp_px in tp_hits_this_bar:
                    open_positions.append(new_buy(tp_px, sl_dist, tp_dist, j, j + 1))
                    open_positions.append(new_sell(tp_px, sl_dist, tp_dist, j, j + 1))
                    reentry_pairs += 1

                if all((not p.open) for p in open_positions):
                    break

            # Time-exit any open positions
            last_close = float(closes[end_bar - 1])
            for p in open_positions:
                if p.open:
                    close_pos(p, end_bar - 1, last_close, "TIME_EXIT")
                all_closed.append(p)

    for p in all_closed:
        if p.side == "BUY":
            pips = ((p.exit_px - p.entry) / pip) - cost_pips
            total_buy += pips
        else:
            pips = ((p.entry - p.exit_px) / pip) - cost_pips
            total_sell += pips
        total += pips
        exit_counts[p.reason] = exit_counts.get(p.reason, 0) + 1

    out = {
        "pair": PAIR.upper(),
        "period": [START_DATE, END_DATE],
        "timeframe": "15m",
        "mode": "8h_open_tp_reentry_chain",
        "sl_mult": SL_MULT,
        "tp_mult": TP_MULT,
        "initial_entries": entries_initial,
        "reentry_pairs": reentry_pairs,
        "total_legs_closed": len(all_closed),
        "cal_ppd": round(total / cal_days, 4),
        "buy_cal_ppd": round(total_buy / cal_days, 4),
        "sell_cal_ppd": round(total_sell / cal_days, 4),
        "total_pips": round(total, 4),
        "exit_counts": exit_counts,
    }
    out_path = OUTPUT_DIR / f"pivot_8h_tp_reentry_chain_{PAIR}_{START_DATE}_{END_DATE}.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(out)
    print(f"saved={out_path}")
    return out


if __name__ == "__main__":
    run()

