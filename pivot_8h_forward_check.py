from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from bl_runner import PAIR_CFG, _load_daily, _load_hourly


PAIR = "usdjpy"
ATR_PERIOD = 14
MAX_HOLD_HOURS = 432
WINDOW_STARTS = [0, 8, 16]
OUTPUT_DIR = Path(__file__).parent


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


def run_window(start_date: str, end_date: str, sl_mult: float, tp_mult: float) -> dict:
    cfg = PAIR_CFG[PAIR]
    cost_pips = float(cfg["cost_pips"])
    pip = pip_size(PAIR)

    daily = _load_daily(Path(cfg["daily"]))
    htimes, hopes, hhighs, hlows, hcloses = _load_hourly(Path(cfg["h1"]))

    h = daily["High"].to_numpy(float)
    l = daily["Low"].to_numpy(float)
    c = daily["Close"].to_numpy(float)
    atr = atr_wilder(h, l, c, ATR_PERIOD)

    start = pd.Timestamp(start_date)
    end = pd.Timestamp(end_date)
    cal_days = (end - start).days + 1

    total = 0.0
    total_buy = 0.0
    total_sell = 0.0
    n_entries = 0
    exit_counts = {"SL": 0, "TP": 0, "BOTH_SL": 0, "TIME_EXIT": 0}

    for i in range(1, len(daily)):
        d = pd.Timestamp(daily.loc[i, "Date"])
        if d < start or d > end:
            continue
        yh = float(daily.loc[i - 1, "High"])
        yl = float(daily.loc[i - 1, "Low"])
        yc = float(daily.loc[i - 1, "Close"])
        atr_prev = float(atr[i - 1])
        if not np.isfinite(atr_prev) or atr_prev <= 0:
            continue
        pp = (yh + yl + yc) / 3.0
        sl_dist = sl_mult * atr_prev
        tp_dist = tp_mult * atr_prev
        if sl_dist <= 0 or tp_dist <= 0:
            continue

        for wh in WINDOW_STARTS:
            wstart = np.datetime64(d + pd.Timedelta(hours=wh))
            i0 = int(np.searchsorted(htimes, wstart, side="left"))
            if i0 >= len(htimes):
                continue
            if pd.Timestamp(htimes[i0]).date() != d.date():
                continue

            n_entries += 1
            end_bar = min(i0 + MAX_HOLD_HOURS, len(hhighs))
            if end_bar <= i0:
                continue

            buy_open = True
            sell_open = True
            buy_entry = pp
            sell_entry = pp
            buy_tp, sell_tp = pp + tp_dist, pp - tp_dist
            buy_sl, sell_sl = pp - sl_dist, pp + sl_dist
            buy_best, sell_best = pp, pp
            buy_exit = sell_exit = None
            buy_reason = sell_reason = None

            for j in range(i0, end_bar):
                h_ = float(hhighs[j])
                l_ = float(hlows[j])
                if buy_open:
                    buy_best = max(buy_best, h_)
                    buy_sl = max(buy_sl, buy_best - sl_dist)
                if sell_open:
                    sell_best = min(sell_best, l_)
                    sell_sl = min(sell_sl, sell_best + sl_dist)

                if buy_open:
                    hit_sl = l_ <= buy_sl
                    hit_tp = h_ >= buy_tp
                    if hit_sl and hit_tp:
                        buy_open = False; buy_exit = buy_sl; buy_reason = "BOTH_SL"
                    elif hit_sl:
                        buy_open = False; buy_exit = buy_sl; buy_reason = "SL"
                    elif hit_tp:
                        buy_open = False; buy_exit = buy_tp; buy_reason = "TP"

                if sell_open:
                    hit_sl = h_ >= sell_sl
                    hit_tp = l_ <= sell_tp
                    if hit_sl and hit_tp:
                        sell_open = False; sell_exit = sell_sl; sell_reason = "BOTH_SL"
                    elif hit_sl:
                        sell_open = False; sell_exit = sell_sl; sell_reason = "SL"
                    elif hit_tp:
                        sell_open = False; sell_exit = sell_tp; sell_reason = "TP"

                if not buy_open and not sell_open:
                    break

            if buy_open:
                buy_exit = float(hcloses[end_bar - 1]); buy_reason = "TIME_EXIT"
            if sell_open:
                sell_exit = float(hcloses[end_bar - 1]); sell_reason = "TIME_EXIT"

            buy_pips = ((buy_exit - buy_entry) / pip) - cost_pips
            sell_pips = ((sell_entry - sell_exit) / pip) - cost_pips
            total_buy += buy_pips
            total_sell += sell_pips
            total += buy_pips + sell_pips
            exit_counts[buy_reason] += 1
            exit_counts[sell_reason] += 1

    return {
        "pair": PAIR.upper(),
        "period": [start_date, end_date],
        "mode": "8h_open",
        "sl_mult": sl_mult,
        "tp_mult": tp_mult,
        "entries": n_entries,
        "cal_ppd": round(total / cal_days, 4),
        "buy_cal_ppd": round(total_buy / cal_days, 4),
        "sell_cal_ppd": round(total_sell / cal_days, 4),
        "total_pips": round(total, 4),
        "exit_counts": exit_counts,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start-date", required=True)
    ap.add_argument("--end-date", required=True)
    ap.add_argument("--sl-mult", type=float, required=True)
    ap.add_argument("--tp-mult", type=float, required=True)
    args = ap.parse_args()

    result = run_window(args.start_date, args.end_date, args.sl_mult, args.tp_mult)
    tag = f"{args.start_date}_{args.end_date}".replace(":", "-")
    out_path = OUTPUT_DIR / f"pivot_8h_forward_{PAIR}_{tag}_sl{args.sl_mult}_tp{args.tp_mult}.json"
    out_path.write_text(json.dumps(result, indent=2))

    print(result)
    print(f"saved={out_path}")


if __name__ == "__main__":
    main()

