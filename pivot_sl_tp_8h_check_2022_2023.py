from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from bl_runner import PAIR_CFG, _load_daily, _load_hourly


PAIR = "usdjpy"
START_DATE = "2022-01-01"
END_DATE = "2023-12-31"
ATR_PERIOD = 14
MAX_HOLD_HOURS = 432

# Best from previous daily-mode sweep
SL_MULT = 0.80
TP_MULT = 0.20

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


def run_mode(mode: str) -> dict:
    cfg = PAIR_CFG[PAIR]
    cost_pips = float(cfg["cost_pips"])
    pip = pip_size(PAIR)
    daily = _load_daily(Path(cfg["daily"]))
    htimes, hopes, hhighs, hlows, hcloses = _load_hourly(Path(cfg["h1"]))

    h = daily["High"].to_numpy(float)
    l = daily["Low"].to_numpy(float)
    c = daily["Close"].to_numpy(float)
    atr = atr_wilder(h, l, c, ATR_PERIOD)

    start = pd.Timestamp(START_DATE)
    end = pd.Timestamp(END_DATE)
    cal_days = (end - start).days + 1

    total = 0.0
    total_buy = 0.0
    total_sell = 0.0
    n_legs = 0
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
        sl_dist = SL_MULT * atr_prev
        tp_dist = TP_MULT * atr_prev

        window_starts = [0] if mode == "daily_touch" else [0, 8, 16]
        for hour in window_starts:
            wstart = np.datetime64(d + pd.Timedelta(hours=hour))
            wend = np.datetime64(d + pd.Timedelta(hours=hour + 8)) if mode == "w8_open" else np.datetime64(d + pd.Timedelta(hours=24))

            i0 = int(np.searchsorted(htimes, wstart, side="left"))
            i1 = int(np.searchsorted(htimes, wend, side="left"))
            if i0 >= len(htimes) or i1 <= i0:
                continue

            # Entry rule:
            # - daily_touch: first touch of PP in the day horizon
            # - w8_open: open immediately at first bar open of each 8h block
            if mode == "daily_touch":
                entry_bar = None
                iend = min(i0 + MAX_HOLD_HOURS, len(hhighs))
                for j in range(i0, iend):
                    if float(hlows[j]) <= pp <= float(hhighs[j]):
                        entry_bar = j
                        break
                if entry_bar is None:
                    continue
            else:
                entry_bar = i0

            n_entries += 1
            day_end = min(entry_bar + MAX_HOLD_HOURS, len(hhighs))

            buy_open = True
            sell_open = True
            buy_entry = pp
            sell_entry = pp
            buy_tp, sell_tp = pp + tp_dist, pp - tp_dist
            buy_sl, sell_sl = pp - sl_dist, pp + sl_dist
            buy_best, sell_best = pp, pp
            buy_exit = sell_exit = None
            buy_reason = sell_reason = None

            for j in range(entry_bar, day_end):
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
                buy_exit = float(hcloses[day_end - 1]); buy_reason = "TIME_EXIT"
            if sell_open:
                sell_exit = float(hcloses[day_end - 1]); sell_reason = "TIME_EXIT"

            buy_pips = ((buy_exit - buy_entry) / pip) - cost_pips
            sell_pips = ((sell_entry - sell_exit) / pip) - cost_pips
            total_buy += buy_pips
            total_sell += sell_pips
            total += buy_pips + sell_pips
            n_legs += 2
            exit_counts[buy_reason] += 1
            exit_counts[sell_reason] += 1

    return {
        "mode": mode,
        "pair": PAIR.upper(),
        "period": [START_DATE, END_DATE],
        "sl_mult": SL_MULT,
        "tp_mult": TP_MULT,
        "entries": n_entries,
        "legs": n_legs,
        "cal_ppd": round(total / cal_days, 4),
        "buy_cal_ppd": round(total_buy / cal_days, 4),
        "sell_cal_ppd": round(total_sell / cal_days, 4),
        "total_pips": round(total, 4),
        "exit_counts": exit_counts,
    }


def main() -> None:
    daily = run_mode("daily_touch")
    w8 = run_mode("w8_open")
    out = {"daily_touch": daily, "w8_open": w8}
    out_path = OUTPUT_DIR / f"pivot_sl_tp_8h_check_{PAIR}_{START_DATE}_{END_DATE}.json"
    out_path.write_text(json.dumps(out, indent=2))

    print("Comparison (same SL/TP params)")
    print(f"daily_touch cal_ppd={daily['cal_ppd']:+.4f} entries={daily['entries']}")
    print(f"w8_open    cal_ppd={w8['cal_ppd']:+.4f} entries={w8['entries']}")
    print(f"saved={out_path}")


if __name__ == "__main__":
    main()

