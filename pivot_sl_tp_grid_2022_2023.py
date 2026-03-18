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

# Search SL/trailing distance and TP distance independently (in ATR units)
SL_MULT_GRID = [0.40, 0.50, 0.60, 0.65, 0.70, 0.80, 0.90, 1.00]
TP_MULT_GRID = [0.15, 0.20, 0.2275, 0.25, 0.30, 0.35, 0.40, 0.50]

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


def first_bar_index_per_day(times) -> dict:
    out = {}
    for i, ts in enumerate(times):
        d = pd.Timestamp(ts).date()
        if d not in out:
            out[d] = i
    return out


def run_one(sl_mult: float, tp_mult: float) -> dict:
    cfg = PAIR_CFG[PAIR]
    cost_pips = float(cfg["cost_pips"])
    pip = pip_size(PAIR)
    daily = _load_daily(Path(cfg["daily"]))
    times, opens, highs, lows, closes = _load_hourly(Path(cfg["h1"]))
    first_bar = first_bar_index_per_day(times)

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
    traded_days = 0
    exit_counts = {"SL": 0, "TP": 0, "BOTH_SL": 0, "TIME_EXIT": 0}

    for i in range(1, len(daily)):
        d = pd.Timestamp(daily.loc[i, "Date"])
        if d < start or d > end or d.date() not in first_bar:
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

        day_start = first_bar[d.date()]
        day_end = min(day_start + MAX_HOLD_HOURS, len(highs))
        if day_end <= day_start:
            continue

        entry_bar = None
        for j in range(day_start, day_end):
            if float(lows[j]) <= pp <= float(highs[j]):
                entry_bar = j
                break
        if entry_bar is None:
            continue

        traded_days += 1
        buy_open, sell_open = True, True
        buy_entry, sell_entry = pp, pp
        buy_tp, sell_tp = pp + tp_dist, pp - tp_dist
        buy_sl, sell_sl = pp - sl_dist, pp + sl_dist
        buy_best, sell_best = pp, pp
        buy_exit = sell_exit = None
        buy_reason = sell_reason = None

        for j in range(entry_bar, day_end):
            h_ = float(highs[j])
            l_ = float(lows[j])

            # trailing keeps SL distance fixed from the best extreme
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
                    buy_open = False
                    buy_exit = buy_sl
                    buy_reason = "BOTH_SL"
                elif hit_sl:
                    buy_open = False
                    buy_exit = buy_sl
                    buy_reason = "SL"
                elif hit_tp:
                    buy_open = False
                    buy_exit = buy_tp
                    buy_reason = "TP"

            if sell_open:
                hit_sl = h_ >= sell_sl
                hit_tp = l_ <= sell_tp
                if hit_sl and hit_tp:
                    sell_open = False
                    sell_exit = sell_sl
                    sell_reason = "BOTH_SL"
                elif hit_sl:
                    sell_open = False
                    sell_exit = sell_sl
                    sell_reason = "SL"
                elif hit_tp:
                    sell_open = False
                    sell_exit = sell_tp
                    sell_reason = "TP"

            if not buy_open and not sell_open:
                break

        if buy_open:
            buy_exit = float(closes[day_end - 1])
            buy_reason = "TIME_EXIT"
        if sell_open:
            sell_exit = float(closes[day_end - 1])
            sell_reason = "TIME_EXIT"

        buy_pips = ((buy_exit - buy_entry) / pip) - cost_pips
        sell_pips = ((sell_entry - sell_exit) / pip) - cost_pips
        total_buy += buy_pips
        total_sell += sell_pips
        total += buy_pips + sell_pips
        exit_counts[buy_reason] = exit_counts.get(buy_reason, 0) + 1
        exit_counts[sell_reason] = exit_counts.get(sell_reason, 0) + 1

    return {
        "sl_mult": sl_mult,
        "tp_mult": tp_mult,
        "traded_days": traded_days,
        "cal_ppd": round(total / cal_days, 4),
        "buy_cal_ppd": round(total_buy / cal_days, 4),
        "sell_cal_ppd": round(total_sell / cal_days, 4),
        "total_pips": round(total, 4),
        "exit_counts": exit_counts,
    }


def main() -> None:
    rows = []
    for slm in SL_MULT_GRID:
        for tpm in TP_MULT_GRID:
            r = run_one(slm, tpm)
            rows.append(r)
            print(f"sl={slm:.4f} tp={tpm:.4f} -> cal_ppd={r['cal_ppd']:+.4f}")

    df = pd.DataFrame(rows).sort_values("cal_ppd", ascending=False).reset_index(drop=True)
    best = df.iloc[0].to_dict()
    out = {
        "pair": PAIR.upper(),
        "period": [START_DATE, END_DATE],
        "description": "Independent sweep of SL/trailing ATR multiple vs TP ATR multiple.",
        "results": df.to_dict(orient="records"),
        "best": best,
    }
    out_path = OUTPUT_DIR / f"pivot_sl_tp_grid_{PAIR}_{START_DATE}_{END_DATE}.json"
    out_path.write_text(json.dumps(out, indent=2))
    print("\nBEST:")
    print(best)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()

