from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


PAIR = "usdjpy"
START_DATE = "2014-01-01"
END_DATE = "2021-12-31"
SL_MULT = 0.2
TP_MULT = 0.4
ATR_PERIOD = 14
MAX_HOLD_HOURS = 432
WINDOW_STARTS = [0, 8, 16]

DATA_ROOT = Path(r"C:\Users\temen\5 joulu\TrailingStopLoss pivot peak trading 6 march 2026")
DAILY_PATH = DATA_ROOT / "NEW_AGENT_HOURLY_33PIPS_PACK" / "pair_datasets" / "usdjpy" / "processed_usdjpy_data_daily.csv"
M15_PATH = DATA_ROOT / "NEW_AGENT_HOURLY_33PIPS_PACK" / "pair_datasets" / "usdjpy" / "processed_usdjpy_data_15m.csv"
OUTPUT_DIR = Path(__file__).parent


def pip_size(pair: str) -> float:
    return 0.01 if pair.endswith("jpy") else 0.0001


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
    entries = 0
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

        for wh in WINDOW_STARTS:
            wstart = np.datetime64(d + pd.Timedelta(hours=wh))
            i0 = int(np.searchsorted(times, wstart, side="left"))
            if i0 >= len(times):
                continue
            if pd.Timestamp(times[i0]).date() != d.date():
                continue
            entries += 1
            end_bar = min(i0 + (MAX_HOLD_HOURS * 4), len(highs))  # 15m bars
            if end_bar <= i0:
                continue

            buy_open = True
            sell_open = True
            buy_entry = pp
            sell_entry = pp
            buy_tp = pp + tp_dist
            sell_tp = pp - tp_dist
            buy_sl = pp - sl_dist
            sell_sl = pp + sl_dist
            buy_best = pp
            sell_best = pp
            buy_exit = sell_exit = None
            buy_reason = sell_reason = None

            for j in range(i0, end_bar):
                h_ = float(highs[j])
                l_ = float(lows[j])

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
                buy_exit = float(closes[end_bar - 1]); buy_reason = "TIME_EXIT"
            if sell_open:
                sell_exit = float(closes[end_bar - 1]); sell_reason = "TIME_EXIT"

            buy_pips = ((buy_exit - buy_entry) / pip) - cost_pips
            sell_pips = ((sell_entry - sell_exit) / pip) - cost_pips
            total_buy += buy_pips
            total_sell += sell_pips
            total += buy_pips + sell_pips
            exit_counts[buy_reason] += 1
            exit_counts[sell_reason] += 1

    return {
        "pair": PAIR.upper(),
        "timeframe": "15m",
        "mode": "8h_open",
        "period": [START_DATE, END_DATE],
        "sl_mult": SL_MULT,
        "tp_mult": TP_MULT,
        "entries": entries,
        "cal_ppd": round(total / cal_days, 4),
        "buy_cal_ppd": round(total_buy / cal_days, 4),
        "sell_cal_ppd": round(total_sell / cal_days, 4),
        "total_pips": round(total, 4),
        "exit_counts": exit_counts,
    }


def main() -> None:
    r = run()
    out_path = OUTPUT_DIR / f"pivot_8h_forward_15m_{PAIR}_{START_DATE}_{END_DATE}_sl{SL_MULT}_tp{TP_MULT}.json"
    out_path.write_text(json.dumps(r, indent=2))
    print(r)
    print(f"saved={out_path}")


if __name__ == "__main__":
    main()

