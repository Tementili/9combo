from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

PAIR = "usdjpy"
START_DATE = "2014-01-01"
END_DATE = "2021-12-31"
ATR_PERIOD = 14
MAX_HOLD_HOURS = 432
SPREAD_PIPS = 1.0  # user-confirmed cost model

# Optimize TP and trailing-stop distances (both ATR-scaled).
SL_GRID = [0.08, 0.10, 0.12, 0.16, 0.20, 0.30, 0.40, 0.50]
TP_GRID = [0.08, 0.10, 0.12, 0.16, 0.20, 0.24, 0.30, 0.40, 0.55, 0.70, 1.00]

ROOT = Path(r"C:\Users\temen\5 joulu\TrailingStopLoss pivot peak trading 6 march 2026")
DAILY_PATH = ROOT / "NEW_AGENT_HOURLY_33PIPS_PACK" / "pair_datasets" / "usdjpy" / "processed_usdjpy_data_daily.csv"
M15_PATH = ROOT / "NEW_AGENT_HOURLY_33PIPS_PACK" / "pair_datasets" / "usdjpy" / "processed_usdjpy_data_15m.csv"
OUTPUT_DIR = Path(__file__).parent


def load_daily(path: Path) -> pd.DataFrame:
    d = pd.read_csv(path, low_memory=False)
    d["Date"] = pd.to_datetime(d["Date"], errors="coerce")
    for c in ["Open", "High", "Low", "Close"]:
        d[c] = pd.to_numeric(d[c], errors="coerce")
    return d.dropna(subset=["Date", "Open", "High", "Low", "Close"]).sort_values("Date").reset_index(drop=True)


def load_m15(path: Path) -> pd.DataFrame:
    m = pd.read_csv(path, usecols=["Datetime", "Open", "High", "Low", "Close"], low_memory=False)
    m["Datetime"] = pd.to_datetime(m["Datetime"], errors="coerce")
    for c in ["Open", "High", "Low", "Close"]:
        m[c] = pd.to_numeric(m[c], errors="coerce")
    return m.dropna(subset=["Datetime", "Open", "High", "Low", "Close"]).sort_values("Datetime").reset_index(drop=True)


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


def pip_size(pair: str) -> float:
    return 0.01 if pair.endswith("jpy") else 0.0001


def find_first_touch(pp: float, highs: np.ndarray, lows: np.ndarray) -> int | None:
    for i in range(len(highs)):
        if lows[i] <= pp <= highs[i]:
            return i
    return None


def run_one(
    by_day: dict,
    m15: pd.DataFrame,
    sl_mult: float,
    tp_mult: float,
    pip: float,
) -> dict:
    start = pd.Timestamp(START_DATE)
    end = pd.Timestamp(END_DATE)
    cal_days = int((end - start).days + 1)

    total = 0.0
    buy_total = 0.0
    sell_total = 0.0
    entries = 0
    no_touch_days = 0
    legs_closed = 0

    for dkey, info in by_day.items():
        day_ts = pd.Timestamp(dkey)
        if day_ts < start or day_ts > end:
            continue

        pp = info["pp"]
        atr_prev = info["atr_prev"]
        if not np.isfinite(atr_prev) or atr_prev <= 0:
            continue
        sl_dist = sl_mult * atr_prev
        tp_dist = tp_mult * atr_prev

        day_df = m15[m15["dkey"] == dkey]
        if day_df.empty:
            continue

        highs = day_df["High"].to_numpy(float)
        lows = day_df["Low"].to_numpy(float)
        closes = day_df["Close"].to_numpy(float)
        times = day_df["Datetime"].to_numpy()

        touch_i = find_first_touch(pp, highs, lows)
        if touch_i is None:
            no_touch_days += 1
            continue

        # Real touch fill at pivot, then start from next bar.
        start_i = touch_i + 1
        if start_i >= len(day_df):
            continue

        entries += 1
        buy_entry = pp
        sell_entry = pp
        buy_tp = pp + tp_dist
        sell_tp = pp - tp_dist
        buy_sl = pp - sl_dist
        sell_sl = pp + sl_dist
        buy_best = pp
        sell_best = pp
        buy_open = True
        sell_open = True

        end_time = pd.Timestamp(times[start_i]) + pd.Timedelta(hours=MAX_HOLD_HOURS)

        for i in range(start_i, len(day_df)):
            if pd.Timestamp(times[i]) > end_time:
                break
            h = highs[i]
            l = lows[i]

            if buy_open:
                buy_best = max(buy_best, h)
                buy_sl = max(buy_sl, buy_best - sl_dist)
                hit_sl = l <= buy_sl
                hit_tp = h >= buy_tp
                if hit_sl and hit_tp:
                    px = buy_sl
                    pips = ((px - buy_entry) / pip) - SPREAD_PIPS
                    buy_total += pips
                    total += pips
                    buy_open = False
                    legs_closed += 1
                elif hit_sl:
                    px = buy_sl
                    pips = ((px - buy_entry) / pip) - SPREAD_PIPS
                    buy_total += pips
                    total += pips
                    buy_open = False
                    legs_closed += 1
                elif hit_tp:
                    px = buy_tp
                    pips = ((px - buy_entry) / pip) - SPREAD_PIPS
                    buy_total += pips
                    total += pips
                    buy_open = False
                    legs_closed += 1

            if sell_open:
                sell_best = min(sell_best, l)
                sell_sl = min(sell_sl, sell_best + sl_dist)
                hit_sl = h >= sell_sl
                hit_tp = l <= sell_tp
                if hit_sl and hit_tp:
                    px = sell_sl
                    pips = ((sell_entry - px) / pip) - SPREAD_PIPS
                    sell_total += pips
                    total += pips
                    sell_open = False
                    legs_closed += 1
                elif hit_sl:
                    px = sell_sl
                    pips = ((sell_entry - px) / pip) - SPREAD_PIPS
                    sell_total += pips
                    total += pips
                    sell_open = False
                    legs_closed += 1
                elif hit_tp:
                    px = sell_tp
                    pips = ((sell_entry - px) / pip) - SPREAD_PIPS
                    sell_total += pips
                    total += pips
                    sell_open = False
                    legs_closed += 1

            if not buy_open and not sell_open:
                break

        # EOD close if still open.
        last_c = float(closes[-1])
        if buy_open:
            pips = ((last_c - buy_entry) / pip) - SPREAD_PIPS
            buy_total += pips
            total += pips
            legs_closed += 1
        if sell_open:
            pips = ((sell_entry - last_c) / pip) - SPREAD_PIPS
            sell_total += pips
            total += pips
            legs_closed += 1

    return {
        "sl_mult": sl_mult,
        "tp_mult": tp_mult,
        "cal_ppd": round(total / max(cal_days, 1), 4),
        "buy_cal_ppd": round(buy_total / max(cal_days, 1), 4),
        "sell_cal_ppd": round(sell_total / max(cal_days, 1), 4),
        "total_pips": round(total, 4),
        "initial_entries": int(entries),
        "no_touch_days": int(no_touch_days),
        "legs_closed": int(legs_closed),
    }


def main() -> None:
    daily = load_daily(DAILY_PATH)
    m15 = load_m15(M15_PATH)
    m15["dkey"] = m15["Datetime"].dt.normalize()
    pip = pip_size(PAIR)

    atr = atr_wilder(
        daily["High"].to_numpy(float),
        daily["Low"].to_numpy(float),
        daily["Close"].to_numpy(float),
        ATR_PERIOD,
    )

    by_day = {}
    for i in range(1, len(daily)):
        d = pd.Timestamp(daily.loc[i, "Date"]).normalize()
        atr_prev = float(atr[i - 1]) if np.isfinite(atr[i - 1]) else np.nan
        if not np.isfinite(atr_prev) or atr_prev <= 0:
            continue
        yh = float(daily.loc[i - 1, "High"])
        yl = float(daily.loc[i - 1, "Low"])
        yc = float(daily.loc[i - 1, "Close"])
        pp = (yh + yl + yc) / 3.0
        by_day[d] = {"pp": pp, "atr_prev": atr_prev}

    rows = []
    for sl in SL_GRID:
        for tp in TP_GRID:
            r = run_one(by_day, m15, sl, tp, pip)
            rows.append(r)
            print(f"sl={sl:.2f} tp={tp:.2f} -> cal_ppd={r['cal_ppd']:+.4f}")

    df = pd.DataFrame(rows).sort_values("cal_ppd", ascending=False).reset_index(drop=True)
    best = df.iloc[0].to_dict()
    out = {
        "pair": PAIR.upper(),
        "period": [START_DATE, END_DATE],
        "timeframe": "15m",
        "mode": "daily_pivot_real_touch_straddle_tp_trailing",
        "fill_rule": "open buy+sell only when daily pivot is actually touched; activate next bar",
        "atr_source": "previous daily ATR14",
        "spread_pips_per_leg": SPREAD_PIPS,
        "sl_grid": SL_GRID,
        "tp_grid": TP_GRID,
        "best": best,
        "top15": df.head(15).to_dict(orient="records"),
    }

    out_json = OUTPUT_DIR / f"daily_pivot_real_touch_optimize_1pip_{PAIR}_{START_DATE}_{END_DATE}.json"
    out_csv = OUTPUT_DIR / f"daily_pivot_real_touch_optimize_1pip_{PAIR}_{START_DATE}_{END_DATE}.csv"
    out_json.write_text(json.dumps(out, indent=2), encoding="utf-8")
    df.to_csv(out_csv, index=False)

    print("\nBEST:")
    print(best)
    print(f"saved={out_json}")
    print(f"saved={out_csv}")


if __name__ == "__main__":
    main()

