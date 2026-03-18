from __future__ import annotations
import json
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

DAILY_PATH = Path(r"C:\Users\temen\5 joulu\TrailingStopLoss pivot peak trading 6 march 2026\pair_workspaces\eurjpy_2003_20260306_15m_v1\eurjpy_daily_ohlcv_ver3.csv")
M15_PATH = Path(r"C:\Users\temen\5 joulu\TrailingStopLoss pivot peak trading 6 march 2026\forex raw data\Dukascopy\weekly_monthly_analysis\2003_20260306_15m\eurjpy\derived_15m\eurjpy_15m_20030804_20260306.csv")
OUT_DIR = Path(__file__).parent

PIP = 0.01
COST_PIPS = 1.0
SL_ATR_MULT = 0.70
ATR_PERIOD = 14


def atr_wilder(h, l, c, period=14):
    n = len(h)
    tr = np.empty(n)
    tr[0] = h[0] - l[0]
    for i in range(1, n):
        tr[i] = max(h[i] - l[i], abs(h[i] - c[i - 1]), abs(l[i] - c[i - 1]))
    atr = np.full(n, np.nan)
    if n >= period:
        atr[period - 1] = np.mean(tr[:period])
        k = 1.0 / period
        for i in range(period, n):
            atr[i] = atr[i - 1] * (1 - k) + tr[i] * k
    return atr


def run_one(tp_mult: float, daily: pd.DataFrame, m15: pd.DataFrame) -> dict:
    h = daily["High"].values
    l = daily["Low"].values
    c = daily["Close"].values
    atr = atr_wilder(h, l, c, ATR_PERIOD)
    daily_loc = daily.copy()
    daily_loc[f"ATR{ATR_PERIOD}"] = atr
    daily_loc["YH"] = daily_loc["High"].shift(1)
    daily_loc["YL"] = daily_loc["Low"].shift(1)
    daily_loc["YC"] = daily_loc["Close"].shift(1)
    daily_loc["PP"] = (daily_loc["YH"] + daily_loc["YL"] + daily_loc["YC"]) / 3.0
    daily_loc["S1"] = 2 * daily_loc["PP"] - daily_loc["YH"]
    daily_loc["atr_prev"] = pd.Series(atr).shift(1)
    daily_loc["date_key"] = daily_loc["Date"].dt.date

    m15_loc = m15.copy()
    m15_loc["date_key"] = m15_loc["Datetime"].dt.date

    m15_highs = m15_loc["High"].values.astype(float)
    m15_lows = m15_loc["Low"].values.astype(float)
    m15_closes = m15_loc["Close"].values.astype(float)
    m15_dates = m15_loc["date_key"].values

    day_first_bar = {}
    day_last_bar = {}
    for i, dk in enumerate(m15_dates):
        if dk not in day_first_bar:
            day_first_bar[dk] = i
        day_last_bar[dk] = i

    ledger = []
    total_bars = len(m15_highs)

    for idx in range(1, len(daily_loc)):
        row = daily_loc.iloc[idx]
        dk = row["date_key"]
        pp = row["PP"]
        atr_val = row["atr_prev"]
        day_open = row["Open"]

        if not np.isfinite(pp) or not np.isfinite(atr_val) or not np.isfinite(day_open):
            continue
        # only when day opens under pivot
        if not (day_open < pp):
            continue
        if atr_val <= 0 or dk not in day_first_bar:
            continue

        sl_dist = SL_ATR_MULT * atr_val
        tp_dist = sl_dist * tp_mult

        i_start = day_first_bar[dk]
        i_end = day_last_bar[dk]

        entry_bar = None
        for j in range(i_start, i_end + 1):
            if m15_lows[j] <= pp:
                entry_bar = j
                break
        if entry_bar is None:
            continue

        entry_px = pp
        sl_px = entry_px + sl_dist
        tp_px = entry_px - tp_dist

        exit_px = None
        exit_reason = None
        exit_bar = None

        for j in range(entry_bar + 1, total_bars):
            bh = m15_highs[j]
            bl = m15_lows[j]
            hit_sl = bh >= sl_px
            hit_tp = bl <= tp_px
            if hit_sl and hit_tp:
                exit_bar = j
                exit_px = sl_px
                exit_reason = "BOTH_SL"
                break
            elif hit_sl:
                exit_bar = j
                exit_px = sl_px
                exit_reason = "SL"
                break
            elif hit_tp:
                exit_bar = j
                exit_px = tp_px
                exit_reason = "TP"
                break

        if exit_px is None:
            exit_bar = total_bars - 1
            exit_px = float(m15_closes[exit_bar])
            exit_reason = "DATASET_END"

        raw_pips = (entry_px - exit_px) / PIP
        net_pips = raw_pips - COST_PIPS
        ledger.append(net_pips)

    if not ledger:
        return {"tp_mult": tp_mult, "n_trades": 0}

    arr = np.array(ledger)
    total_pips = float(arr.sum())
    n_trades = int(len(arr))
    wins = int((arr > 0).sum())
    losses = int((arr <= 0).sum())
    win_rate = wins / n_trades

    first_date = daily_loc["date_key"].min()
    last_date = daily_loc["date_key"].max()
    cal_days = (pd.Timestamp(last_date) - pd.Timestamp(first_date)).days + 1
    cal_ppd = total_pips / cal_days

    return {
        "tp_mult": tp_mult,
        "n_trades": n_trades,
        "total_pips": round(total_pips, 2),
        "cal_ppd": round(cal_ppd, 4),
        "win_rate": round(win_rate, 4),
        "avg_pips_per_trade": round(total_pips / n_trades, 4),
    }


def run():
    daily = pd.read_csv(DAILY_PATH)
    daily["Date"] = pd.to_datetime(daily["Date"])
    for c in ["Open", "High", "Low", "Close"]:
        daily[c] = pd.to_numeric(daily[c], errors="coerce")
    daily = daily.dropna().sort_values("Date").reset_index(drop=True)

    m15 = pd.read_csv(M15_PATH)
    m15["Datetime"] = pd.to_datetime(m15["Datetime"])
    for c in ["Open", "High", "Low", "Close"]:
        m15[c] = pd.to_numeric(m15[c], errors="coerce")
    m15 = m15.dropna().sort_values("Datetime").reset_index(drop=True)

    tps: List[float] = list(np.round(np.arange(1.20, 1.5001, 0.02), 2))
    rows = []
    for tp in tps:
        out = run_one(tp, daily, m15)
        print(f"tp={tp:.2f} -> trades={out.get('n_trades',0)}  cal_ppd={out.get('cal_ppd','NA')}")
        rows.append(out)

    res_df = pd.DataFrame(rows)
    out_csv = OUT_DIR / "sell_pp_s1_sl70_tp_sweep_atr14_open_underpp_results.csv"
    res_df.to_csv(out_csv, index=False)

    best = res_df.sort_values("cal_ppd", ascending=False).iloc[0].to_dict()
    print("\nBest TP multiplier (by cal_ppd):")
    print(best)
    summary_path = OUT_DIR / "sell_pp_s1_sl70_tp_sweep_atr14_open_underpp_summary.json"
    Path(summary_path).write_text(json.dumps({"best": best, "results": rows}, indent=2))
    print(f"\nResults written: {out_csv} , {summary_path}")


if __name__ == "__main__":
    run()

