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
ATR_PERIOD = 14
SL_ATR_MULT = 0.70
TP_MULT_FROM_SL = 1.4  # TP = SL * 1.4
TURTLE_N = 55


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


def adx_plus_minus(h: np.ndarray, l: np.ndarray, c: np.ndarray, period=14):
    n = len(h)
    tr = np.empty(n)
    plus_dm = np.zeros(n)
    minus_dm = np.zeros(n)
    tr[0] = h[0] - l[0]
    for i in range(1, n):
        tr[i] = max(h[i] - l[i], abs(h[i] - c[i - 1]), abs(l[i] - c[i - 1]))
        up = h[i] - h[i - 1]
        down = l[i - 1] - l[i]
        if up > down and up > 0:
            plus_dm[i] = up
        if down > up and down > 0:
            minus_dm[i] = down
    atr = np.full(n, np.nan)
    plus_dm_sm = np.full(n, np.nan)
    minus_dm_sm = np.full(n, np.nan)
    if n >= period:
        atr[period - 1] = np.mean(tr[:period])
        plus_dm_sm[period - 1] = np.mean(plus_dm[:period])
        minus_dm_sm[period - 1] = np.mean(minus_dm[:period])
        k = 1.0 / period
        for i in range(period, n):
            atr[i] = atr[i - 1] * (1 - k) + tr[i] * k
            plus_dm_sm[i] = plus_dm_sm[i - 1] * (1 - k) + plus_dm[i] * k
            minus_dm_sm[i] = minus_dm_sm[i - 1] * (1 - k) + minus_dm[i] * k
    plus_di = 100.0 * (plus_dm_sm / atr)
    minus_di = 100.0 * (minus_dm_sm / atr)
    return plus_di, minus_di


def compute_turtle_state(donch_mid: pd.Series) -> pd.Series:
    # state_up True if the last non-zero change prior to day was positive and hasn't turned down.
    diff = donch_mid.diff()
    # get last non-zero sign up to each day
    signs = diff.replace(0.0, np.nan).dropna().apply(np.sign)
    # create array of state by forward-filling last non-zero sign and checking >0
    last_sign = diff.copy().replace(0.0, np.nan).ffill()
    state_up = last_sign > 0
    # fill NaN with False
    return state_up.fillna(False)


def run_period(start_date: str, end_date: str):
    daily = pd.read_csv(DAILY_PATH)
    daily["Date"] = pd.to_datetime(daily["Date"])
    for c in ["Open", "High", "Low", "Close"]:
        daily[c] = pd.to_numeric(daily[c], errors="coerce")
    daily = daily.dropna().sort_values("Date").reset_index(drop=True)
    # will filter requested range after we enrich daily with indicators

    h = daily["High"].values
    l = daily["Low"].values
    c = daily["Close"].values
    plus_di, minus_di = adx_plus_minus(h, l, c, ATR_PERIOD)
    atr = atr_wilder(h, l, c, ATR_PERIOD)

    # compute turtle donchian mid (based on previous days)
    donch_high = daily["High"].shift(1).rolling(TURTLE_N).max()
    donch_low = daily["Low"].shift(1).rolling(TURTLE_N).min()
    donch_mid = (donch_high + donch_low) / 2.0
    turtle_state = compute_turtle_state(donch_mid)

    daily["PP"] = ((daily["High"].shift(1) + daily["Low"].shift(1) + daily["Close"].shift(1)) / 3.0)
    daily["atr_prev"] = pd.Series(atr).shift(1)
    daily["+DI_prev"] = pd.Series(plus_di).shift(1)
    daily["-DI_prev"] = pd.Series(minus_di).shift(1)
    daily["donch_mid"] = donch_mid
    daily["turtle_state_up"] = turtle_state
    daily["date_key"] = daily["Date"].dt.date

    # filter requested range (now that daily has indicator columns)
    mask_range = (daily["Date"] >= pd.to_datetime(start_date)) & (daily["Date"] <= pd.to_datetime(end_date))
    if not mask_range.any():
        print("No daily rows in range")
        return
    daily_range = daily.loc[mask_range].copy().reset_index(drop=True)

    # load m15
    m15 = pd.read_csv(M15_PATH)
    m15["Datetime"] = pd.to_datetime(m15["Datetime"])
    for ccol in ["Open", "High", "Low", "Close"]:
        m15[ccol] = pd.to_numeric(m15[ccol], errors="coerce")
    m15 = m15.dropna().sort_values("Datetime").reset_index(drop=True)
    m15["date_key"] = m15["Datetime"].dt.date
    # map date -> first/last bar
    day_first = {}
    day_last = {}
    for i, dk in enumerate(m15["date_key"].values):
        if dk not in day_first:
            day_first[dk] = i
        day_last[dk] = i

    # build touched set for quick lookup: days where intraday touched PP
    touched = set()
    # iterate only dates in period
    for _, row in daily_range.iterrows():
        dk = row["date_key"]
        if dk not in day_first:
            continue
        i0 = day_first[dk]
        i1 = day_last[dk]
        lows = m15["Low"].values[i0 : i1 + 1]
        highs = m15["High"].values[i0 : i1 + 1]
        pp = row["PP"]
        if np.isfinite(pp) and (lows.min() <= pp <= highs.max()):
            touched.add(dk)

    def simulate(mask_name: str, condition_mask: pd.Series):
        ledger = []
        for _, row in daily_range.iterrows():
            dk = row["date_key"]
            if not condition_mask.loc[row.name]:
                continue
            if dk not in day_first:
                continue
            if dk not in touched:
                continue
            if not (row["Open"] < row["PP"]):
                continue
            atr_prev = row["atr_prev"]
            if not np.isfinite(atr_prev) or atr_prev <= 0:
                continue
            sl_dist = SL_ATR_MULT * atr_prev
            tp_dist = sl_dist * TP_MULT_FROM_SL
            i_start = day_first[dk]
            i_end = day_last[dk]
            entry_bar = None
            lows = m15["Low"].values
            highs = m15["High"].values
            closes = m15["Close"].values
            for j in range(i_start, i_end + 1):
                if lows[j] <= row["PP"]:
                    entry_bar = j
                    break
            if entry_bar is None:
                continue
            entry_px = row["PP"]
            sl_px = entry_px + sl_dist
            tp_px = entry_px - tp_dist
            exit_px = None
            exit_reason = None
            exit_bar = None
            for j in range(entry_bar + 1, len(lows)):
                bh = highs[j]
                bl = lows[j]
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
                exit_bar = day_last[dk]
                exit_px = float(closes[exit_bar])
                exit_reason = "DATASET_END"
            raw_pips = (entry_px - exit_px) / PIP
            net_pips = raw_pips - COST_PIPS
            ledger.append({
                "date": str(dk),
                "entry_px": round(entry_px, 5),
                "exit_px": round(exit_px, 5),
                "net_pips": round(net_pips, 4),
                "exit_reason": exit_reason,
            })
        df = pd.DataFrame(ledger)
        if df.empty:
            return {"mask": mask_name, "n_trades": 0}
        total = float(df["net_pips"].sum())
        n = len(df)
        avg = float(df["net_pips"].mean())
        med = float(df["net_pips"].median())
        std = float(df["net_pips"].std(ddof=0))
        top10 = df.sort_values("net_pips", ascending=False).head(10).to_dict(orient="records")
        out_path = OUT_DIR / f"ledger_2y_{mask_name}.csv"
        df.to_csv(out_path, index=False)
        return {
            "mask": mask_name,
            "n_trades": n,
            "total_pips": round(total, 2),
            "avg_pips": round(avg, 4),
            "median_pips": round(med, 4),
            "std_pips": round(std, 4),
            "top10": top10,
            "ledger": str(out_path),
        }

    # build condition masks for daily_range (indexed from 0..)
    idx_map = {d: i for i, d in enumerate(daily_range["date_key"])}
    cond_plus_gt_minus = []
    cond_turtle_up = []
    for i, row in daily_range.iterrows():
        plus = daily.loc[daily["date_key"] == row["date_key"], "+DI_prev"]
        minus = daily.loc[daily["date_key"] == row["date_key"], "-DI_prev"]
        # get scalar if exists
        p = float(plus.iloc[0]) if not plus.empty and np.isfinite(plus.iloc[0]) else np.nan
        m = float(minus.iloc[0]) if not minus.empty and np.isfinite(minus.iloc[0]) else np.nan
        cond_plus_gt_minus.append(False if np.isnan(p) or np.isnan(m) else (p > m))
        # turtle state for that date (use daily['turtle_state_up'])
        ts = daily.loc[daily["date_key"] == row["date_key"], "turtle_state_up"]
        cond_turtle_up.append(bool(ts.iloc[0]) if not ts.empty else False)

    cond_plus_gt_minus = pd.Series(cond_plus_gt_minus, index=daily_range.index)
    cond_turtle_up = pd.Series(cond_turtle_up, index=daily_range.index)

    out_up = simulate("both_up", cond_plus_gt_minus & cond_turtle_up)
    out_down = simulate("both_down", (~cond_plus_gt_minus) & (~cond_turtle_up))

    summary = {"period": [start_date, end_date], "both_up": out_up, "both_down": out_down}
    summary_path = OUT_DIR / "sell_pp_s1_2y_turtle_di_states_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    # last two calendar years 2024-01-01 .. 2025-12-31
    run_period("2024-01-01", "2025-12-31")

