from __future__ import annotations
import json
from pathlib import Path
from statistics import mean, median, pstdev

import numpy as np
import pandas as pd

DAILY_PATH = Path(r"C:\Users\temen\5 joulu\TrailingStopLoss pivot peak trading 6 march 2026\pair_workspaces\eurjpy_2003_20260306_15m_v1\eurjpy_daily_ohlcv_ver3.csv")
M15_PATH = Path(r"C:\Users\temen\5 joulu\TrailingStopLoss pivot peak trading 6 march 2026\forex raw data\Dukascopy\weekly_monthly_analysis\2003_20260306_15m\eurjpy\derived_15m\eurjpy_15m_20030804_20260306.csv")
OUT_DIR = Path(__file__).parent

PIP = 0.01
COST_PIPS = 1.0
SL_ATR_MULT = 0.70
TP_MULT = 1.3
ATR_PERIOD = 14
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


def adx_wilder(h: np.ndarray, l: np.ndarray, c: np.ndarray, period=14):
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
    dx = 100.0 * (np.abs(plus_di - minus_di) / (plus_di + minus_di))
    adx = np.full(n, np.nan)
    if n >= period:
        adx[period - 1] = np.nanmean(dx[:period])
        k = 1.0 / period
        for i in range(period, n):
            adx[i] = adx[i - 1] * (1 - k) + dx[i] * k

    return adx, plus_di, minus_di


def build_masks(daily: pd.DataFrame):
    donch_high = daily["High"].shift(1).rolling(TURTLE_N).max()
    donch_low = daily["Low"].shift(1).rolling(TURTLE_N).min()
    donch_mid = (donch_high + donch_low) / 2.0
    daily["donch_mid"] = donch_mid
    daily["donch_mid_prev"] = donch_mid.shift(1)
    daily["+DI_prev"] = daily["+DI_prev"]
    daily["-DI_prev"] = daily["-DI_prev"]
    mask_both_up = set(daily.loc[(daily["donch_mid"] > daily["donch_mid_prev"]) & (daily["+DI_prev"] > daily["-DI_prev"]), "date_key"].dropna())
    mask_both_down = set(daily.loc[(daily["donch_mid"] < daily["donch_mid_prev"]) & (daily["-DI_prev"] > daily["+DI_prev"]), "date_key"].dropna())
    mask_disagree1 = set(daily.loc[(daily["donch_mid"] > daily["donch_mid_prev"]) & (daily["-DI_prev"] > daily["+DI_prev"]), "date_key"].dropna())
    mask_disagree2 = set(daily.loc[(daily["donch_mid"] < daily["donch_mid_prev"]) & (daily["+DI_prev"] > daily["-DI_prev"]), "date_key"].dropna())
    return {"both_up": mask_both_up, "both_down": mask_both_down, "turtle_up_di_down": mask_disagree1, "turtle_down_di_up": mask_disagree2}


def simulate_mask(mask_name: str, mask_dates, daily: pd.DataFrame, m15: pd.DataFrame):
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
    for idx in range(1, len(daily)):
        row = daily.iloc[idx]
        dk = row["date_key"]
        if dk not in mask_dates:
            continue
        pp = row["PP"]
        atr_val = row["ATR_prev"]
        day_open = row["Open"]
        if not np.isfinite(pp) or not np.isfinite(atr_val) or not np.isfinite(day_open):
            continue
        if not (day_open < pp):
            continue
        if atr_val <= 0 or dk not in day_first_bar:
            continue
        sl_dist = SL_ATR_MULT * atr_val
        tp_dist = sl_dist * TP_MULT
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
        exit_bar = None
        for j in range(entry_bar + 1, total_bars):
            bh = m15_highs[j]
            bl = m15_lows[j]
            hit_sl = bh >= sl_px
            hit_tp = bl <= tp_px
            if hit_sl and hit_tp:
                exit_bar = j
                exit_px = sl_px
                break
            elif hit_sl:
                exit_bar = j
                exit_px = sl_px
                break
            elif hit_tp:
                exit_bar = j
                exit_px = tp_px
                break
        if exit_px is None:
            exit_bar = total_bars - 1
            exit_px = float(m15_closes[exit_bar])
        raw_pips = (entry_px - exit_px) / PIP
        net_pips = raw_pips - COST_PIPS
        ledger.append({"date": str(dk), "net_pips": net_pips})
    df = pd.DataFrame(ledger)
    ledger_path = OUT_DIR / f"ledger_{mask_name}.csv"
    df.to_csv(ledger_path, index=False)
    if df.empty:
        return {"mask": mask_name, "n_trades": 0, "ledger": str(ledger_path)}
    total_pips = float(df["net_pips"].sum())
    n_trades = len(df)
    avg = float(df["net_pips"].mean())
    med = float(df["net_pips"].median())
    std = float(df["net_pips"].std(ddof=0))
    # pips per active trading day
    per_day = df.groupby("date")["net_pips"].sum()
    active_days = len(per_day)
    pips_per_active_day = float(per_day.sum() / max(active_days, 1))
    # calendar ppd
    first = pd.to_datetime(df["date"].min())
    last = pd.to_datetime(df["date"].max())
    cal_days = (last - first).days + 1
    cal_ppd = total_pips / cal_days
    return {
        "mask": mask_name,
        "n_trades": n_trades,
        "total_pips": round(total_pips, 2),
        "avg_pips_trade": round(avg, 4),
        "median_pips_trade": round(med, 4),
        "std_pips_trade": round(std, 4),
        "active_days": active_days,
        "pips_per_active_day": round(pips_per_active_day, 4),
        "cal_days": cal_days,
        "cal_ppd": round(cal_ppd, 4),
        "ledger": str(ledger_path),
    }


def run():
    daily = pd.read_csv(DAILY_PATH)
    daily["Date"] = pd.to_datetime(daily["Date"])
    for c in ["Open", "High", "Low", "Close"]:
        daily[c] = pd.to_numeric(daily[c], errors="coerce")
    daily = daily.dropna().sort_values("Date").reset_index(drop=True)
    h = daily["High"].values
    l = daily["Low"].values
    c = daily["Close"].values
    atr = atr_wilder(h, l, c, ATR_PERIOD)
    adx, plus_di, minus_di = adx_wilder(h, l, c, ATR_PERIOD)
    daily["ATR_prev"] = pd.Series(atr).shift(1)
    daily["+DI_prev"] = pd.Series(plus_di).shift(1)
    daily["-DI_prev"] = pd.Series(minus_di).shift(1)
    daily["YH"] = daily["High"].shift(1)
    daily["YL"] = daily["Low"].shift(1)
    daily["YC"] = daily["Close"].shift(1)
    daily["PP"] = (daily["YH"] + daily["YL"] + daily["YC"]) / 3.0
    daily["S1"] = 2 * daily["PP"] - daily["YH"]
    daily["date_key"] = daily["Date"].dt.date
    masks = build_masks(daily)
    m15 = pd.read_csv(M15_PATH)
    m15["Datetime"] = pd.to_datetime(m15["Datetime"])
    m15 = m15.dropna().sort_values("Datetime").reset_index(drop=True)
    results = {}
    for name, mask in masks.items():
        out = simulate_mask(name, mask, daily, m15)
        print(name, out)
        results[name] = out
    out_path = OUT_DIR / "sell_pp_s1_turtle_adx_disagree_stats_summary.json"
    out_path.write_text(json.dumps(results, indent=2))
    print("Written:", out_path)


if __name__ == "__main__":
    run()

