from __future__ import annotations
import json
from pathlib import Path

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
ADX_PERIOD = 14
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


def simulate_days(mask_name: str, mask_dates, daily, m15):
    # mask_dates: set of date objects to trade
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
        ledger.append({"date": str(dk), "net_pips": net_pips})

    df = pd.DataFrame(ledger)
    if df.empty:
        return {"mask": mask_name, "n_trades": 0}
    total_pips = float(df["net_pips"].sum())
    n_trades = len(df)
    wins = int((df["net_pips"] > 0).sum())
    losses = n_trades - wins
    win_rate = wins / n_trades
    first_date = df["date"].min()
    last_date = df["date"].max()
    cal_days = (pd.Timestamp(last_date) - pd.Timestamp(first_date)).days + 1
    cal_ppd = total_pips / cal_days
    return {
        "mask": mask_name,
        "n_trades": n_trades,
        "total_pips": round(total_pips, 2),
        "cal_ppd": round(cal_ppd, 4),
        "wins": wins,
        "losses": losses,
        "win_rate": round(win_rate, 4),
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
    adx, plus_di, minus_di = adx_wilder(h, l, c, ADX_PERIOD)
    daily["ATR_prev"] = pd.Series(atr).shift(1)
    daily["+DI_prev"] = pd.Series(plus_di).shift(1)
    daily["-DI_prev"] = pd.Series(minus_di).shift(1)
    donch_high = daily["High"].shift(1).rolling(TURTLE_N).max()
    donch_low = daily["Low"].shift(1).rolling(TURTLE_N).min()
    donch_mid = (donch_high + donch_low) / 2.0
    daily["donch_mid"] = donch_mid
    daily["donch_mid_prev"] = donch_mid.shift(1)
    daily["YH"] = daily["High"].shift(1)
    daily["YL"] = daily["Low"].shift(1)
    daily["YC"] = daily["Close"].shift(1)
    daily["PP"] = (daily["YH"] + daily["YL"] + daily["YC"]) / 3.0
    daily["S1"] = 2 * daily["PP"] - daily["YH"]
    daily["date_key"] = daily["Date"].dt.date

    # masks
    mask_both_up = set(daily.loc[(daily["donch_mid"] > daily["donch_mid_prev"]) & (daily["+DI_prev"] > daily["-DI_prev"]), "date_key"].dropna())
    mask_both_down = set(daily.loc[(daily["donch_mid"] < daily["donch_mid_prev"]) & (daily["-DI_prev"] > daily["+DI_prev"]), "date_key"].dropna())
    mask_disagree1 = set(daily.loc[(daily["donch_mid"] > daily["donch_mid_prev"]) & (daily["-DI_prev"] > daily["+DI_prev"]), "date_key"].dropna())  # Turtle up, DI down
    mask_disagree2 = set(daily.loc[(daily["donch_mid"] < daily["donch_mid_prev"]) & (daily["+DI_prev"] > daily["-DI_prev"]), "date_key"].dropna())  # Turtle down, DI up

    m15 = pd.read_csv(M15_PATH)
    m15["Datetime"] = pd.to_datetime(m15["Datetime"])
    m15 = m15.dropna().sort_values("Datetime").reset_index(drop=True)

    out_bu = simulate_days("both_up", mask_both_up, daily, m15)
    out_bd = simulate_days("both_down", mask_both_down, daily, m15)
    out_d1 = simulate_days("turtle_up_di_down", mask_disagree1, daily, m15)
    out_d2 = simulate_days("turtle_down_di_up", mask_disagree2, daily, m15)

    results = {"both_up": out_bu, "both_down": out_bd, "turtle_up_di_down": out_d1, "turtle_down_di_up": out_d2}
    out_path = OUT_DIR / "sell_pp_s1_turtle_adx_disagree_summary.json"
    out_path.write_text(json.dumps(results, indent=2))
    print("Results written:", out_path)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    run()

