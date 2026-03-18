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

    # Donchian based turtle channel using previous-day windows (no lookahead)
    donch_high = daily["High"].shift(1).rolling(TURTLE_N).max()
    donch_low = daily["Low"].shift(1).rolling(TURTLE_N).min()
    donch_mid = (donch_high + donch_low) / 2.0
    donch_mid_prev = donch_mid.shift(1)

    daily["ATR14_prev"] = pd.Series(atr).shift(1)
    daily["ADX"] = adx
    daily["+DI_prev"] = pd.Series(plus_di).shift(1)
    daily["-DI_prev"] = pd.Series(minus_di).shift(1)
    daily["donch_mid"] = donch_mid
    daily["donch_mid_prev"] = donch_mid_prev
    daily["YH"] = daily["High"].shift(1)
    daily["YL"] = daily["Low"].shift(1)
    daily["YC"] = daily["Close"].shift(1)
    daily["PP"] = (daily["YH"] + daily["YL"] + daily["YC"]) / 3.0
    daily["S1"] = 2 * daily["PP"] - daily["YH"]
    daily["date_key"] = daily["Date"].dt.date

    m15 = pd.read_csv(M15_PATH)
    m15["Datetime"] = pd.to_datetime(m15["Datetime"])
    for c in ["Open", "High", "Low", "Close"]:
        m15[c] = pd.to_numeric(m15[c], errors="coerce")
    m15 = m15.dropna().sort_values("Datetime").reset_index(drop=True)
    m15["date_key"] = m15["Datetime"].dt.date

    m15_highs = m15["High"].values.astype(float)
    m15_lows = m15["Low"].values.astype(float)
    m15_closes = m15["Close"].values.astype(float)
    m15_dates = m15["date_key"].values

    day_first_bar = {}
    day_last_bar = {}
    for i, dk in enumerate(m15_dates):
        if dk not in day_first_bar:
            day_first_bar[dk] = i
        day_last_bar[dk] = i

    def run_filter(trend_direction: str):
        # trend_direction: "both_up" => donch_mid rising & +DI > -DI
        #                  "both_down" => donch_mid falling & -DI > +DI
        ledger = []
        total_bars = len(m15_highs)
        for idx in range(1, len(daily)):
            row = daily.iloc[idx]
            dk = row["date_key"]
            pp = row["PP"]
            atr_val = row["ATR14_prev"]
            day_open = row["Open"]
            donch_m = row["donch_mid"]
            donch_m_prev = row["donch_mid_prev"]
            di_plus = row["+DI_prev"]
            di_minus = row["-DI_prev"]

            if not np.isfinite(pp) or not np.isfinite(atr_val):
                continue
            if atr_val <= 0 or dk not in day_first_bar:
                continue
            if not np.isfinite(donch_m) or not np.isfinite(donch_m_prev):
                continue
            if not np.isfinite(di_plus) or not np.isfinite(di_minus):
                continue
            # only when day opens under pivot (sell setups)
            if not (day_open < pp):
                continue

            if trend_direction == "both_up":
                if not (donch_m > donch_m_prev and di_plus > di_minus):
                    continue
            else:  # both_down
                if not (donch_m < donch_m_prev and di_minus > di_plus):
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
            ledger.append({
                "date": str(dk),
                "entry_px": round(entry_px, 5),
                "exit_px": round(exit_px, 5),
                "net_pips": round(net_pips, 4),
                "exit_reason": exit_reason,
                "donch_mid": round(float(donch_m), 5),
                "+DI_prev": round(float(di_plus), 4),
                "-DI_prev": round(float(di_minus), 4),
            })

        df = pd.DataFrame(ledger)
        if df.empty:
            return {"mode": trend_direction, "n_trades": 0}
        total_pips = float(df["net_pips"].sum())
        n_trades = len(df)
        wins = int((df["net_pips"] > 0).sum())
        losses = int((df["net_pips"] <= 0).sum())
        win_rate = wins / n_trades
        first_date = df["date"].min()
        last_date = df["date"].max()
        cal_days = (pd.Timestamp(last_date) - pd.Timestamp(first_date)).days + 1
        cal_ppd = total_pips / cal_days
        ledger_path = OUT_DIR / f"sell_pp_s1_atr14_turtle{TURTLE_N}_{trend_direction}_ledger.csv"
        df.to_csv(ledger_path, index=False)
        return {
            "mode": trend_direction,
            "n_trades": n_trades,
            "total_pips": round(total_pips, 2),
            "cal_ppd": round(cal_ppd, 4),
            "wins": wins,
            "losses": losses,
            "win_rate": round(win_rate, 4),
            "ledger": str(ledger_path),
        }

    out_up = run_filter("both_up")
    print("both_up ->", out_up)
    out_down = run_filter("both_down")
    print("both_down ->", out_down)
    summary_path = OUT_DIR / "sell_pp_s1_atr14_turtle55_adx_summary.json"
    summary_path.write_text(json.dumps({"both_up": out_up, "both_down": out_down}, indent=2))
    print(f"\nSummary written: {summary_path}")


if __name__ == "__main__":
    run()

