from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

ROOT = Path(r"C:\Users\temen\5 joulu\TrailingStopLoss pivot peak trading 6 march 2026")
M15_PATH = ROOT / "forex raw data" / "Dukascopy" / "weekly_monthly_analysis" / "2003_20260306_15m" / "eurjpy" / "derived_15m" / "eurjpy_15m_20030804_20260306.csv"
DAILY_PATH = ROOT / "pair_workspaces" / "eurjpy_2003_20260306_15m_v1" / "eurjpy_daily_ohlcv_ver3.csv"
OUT_DIR = Path(__file__).parent

PIP = 0.01
SPREAD_PIPS = 1.0
TP_PIPS = 50
SL_PIPS = 300
MAX_HOLD_HOURS = 432


def load_data():
    d = pd.read_csv(DAILY_PATH, usecols=["Date", "High", "Low"], low_memory=False)
    d["Date"] = pd.to_datetime(d["Date"], errors="coerce")
    d["High"] = pd.to_numeric(d["High"], errors="coerce")
    d["Low"] = pd.to_numeric(d["Low"], errors="coerce")
    d = d.dropna().sort_values("Date").reset_index(drop=True)
    d["yh"] = d["High"].shift(1)
    d["yl"] = d["Low"].shift(1)
    d = d.dropna(subset=["yh", "yl"]).copy()
    d["dkey"] = d["Date"].dt.normalize()

    m = pd.read_csv(M15_PATH, usecols=["Datetime", "High", "Low", "Close"], low_memory=False)
    m["Datetime"] = pd.to_datetime(m["Datetime"], errors="coerce")
    for c in ["High", "Low", "Close"]:
        m[c] = pd.to_numeric(m[c], errors="coerce")
    m = m.dropna().sort_values("Datetime").reset_index(drop=True)
    m["dkey"] = m["Datetime"].dt.normalize()

    # Full days only
    full_days = m.groupby("dkey").size()
    full_days = set(full_days[full_days == 96].index)
    m = m[m["dkey"].isin(full_days)].copy()

    bars = {}
    first_idx = {}
    for dkey, g in m.groupby("dkey", sort=True):
        g = g.reset_index()
        first_idx[dkey] = int(g.loc[0, "index"])
        bars[dkey] = {
            "h": g["High"].to_list(),
            "l": g["Low"].to_list(),
            "last_c": float(g["Close"].iloc[-1]),
        }

    info = {}
    for _, r in d.iterrows():
        k = r["dkey"]
        if k in bars:
            info[k] = {"yh": float(r["yh"]), "yl": float(r["yl"]), "i0": first_idx[k]}
    return m, info, bars


def close_pips(side: str, entry: float, exit_px: float) -> float:
    if side == "BUY":
        return ((exit_px - entry) / PIP) - SPREAD_PIPS
    return ((entry - exit_px) / PIP) - SPREAD_PIPS


def main():
    _m, info, bars = load_data()
    days = sorted(info.keys())
    max_bars = MAX_HOLD_HOURS * 4

    total = 0.0
    buy_total = 0.0
    sell_total = 0.0
    buy_wins = buy_losses = 0
    sell_wins = sell_losses = 0
    buy_trades = sell_trades = 0
    buy_time = sell_time = 0

    for d in days:
        lv = info[d]
        yh = lv["yh"]
        yl = lv["yl"]
        h = bars[d]["h"]
        l = bars[d]["l"]
        last_c = bars[d]["last_c"]

        # First touches during the day.
        touch_yh = None
        touch_yl = None
        for i, (hi, lo) in enumerate(zip(h, l)):
            if touch_yh is None and (lo <= yh <= hi):
                touch_yh = i
            if touch_yl is None and (lo <= yl <= hi):
                touch_yl = i
            if (touch_yh is not None) and (touch_yl is not None):
                break

        # SELL on YH touch
        if touch_yh is not None and (touch_yh + 1) < len(h):
            entry = yh
            tp = entry - TP_PIPS * PIP
            sl = entry + SL_PIPS * PIP
            sell_trades += 1
            end_i = min(touch_yh + 1 + max_bars, len(h) - 1)
            done = False
            for j in range(touch_yh + 1, end_i + 1):
                hi, lo = h[j], l[j]
                hit_sl = hi >= sl
                hit_tp = lo <= tp
                if hit_sl and hit_tp:
                    px = sl
                    p = close_pips("SELL", entry, px)
                    sell_total += p
                    total += p
                    sell_losses += 1
                    done = True
                    break
                if hit_sl:
                    px = sl
                    p = close_pips("SELL", entry, px)
                    sell_total += p
                    total += p
                    sell_losses += 1
                    done = True
                    break
                if hit_tp:
                    px = tp
                    p = close_pips("SELL", entry, px)
                    sell_total += p
                    total += p
                    sell_wins += 1
                    done = True
                    break
            if not done:
                p = close_pips("SELL", entry, last_c)
                sell_total += p
                total += p
                sell_time += 1
                if p >= 0:
                    sell_wins += 1
                else:
                    sell_losses += 1

        # BUY on YL touch
        if touch_yl is not None and (touch_yl + 1) < len(h):
            entry = yl
            tp = entry + TP_PIPS * PIP
            sl = entry - SL_PIPS * PIP
            buy_trades += 1
            end_i = min(touch_yl + 1 + max_bars, len(h) - 1)
            done = False
            for j in range(touch_yl + 1, end_i + 1):
                hi, lo = h[j], l[j]
                hit_sl = lo <= sl
                hit_tp = hi >= tp
                if hit_sl and hit_tp:
                    px = sl
                    p = close_pips("BUY", entry, px)
                    buy_total += p
                    total += p
                    buy_losses += 1
                    done = True
                    break
                if hit_sl:
                    px = sl
                    p = close_pips("BUY", entry, px)
                    buy_total += p
                    total += p
                    buy_losses += 1
                    done = True
                    break
                if hit_tp:
                    px = tp
                    p = close_pips("BUY", entry, px)
                    buy_total += p
                    total += p
                    buy_wins += 1
                    done = True
                    break
            if not done:
                p = close_pips("BUY", entry, last_c)
                buy_total += p
                total += p
                buy_time += 1
                if p >= 0:
                    buy_wins += 1
                else:
                    buy_losses += 1

    cal_days = len(days)
    out = {
        "pair": "EURJPY",
        "period": [str(days[0].date()), str(days[-1].date())],
        "rule": "SELL on YH touch, BUY on YL touch",
        "tp_pips": TP_PIPS,
        "sl_pips": SL_PIPS,
        "max_hold_hours": MAX_HOLD_HOURS,
        "spread_pips": SPREAD_PIPS,
        "days": cal_days,
        "buy_trades": buy_trades,
        "sell_trades": sell_trades,
        "buy_wins": buy_wins,
        "buy_losses": buy_losses,
        "sell_wins": sell_wins,
        "sell_losses": sell_losses,
        "buy_time_exits": buy_time,
        "sell_time_exits": sell_time,
        "buy_cal_ppd": round(buy_total / max(cal_days, 1), 4),
        "sell_cal_ppd": round(sell_total / max(cal_days, 1), 4),
        "cal_ppd": round(total / max(cal_days, 1), 4),
        "total_pips": round(total, 4),
    }
    out_json = OUT_DIR / "yh_yl_touch_reversal_fixed_50tp_300sl_eurjpy_full_days_432h.json"
    out_json.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=2))
    print(f"saved={out_json}")


if __name__ == "__main__":
    main()

