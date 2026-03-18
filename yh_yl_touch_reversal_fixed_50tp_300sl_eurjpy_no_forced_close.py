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
    day_levels = {r["dkey"]: {"yh": float(r["yh"]), "yl": float(r["yl"])} for _, r in d.iterrows()}

    m = pd.read_csv(M15_PATH, usecols=["Datetime", "High", "Low", "Close"], low_memory=False)
    m["Datetime"] = pd.to_datetime(m["Datetime"], errors="coerce")
    for c in ["High", "Low", "Close"]:
        m[c] = pd.to_numeric(m[c], errors="coerce")
    m = m.dropna().sort_values("Datetime").reset_index(drop=True)
    m["dkey"] = m["Datetime"].dt.normalize()
    # Use full 96-bar days only to keep consistency.
    full_days = m.groupby("dkey").size()
    full_days = set(full_days[full_days == 96].index)
    m = m[m["dkey"].isin(full_days)].copy().reset_index(drop=True)
    return m, day_levels


def close_pips(side: str, entry: float, exit_px: float) -> float:
    if side == "BUY":
        return ((exit_px - entry) / PIP) - SPREAD_PIPS
    return ((entry - exit_px) / PIP) - SPREAD_PIPS


def main():
    m, day_levels = load_data()
    highs = m["High"].to_numpy(float)
    lows = m["Low"].to_numpy(float)
    closes = m["Close"].to_numpy(float)
    dkeys = m["dkey"].to_numpy()

    triggered_buy = set()
    triggered_sell = set()
    open_positions = []

    total = 0.0
    buy_total = 0.0
    sell_total = 0.0
    buy_wins = buy_losses = 0
    sell_wins = sell_losses = 0
    buy_trades = sell_trades = 0

    for i in range(len(m)):
        dkey = dkeys[i]
        lv = day_levels.get(dkey)
        if lv is None:
            continue
        hi = highs[i]
        lo = lows[i]

        # Open SELL on first YH touch of the day.
        yh = lv["yh"]
        if dkey not in triggered_sell and (lo <= yh <= hi):
            triggered_sell.add(dkey)
            entry = yh
            open_positions.append(
                {"side": "SELL", "entry": entry, "tp": entry - TP_PIPS * PIP, "sl": entry + SL_PIPS * PIP}
            )
            sell_trades += 1

        # Open BUY on first YL touch of the day.
        yl = lv["yl"]
        if dkey not in triggered_buy and (lo <= yl <= hi):
            triggered_buy.add(dkey)
            entry = yl
            open_positions.append(
                {"side": "BUY", "entry": entry, "tp": entry + TP_PIPS * PIP, "sl": entry - SL_PIPS * PIP}
            )
            buy_trades += 1

        # Evaluate exits; no time/day forced close.
        still_open = []
        for p in open_positions:
            if p["side"] == "BUY":
                hit_sl = lo <= p["sl"]
                hit_tp = hi >= p["tp"]
                if hit_sl and hit_tp:
                    px = p["sl"]
                    pips = close_pips("BUY", p["entry"], px)
                    total += pips
                    buy_total += pips
                    buy_losses += 1
                elif hit_sl:
                    px = p["sl"]
                    pips = close_pips("BUY", p["entry"], px)
                    total += pips
                    buy_total += pips
                    buy_losses += 1
                elif hit_tp:
                    px = p["tp"]
                    pips = close_pips("BUY", p["entry"], px)
                    total += pips
                    buy_total += pips
                    buy_wins += 1
                else:
                    still_open.append(p)
            else:
                hit_sl = hi >= p["sl"]
                hit_tp = lo <= p["tp"]
                if hit_sl and hit_tp:
                    px = p["sl"]
                    pips = close_pips("SELL", p["entry"], px)
                    total += pips
                    sell_total += pips
                    sell_losses += 1
                elif hit_sl:
                    px = p["sl"]
                    pips = close_pips("SELL", p["entry"], px)
                    total += pips
                    sell_total += pips
                    sell_losses += 1
                elif hit_tp:
                    px = p["tp"]
                    pips = close_pips("SELL", p["entry"], px)
                    total += pips
                    sell_total += pips
                    sell_wins += 1
                else:
                    still_open.append(p)
        open_positions = still_open

    # Do NOT close remaining positions. Report unrealized separately.
    last_close = float(closes[-1])
    unrealized = 0.0
    for p in open_positions:
        unrealized += close_pips(p["side"], p["entry"], last_close)

    cal_days = len(set(dkeys))
    unique_days = sorted(set(dkeys))
    out = {
        "pair": "EURJPY",
        "period": [str(pd.Timestamp(unique_days[0]).date()), str(pd.Timestamp(unique_days[-1]).date())],
        "rule": "SELL on YH touch, BUY on YL touch",
        "tp_pips": TP_PIPS,
        "sl_pips": SL_PIPS,
        "forced_close": "none",
        "spread_pips": SPREAD_PIPS,
        "days": cal_days,
        "buy_trades_opened": buy_trades,
        "sell_trades_opened": sell_trades,
        "buy_wins": buy_wins,
        "buy_losses": buy_losses,
        "sell_wins": sell_wins,
        "sell_losses": sell_losses,
        "open_positions_at_end": len(open_positions),
        "realized_buy_cal_ppd": round(buy_total / max(cal_days, 1), 4),
        "realized_sell_cal_ppd": round(sell_total / max(cal_days, 1), 4),
        "realized_cal_ppd": round(total / max(cal_days, 1), 4),
        "realized_total_pips": round(total, 4),
        "unrealized_pips_at_last_close": round(unrealized, 4),
    }
    out_json = OUT_DIR / "yh_yl_touch_reversal_fixed_50tp_300sl_eurjpy_no_forced_close.json"
    out_json.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=2))
    print(f"saved={out_json}")


if __name__ == "__main__":
    main()

