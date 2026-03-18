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
TP_MULT = 1.3
SL_MULT = 1.0


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
    d["range_prev"] = d["yh"] - d["yl"]

    m = pd.read_csv(M15_PATH, usecols=["Datetime", "Open", "High", "Low", "Close"], low_memory=False)
    m["Datetime"] = pd.to_datetime(m["Datetime"], errors="coerce")
    for c in ["Open", "High", "Low", "Close"]:
        m[c] = pd.to_numeric(m[c], errors="coerce")
    m = m.dropna().sort_values("Datetime").reset_index(drop=True)
    m["dkey"] = m["Datetime"].dt.normalize()

    # Full day bars only
    full = m.groupby("dkey").size()
    full_days = set(full[full == 96].index)
    m = m[m["dkey"].isin(full_days)].copy().reset_index(drop=True)

    first_idx = m.groupby("dkey").head(1).reset_index()[["dkey", "index"]]
    day_to_i0 = {r["dkey"]: int(r["index"]) for _, r in first_idx.iterrows()}

    info = {}
    for _, r in d.iterrows():
        k = r["dkey"]
        if k in day_to_i0 and float(r["range_prev"]) > 0:
            info[k] = {"range_prev": float(r["range_prev"]), "i0": day_to_i0[k]}
    return m, info


def close_pips(side: str, entry: float, px: float) -> float:
    if side == "BUY":
        return ((px - entry) / PIP) - SPREAD_PIPS
    return ((entry - px) / PIP) - SPREAD_PIPS


def main():
    m, info = load_data()
    highs = m["High"].to_numpy(float)
    lows = m["Low"].to_numpy(float)
    closes = m["Close"].to_numpy(float)

    open_positions = []
    buy_opened = sell_opened = 0
    buy_wins = buy_losses = 0
    sell_wins = sell_losses = 0
    buy_realized = sell_realized = total_realized = 0.0

    days = sorted(info.keys())

    # Open both sides at each day open.
    for d in days:
        i0 = info[d]["i0"]
        entry = float(m.at[i0, "Open"])
        r = info[d]["range_prev"]
        tp_dist = TP_MULT * r
        sl_dist = SL_MULT * r

        open_positions.append(
            {"side": "BUY", "entry": entry, "tp": entry + tp_dist, "sl": entry - sl_dist, "open_from": i0}
        )
        open_positions.append(
            {"side": "SELL", "entry": entry, "tp": entry - tp_dist, "sl": entry + sl_dist, "open_from": i0}
        )
        buy_opened += 1
        sell_opened += 1

    # Simulate forward with no forced close.
    still_open = []
    for p in open_positions:
        side = p["side"]
        entry = p["entry"]
        tp = p["tp"]
        sl = p["sl"]
        done = False
        for j in range(p["open_from"], len(highs)):
            hi = highs[j]
            lo = lows[j]
            if side == "BUY":
                hit_sl = lo <= sl
                hit_tp = hi >= tp
            else:
                hit_sl = hi >= sl
                hit_tp = lo <= tp

            if hit_sl and hit_tp:
                px = sl
            elif hit_sl:
                px = sl
            elif hit_tp:
                px = tp
            else:
                continue

            pips = close_pips(side, entry, px)
            total_realized += pips
            if side == "BUY":
                buy_realized += pips
                if px == tp:
                    buy_wins += 1
                else:
                    buy_losses += 1
            else:
                sell_realized += pips
                if px == tp:
                    sell_wins += 1
                else:
                    sell_losses += 1
            done = True
            break

        if not done:
            still_open.append(p)

    # Unrealized at last close (for visibility only)
    last_c = float(closes[-1])
    unrealized = 0.0
    for p in still_open:
        unrealized += close_pips(p["side"], p["entry"], last_c)

    cal_days = len(days)
    out = {
        "pair": "EURJPY",
        "period": [str(days[0].date()), str(days[-1].date())],
        "rule": "At each day open open BUY+SELL; TP=1.3*yesterday_range; SL=1.0*yesterday_range",
        "forced_close": "none",
        "spread_pips": SPREAD_PIPS,
        "days": cal_days,
        "buy_opened": buy_opened,
        "sell_opened": sell_opened,
        "buy_wins": buy_wins,
        "buy_losses": buy_losses,
        "sell_wins": sell_wins,
        "sell_losses": sell_losses,
        "open_positions_at_end": len(still_open),
        "realized_buy_cal_ppd": round(buy_realized / max(cal_days, 1), 4),
        "realized_sell_cal_ppd": round(sell_realized / max(cal_days, 1), 4),
        "realized_cal_ppd": round(total_realized / max(cal_days, 1), 4),
        "realized_total_pips": round(total_realized, 4),
        "unrealized_pips_at_last_close": round(unrealized, 4),
    }

    out_json = OUT_DIR / "daily_open_both_sides_yday_range_tp13_sl1_eurjpy_no_forced_close.json"
    out_json.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=2))
    print(f"saved={out_json}")


if __name__ == "__main__":
    main()

