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
BUY_SIZE = 100.0
SELL_SL_ABOVE_YH_PIPS = 250  # fixed as requested

# Optimize how close below YH to start (real touch).
ENTRY_BELOW_YH_PIPS_GRID = [5, 10, 15, 20, 25, 30, 40, 50]
# Optimize buy TP placement relative to YH.
BUY_TP_ABOVE_YH_PIPS_GRID = [0, 10, 20, 30, 40, 50, 75, 100, 125, 150]
# Optimize bet ratio (SELL size = ratio * BUY size).
SELL_BUY_RATIO_GRID = [1.00, 1.10, 1.20, 1.27, 1.30, 1.40, 1.50, 1.60]


def load_data():
    d = pd.read_csv(DAILY_PATH, usecols=["Date", "High", "Low", "Close"], low_memory=False)
    d["Date"] = pd.to_datetime(d["Date"], errors="coerce")
    for c in ["High", "Low", "Close"]:
        d[c] = pd.to_numeric(d[c], errors="coerce")
    d = d.dropna().sort_values("Date").reset_index(drop=True)
    d["yh"] = d["High"].shift(1)
    d["yl"] = d["Low"].shift(1)
    d = d.dropna(subset=["yh", "yl"]).copy()
    d["dkey"] = d["Date"].dt.normalize()

    m = pd.read_csv(M15_PATH, usecols=["Datetime", "Open", "High", "Low", "Close"], low_memory=False)
    m["Datetime"] = pd.to_datetime(m["Datetime"], errors="coerce")
    for c in ["Open", "High", "Low", "Close"]:
        m[c] = pd.to_numeric(m[c], errors="coerce")
    m = m.dropna().sort_values("Datetime").reset_index(drop=True)
    m["dkey"] = m["Datetime"].dt.normalize()

    full_days = m.groupby("dkey").size()
    full_days = set(full_days[full_days == 96].index)
    m = m[m["dkey"].isin(full_days)].copy()

    bars = {}
    for dkey, g in m.groupby("dkey", sort=True):
        bars[dkey] = {
            "h": g["High"].to_list(),
            "l": g["Low"].to_list(),
            "last_c": float(g["Close"].iloc[-1]),
        }

    info = {}
    for _, r in d.iterrows():
        k = r["dkey"]
        if k in bars:
            info[k] = {"yh": float(r["yh"]), "yl": float(r["yl"])}
    days = sorted(info.keys())
    return days, info, bars


def run_one(days, info, bars, entry_below_yh_pips: int, buy_tp_above_yh_pips: int, sell_buy_ratio: float):
    total_weighted_pips = 0.0
    buy_weighted_pips = 0.0
    sell_weighted_pips = 0.0
    trade_days = 0
    filled_days = 0
    buy_wins = 0
    buy_losses = 0
    sell_wins = 0
    sell_losses = 0
    sell_time_exits = 0

    sell_size = BUY_SIZE * sell_buy_ratio

    for d in days:
        lv = info[d]
        b = bars[d]
        yh = lv["yh"]

        entry_px = yh - entry_below_yh_pips * PIP
        buy_tp = yh + buy_tp_above_yh_pips * PIP
        buy_sl = lv["yl"]  # from prior rule family
        sell_sl = yh + SELL_SL_ABOVE_YH_PIPS * PIP

        trade_days += 1
        h_arr = b["h"]
        l_arr = b["l"]

        # real touch start at entry level
        touch_i = None
        for i, (hi, lo) in enumerate(zip(h_arr, l_arr)):
            if lo <= entry_px <= hi:
                touch_i = i
                break
        if touch_i is None:
            continue
        filled_days += 1

        # activate next bar
        i0 = touch_i + 1
        if i0 >= len(h_arr):
            continue

        buy_open = True
        sell_open = True

        for hi, lo in zip(h_arr[i0:], l_arr[i0:]):
            if buy_open:
                hit_sl = lo <= buy_sl
                hit_tp = hi >= buy_tp
                if hit_sl and hit_tp:
                    px = buy_sl
                    pips = ((px - entry_px) / PIP) - SPREAD_PIPS
                    w = pips * BUY_SIZE
                    buy_weighted_pips += w
                    total_weighted_pips += w
                    buy_open = False
                    buy_losses += 1
                elif hit_sl:
                    px = buy_sl
                    pips = ((px - entry_px) / PIP) - SPREAD_PIPS
                    w = pips * BUY_SIZE
                    buy_weighted_pips += w
                    total_weighted_pips += w
                    buy_open = False
                    buy_losses += 1
                elif hit_tp:
                    px = buy_tp
                    pips = ((px - entry_px) / PIP) - SPREAD_PIPS
                    w = pips * BUY_SIZE
                    buy_weighted_pips += w
                    total_weighted_pips += w
                    buy_open = False
                    buy_wins += 1

            if sell_open:
                hit_sl = hi >= sell_sl
                if hit_sl:
                    px = sell_sl
                    pips = ((entry_px - px) / PIP) - SPREAD_PIPS
                    w = pips * sell_size
                    sell_weighted_pips += w
                    total_weighted_pips += w
                    sell_open = False
                    sell_losses += 1

            if (not buy_open) and (not sell_open):
                break

        # EOD close if still open
        last_c = b["last_c"]
        if buy_open:
            pips = ((last_c - entry_px) / PIP) - SPREAD_PIPS
            w = pips * BUY_SIZE
            buy_weighted_pips += w
            total_weighted_pips += w
            buy_open = False
            if pips >= 0:
                buy_wins += 1
            else:
                buy_losses += 1

        if sell_open:
            pips = ((entry_px - last_c) / PIP) - SPREAD_PIPS
            w = pips * sell_size
            sell_weighted_pips += w
            total_weighted_pips += w
            sell_open = False
            sell_time_exits += 1
            if pips >= 0:
                sell_wins += 1
            else:
                sell_losses += 1

    cal_days = len(days)
    return {
        "entry_below_yh_pips": entry_below_yh_pips,
        "buy_tp_above_yh_pips": buy_tp_above_yh_pips,
        "sell_buy_ratio": round(sell_buy_ratio, 4),
        "buy_size": BUY_SIZE,
        "sell_size": round(sell_size, 2),
        "trade_days": trade_days,
        "filled_days": filled_days,
        "filled_day_rate": round(filled_days / max(trade_days, 1), 6),
        "buy_wins": buy_wins,
        "buy_losses": buy_losses,
        "sell_wins": sell_wins,
        "sell_losses": sell_losses,
        "sell_time_exits": sell_time_exits,
        "weighted_buy_pips": round(buy_weighted_pips, 4),
        "weighted_sell_pips": round(sell_weighted_pips, 4),
        "weighted_total_pips": round(total_weighted_pips, 4),
        "cal_ppd_weighted": round(total_weighted_pips / max(cal_days, 1), 4),
    }


def main():
    days, info, bars = load_data()
    rows = []
    total_cfg = len(ENTRY_BELOW_YH_PIPS_GRID) * len(BUY_TP_ABOVE_YH_PIPS_GRID) * len(SELL_BUY_RATIO_GRID)
    done = 0
    for e in ENTRY_BELOW_YH_PIPS_GRID:
        for tp in BUY_TP_ABOVE_YH_PIPS_GRID:
            for r in SELL_BUY_RATIO_GRID:
                row = run_one(days, info, bars, e, tp, r)
                rows.append(row)
                done += 1
                if done % 100 == 0:
                    best_so_far = max(x["cal_ppd_weighted"] for x in rows)
                    print(f"{done}/{total_cfg} cfg ... best_so_far={best_so_far:+.4f}")

    df = pd.DataFrame(rows).sort_values("cal_ppd_weighted", ascending=False).reset_index(drop=True)
    best = df.iloc[0].to_dict()
    out = {
        "pair": "EURJPY",
        "period": [str(days[0].date()), str(days[-1].date())],
        "rule": "real touch at YH-X pips then open BUY(100) + SELL(ratio*100)",
        "fixed": {
            "sell_sl_above_yh_pips": SELL_SL_ABOVE_YH_PIPS,
            "buy_sl": "yesterday low",
            "sell_tp": "none; sell closes by SL or EOD",
            "spread_pips_per_leg": SPREAD_PIPS,
        },
        "grids": {
            "entry_below_yh_pips": ENTRY_BELOW_YH_PIPS_GRID,
            "buy_tp_above_yh_pips": BUY_TP_ABOVE_YH_PIPS_GRID,
            "sell_buy_ratio": SELL_BUY_RATIO_GRID,
        },
        "best": best,
        "top20": df.head(20).to_dict(orient="records"),
    }

    out_json = OUT_DIR / "yh_proximity_entry_buy100_sellratio_optimize_eurjpy_full_days.json"
    out_csv = OUT_DIR / "yh_proximity_entry_buy100_sellratio_optimize_eurjpy_full_days.csv"
    out_json.write_text(json.dumps(out, indent=2), encoding="utf-8")
    df.to_csv(out_csv, index=False)

    print("\nBEST:")
    print(best)
    print(f"saved={out_json}")
    print(f"saved={out_csv}")


if __name__ == "__main__":
    main()

