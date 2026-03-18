from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

ROOT = Path(r"C:\Users\temen\5 joulu\TrailingStopLoss pivot peak trading 6 march 2026")
M15_PATH = ROOT / "forex raw data" / "Dukascopy" / "weekly_monthly_analysis" / "2003_20260306_15m" / "eurjpy" / "derived_15m" / "eurjpy_15m_20030804_20260306.csv"
DAILY_PATH = ROOT / "pair_workspaces" / "eurjpy_2003_20260306_15m_v1" / "eurjpy_daily_ohlcv_ver3.csv"
OUT_DIR = Path(__file__).parent

PAIR = "eurjpy"
PIP = 0.01
SPREAD_PIPS = 1.0
BUY_SIZE = 1.00
SELL_SIZE = 1.27

# Buy TP at yesterday high or above.
BUY_TP_ABOVE_YH_PIPS_GRID = [0, 10, 20, 30, 40, 50, 75, 100, 125, 150]
# Sell SL above yesterday high.
SELL_SL_ABOVE_YH_PIPS_GRID = [50, 100, 150, 200, 250, 300, 400]
# After buy TP hits, move sell TP to YH - x pips.
SELL_TP_AFTER_BUY_BELOW_YH_PIPS_GRID = [0, 10, 20, 30, 40, 50, 75, 100]


def load_data():
    d = pd.read_csv(DAILY_PATH, usecols=["Date", "High", "Low", "Close"], low_memory=False)
    d["Date"] = pd.to_datetime(d["Date"], errors="coerce")
    for c in ["High", "Low", "Close"]:
        d[c] = pd.to_numeric(d[c], errors="coerce")
    d = d.dropna().sort_values("Date").reset_index(drop=True)
    d["yh"] = d["High"].shift(1)
    d["yl"] = d["Low"].shift(1)
    d["yc"] = d["Close"].shift(1)
    d["pp"] = (d["yh"] + d["yl"] + d["yc"]) / 3.0
    d = d.dropna(subset=["pp", "yh", "yl"]).copy()
    d["dkey"] = d["Date"].dt.normalize()

    m = pd.read_csv(M15_PATH, usecols=["Datetime", "Open", "High", "Low", "Close"], low_memory=False)
    m["Datetime"] = pd.to_datetime(m["Datetime"], errors="coerce")
    for c in ["Open", "High", "Low", "Close"]:
        m[c] = pd.to_numeric(m[c], errors="coerce")
    m = m.dropna().sort_values("Datetime").reset_index(drop=True)
    m["dkey"] = m["Datetime"].dt.normalize()

    # Full 96-bar days only.
    full_days = m.groupby("dkey").size()
    full_days = set(full_days[full_days == 96].index)
    m = m[m["dkey"].isin(full_days)].copy()

    bars = {}
    for dkey, g in m.groupby("dkey", sort=True):
        bars[dkey] = {
            "open": float(g["Open"].iloc[0]),
            "h": g["High"].to_list(),
            "l": g["Low"].to_list(),
            "last_c": float(g["Close"].iloc[-1]),
        }

    info = {}
    for _, r in d.iterrows():
        k = r["dkey"]
        if k in bars:
            info[k] = {
                "pp": float(r["pp"]),
                "yh": float(r["yh"]),
                "yl": float(r["yl"]),
            }

    days = sorted(info.keys())
    return days, info, bars


def run_one(days, info, bars, buy_tp_above_yh_pips: int, sell_sl_above_yh_pips: int, sell_tp_after_buy_below_yh_pips: int):
    total_weighted_pips = 0.0
    buy_weighted = 0.0
    sell_weighted = 0.0
    trades = 0
    above_pivot_days = 0

    for d in days:
        lv = info[d]
        b = bars[d]
        day_open = b["open"]
        pp = lv["pp"]
        yh = lv["yh"]
        yl = lv["yl"]
        if day_open <= pp:
            continue
        above_pivot_days += 1

        buy_entry = day_open
        sell_entry = day_open

        buy_sl = yl
        buy_tp = yh + buy_tp_above_yh_pips * PIP

        sell_sl = yh + sell_sl_above_yh_pips * PIP
        sell_tp = yl  # initial as requested
        sell_tp_pending = None  # apply next bar after buy TP hit

        buy_open = True
        sell_open = True

        h_arr = b["h"]
        l_arr = b["l"]

        for i in range(len(h_arr)):
            hi = h_arr[i]
            lo = l_arr[i]

            if sell_tp_pending is not None:
                sell_tp = sell_tp_pending
                sell_tp_pending = None

            # BUY leg
            buy_tp_hit = False
            if buy_open:
                hit_sl = lo <= buy_sl
                hit_tp = hi >= buy_tp
                if hit_sl and hit_tp:
                    px = buy_sl
                    pips = ((px - buy_entry) / PIP) - SPREAD_PIPS
                    w = pips * BUY_SIZE
                    buy_weighted += w
                    total_weighted_pips += w
                    buy_open = False
                    trades += 1
                elif hit_sl:
                    px = buy_sl
                    pips = ((px - buy_entry) / PIP) - SPREAD_PIPS
                    w = pips * BUY_SIZE
                    buy_weighted += w
                    total_weighted_pips += w
                    buy_open = False
                    trades += 1
                elif hit_tp:
                    px = buy_tp
                    pips = ((px - buy_entry) / PIP) - SPREAD_PIPS
                    w = pips * BUY_SIZE
                    buy_weighted += w
                    total_weighted_pips += w
                    buy_open = False
                    trades += 1
                    buy_tp_hit = True

            # After buy TP, reset sell TP to YH - offset (next bar activation).
            if buy_tp_hit and sell_open:
                sell_tp_pending = yh - sell_tp_after_buy_below_yh_pips * PIP

            # SELL leg
            if sell_open:
                hit_sl = hi >= sell_sl
                hit_tp = lo <= sell_tp
                if hit_sl and hit_tp:
                    px = sell_sl
                    pips = ((sell_entry - px) / PIP) - SPREAD_PIPS
                    w = pips * SELL_SIZE
                    sell_weighted += w
                    total_weighted_pips += w
                    sell_open = False
                    trades += 1
                elif hit_sl:
                    px = sell_sl
                    pips = ((sell_entry - px) / PIP) - SPREAD_PIPS
                    w = pips * SELL_SIZE
                    sell_weighted += w
                    total_weighted_pips += w
                    sell_open = False
                    trades += 1
                elif hit_tp:
                    px = sell_tp
                    pips = ((sell_entry - px) / PIP) - SPREAD_PIPS
                    w = pips * SELL_SIZE
                    sell_weighted += w
                    total_weighted_pips += w
                    sell_open = False
                    trades += 1

            if (not buy_open) and (not sell_open):
                break

        # EOD close if still open.
        last_c = b["last_c"]
        if buy_open:
            pips = ((last_c - buy_entry) / PIP) - SPREAD_PIPS
            w = pips * BUY_SIZE
            buy_weighted += w
            total_weighted_pips += w
            trades += 1
        if sell_open:
            pips = ((sell_entry - last_c) / PIP) - SPREAD_PIPS
            w = pips * SELL_SIZE
            sell_weighted += w
            total_weighted_pips += w
            trades += 1

    cal_days = len(days)
    return {
        "buy_tp_above_yh_pips": buy_tp_above_yh_pips,
        "sell_sl_above_yh_pips": sell_sl_above_yh_pips,
        "sell_tp_after_buy_below_yh_pips": sell_tp_after_buy_below_yh_pips,
        "above_pivot_days": above_pivot_days,
        "closed_legs": trades,
        "weighted_total_pips": round(total_weighted_pips, 4),
        "weighted_buy_pips": round(buy_weighted, 4),
        "weighted_sell_pips": round(sell_weighted, 4),
        "cal_ppd_weighted": round(total_weighted_pips / max(cal_days, 1), 4),
    }


def main():
    days, info, bars = load_data()
    rows = []
    total_cfg = (
        len(BUY_TP_ABOVE_YH_PIPS_GRID)
        * len(SELL_SL_ABOVE_YH_PIPS_GRID)
        * len(SELL_TP_AFTER_BUY_BELOW_YH_PIPS_GRID)
    )
    done = 0
    for btp in BUY_TP_ABOVE_YH_PIPS_GRID:
        for ssl in SELL_SL_ABOVE_YH_PIPS_GRID:
            for stp2 in SELL_TP_AFTER_BUY_BELOW_YH_PIPS_GRID:
                r = run_one(days, info, bars, btp, ssl, stp2)
                rows.append(r)
                done += 1
                if done % 50 == 0:
                    best_so_far = max(x["cal_ppd_weighted"] for x in rows)
                    print(f"{done}/{total_cfg} cfg ... best_so_far={best_so_far:+.4f}")

    df = pd.DataFrame(rows).sort_values("cal_ppd_weighted", ascending=False).reset_index(drop=True)
    best = df.iloc[0].to_dict()
    out = {
        "pair": "EURJPY",
        "period": [str(days[0].date()), str(days[-1].date())],
        "rule": "if day open > pivot: open BUY(1.00) + SELL(1.27) at day open",
        "fixed_rules": {
            "sell_initial_tp": "yesterday low",
            "buy_sl": "yesterday low",
            "cost_pips_per_leg": SPREAD_PIPS,
        },
        "optimized": {
            "buy_tp_above_yh_pips_grid": BUY_TP_ABOVE_YH_PIPS_GRID,
            "sell_sl_above_yh_pips_grid": SELL_SL_ABOVE_YH_PIPS_GRID,
            "sell_tp_after_buy_below_yh_pips_grid": SELL_TP_AFTER_BUY_BELOW_YH_PIPS_GRID,
        },
        "best": best,
        "top20": df.head(20).to_dict(orient="records"),
    }

    out_json = OUT_DIR / "above_pivot_buy100_sell127_optimize_eurjpy_full_days.json"
    out_csv = OUT_DIR / "above_pivot_buy100_sell127_optimize_eurjpy_full_days.csv"
    out_json.write_text(json.dumps(out, indent=2), encoding="utf-8")
    df.to_csv(out_csv, index=False)

    print("\nBEST:")
    print(best)
    print(f"saved={out_json}")
    print(f"saved={out_csv}")


if __name__ == "__main__":
    main()

