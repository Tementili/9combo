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
START_DATE = pd.Timestamp("2003-01-01")
END_DATE = pd.Timestamp("2015-12-31")

ENTRY_NORM = 0.225   # 22.5% away from pivot
SL_NORM = 0.55       # 55% away from pivot (same side as entry)
TP_NORM = 0.5097     # 50.97% beyond pivot (pivot-centered coordinate)
MIN_SIDE_RANGE_PIPS = 10.0


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
    d["r2"] = d["pp"] + (d["yh"] - d["yl"])
    d["s2"] = d["pp"] - (d["yh"] - d["yl"])
    d = d.dropna(subset=["pp", "r2", "s2"]).copy()
    d["dkey"] = d["Date"].dt.normalize()

    m = pd.read_csv(M15_PATH, usecols=["Datetime", "High", "Low", "Close"], low_memory=False)
    m["Datetime"] = pd.to_datetime(m["Datetime"], errors="coerce")
    for c in ["High", "Low", "Close"]:
        m[c] = pd.to_numeric(m[c], errors="coerce")
    m = m.dropna().sort_values("Datetime").reset_index(drop=True)
    m["dkey"] = m["Datetime"].dt.normalize()

    full = m.groupby("dkey").size()
    full_days = set(full[full == 96].index)
    m = m[m["dkey"].isin(full_days)].copy()

    bars = {}
    for dkey, g in m.groupby("dkey", sort=True):
        bars[dkey] = {
            "h": g["High"].to_numpy(float),
            "l": g["Low"].to_numpy(float),
            "c": g["Close"].to_numpy(float),
        }

    info = {}
    for _, r in d.iterrows():
        k = r["dkey"]
        if START_DATE <= k <= END_DATE and k in bars:
            info[k] = {"pp": float(r["pp"]), "r2": float(r["r2"]), "s2": float(r["s2"])}
    return sorted(info.keys()), info, bars


def pips_from_trade(side: str, entry_px: float, exit_px: float) -> float:
    if side == "BUY":
        return ((exit_px - entry_px) / PIP) - SPREAD_PIPS
    return ((entry_px - exit_px) / PIP) - SPREAD_PIPS


def run(days, info, bars):
    min_side_px = MIN_SIDE_RANGE_PIPS * PIP
    total_pips = 0.0
    wins = 0
    losses = 0
    time_exits = 0
    skipped_ambiguous = 0
    no_entry = 0
    buys = 0
    sells = 0

    rows = []
    for d in days:
        pp = info[d]["pp"]
        up_rng = info[d]["r2"] - pp
        dn_rng = pp - info[d]["s2"]
        if up_rng < min_side_px or dn_rng < min_side_px:
            no_entry += 1
            continue

        buy_entry = pp - ENTRY_NORM * dn_rng
        buy_sl = pp - SL_NORM * dn_rng
        buy_tp = pp + TP_NORM * dn_rng

        sell_entry = pp + ENTRY_NORM * up_rng
        sell_sl = pp + SL_NORM * up_rng
        sell_tp = pp - TP_NORM * up_rng

        h = bars[d]["h"]
        l = bars[d]["l"]
        c = bars[d]["c"]

        # first touch of either entry level
        side = None
        entry_px = sl_px = tp_px = None
        trig_i = None
        ambiguous = False
        for i in range(96):
            hit_buy = l[i] <= buy_entry <= h[i]
            hit_sell = l[i] <= sell_entry <= h[i]
            if hit_buy and hit_sell:
                ambiguous = True
                break
            if hit_buy:
                side = "BUY"
                entry_px = buy_entry
                sl_px = buy_sl
                tp_px = buy_tp
                trig_i = i
                buys += 1
                break
            if hit_sell:
                side = "SELL"
                entry_px = sell_entry
                sl_px = sell_sl
                tp_px = sell_tp
                trig_i = i
                sells += 1
                break

        if ambiguous:
            skipped_ambiguous += 1
            continue
        if side is None:
            no_entry += 1
            continue

        # Manage from next bar to avoid intrabar ordering guess.
        start_i = trig_i + 1
        if start_i >= 96:
            # no bar left -> day close
            px = c[-1]
            pips = pips_from_trade(side, entry_px, px)
            total_pips += pips
            time_exits += 1
            if pips >= 0:
                wins += 1
                outcome = "TIME_WIN"
            else:
                losses += 1
                outcome = "TIME_LOSS"
            rows.append({"date": d, "side": side, "outcome": outcome, "pips": pips})
            continue

        done = False
        for j in range(start_i, 96):
            hi = h[j]
            lo = l[j]
            if side == "BUY":
                hit_sl = lo <= sl_px
                hit_tp = hi >= tp_px
            else:
                hit_sl = hi >= sl_px
                hit_tp = lo <= tp_px

            if hit_sl and hit_tp:
                # conservative tie-break
                exit_px = sl_px
                outcome = "SL"
                done = True
            elif hit_sl:
                exit_px = sl_px
                outcome = "SL"
                done = True
            elif hit_tp:
                exit_px = tp_px
                outcome = "TP"
                done = True
            else:
                continue

            pips = pips_from_trade(side, entry_px, exit_px)
            total_pips += pips
            if outcome == "TP":
                wins += 1
            else:
                losses += 1
            rows.append({"date": d, "side": side, "outcome": outcome, "pips": pips})
            break

        if not done:
            exit_px = c[-1]
            pips = pips_from_trade(side, entry_px, exit_px)
            total_pips += pips
            time_exits += 1
            if pips >= 0:
                wins += 1
                outcome = "TIME_WIN"
            else:
                losses += 1
                outcome = "TIME_LOSS"
            rows.append({"date": d, "side": side, "outcome": outcome, "pips": pips})

    trades = wins + losses
    cal_days = len(days)
    return {
        "pair": "EURJPY",
        "period": [str(START_DATE.date()), str(END_DATE.date())],
        "entry_norm": ENTRY_NORM,
        "sl_norm": SL_NORM,
        "tp_norm": TP_NORM,
        "scale": "BUY uses PP->S2 as 1.0; SELL uses PP->R2 as 1.0; pivot-centered TP",
        "trading_window": "same day (entry touch to day close)",
        "spread_pips": SPREAD_PIPS,
        "days": cal_days,
        "trades": trades,
        "wins": wins,
        "losses": losses,
        "time_exits": time_exits,
        "win_rate": round(wins / max(trades, 1), 6),
        "buys": buys,
        "sells": sells,
        "skipped_ambiguous_first_touch": skipped_ambiguous,
        "no_entry_days": no_entry,
        "total_pips": round(total_pips, 4),
        "cal_ppd": round(total_pips / max(cal_days, 1), 4),
        "avg_pips_per_trade": round(total_pips / max(trades, 1), 4),
        "trade_rows": rows,
    }


def main():
    days, info, bars = load_data()
    out = run(days, info, bars)
    rows = out.pop("trade_rows")

    out_json = OUT_DIR / "pivot_norm_entry0225_sl055_tp05097_2003_2015_eurjpy.json"
    out_csv = OUT_DIR / "pivot_norm_entry0225_sl055_tp05097_2003_2015_eurjpy_trades.csv"
    out_json.write_text(json.dumps(out, indent=2), encoding="utf-8")
    pd.DataFrame(rows).to_csv(out_csv, index=False)

    print("RESULT")
    print(json.dumps(out, indent=2))
    print(f"saved={out_json}")
    print(f"saved={out_csv}")


if __name__ == "__main__":
    main()

