from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

ROOT = Path(r"C:\Users\temen\5 joulu\TrailingStopLoss pivot peak trading 6 march 2026")
M15_PATH = ROOT / "forex raw data" / "Dukascopy" / "weekly_monthly_analysis" / "2003_20260306_15m" / "eurjpy" / "derived_15m" / "eurjpy_15m_20030804_20260306.csv"
DAILY_PATH = ROOT / "pair_workspaces" / "eurjpy_2003_20260306_15m_v1" / "eurjpy_daily_ohlcv_ver3.csv"
OUT_DIR = Path(__file__).parent

START_DATE = pd.Timestamp("2003-01-01")
END_DATE = pd.Timestamp("2015-12-31")
PIP = 0.01
SPREAD_PIPS = 1.0
TP_MULT = 1.0
SL_MULT = 0.70


def load_data():
    d = pd.read_csv(DAILY_PATH, usecols=["Date", "Open", "High", "Low", "Close"], low_memory=False)
    d["Date"] = pd.to_datetime(d["Date"], errors="coerce")
    for c in ["Open", "High", "Low", "Close"]:
        d[c] = pd.to_numeric(d[c], errors="coerce")
    d = d.dropna().sort_values("Date").reset_index(drop=True)
    d["dkey"] = d["Date"].dt.normalize()

    # previous-day pivot ladder and range
    d["yh"] = d["High"].shift(1)
    d["yl"] = d["Low"].shift(1)
    d["yc"] = d["Close"].shift(1)
    d["prev_range"] = d["yh"] - d["yl"]
    d["pp"] = (d["yh"] + d["yl"] + d["yc"]) / 3.0
    d["r1"] = 2.0 * d["pp"] - d["yl"]
    d["s1"] = 2.0 * d["pp"] - d["yh"]
    d = d.dropna(subset=["prev_range", "pp", "r1", "s1"]).copy()

    m = pd.read_csv(M15_PATH, usecols=["Datetime", "Open", "High", "Low", "Close"], low_memory=False)
    m["Datetime"] = pd.to_datetime(m["Datetime"], errors="coerce")
    for c in ["Open", "High", "Low", "Close"]:
        m[c] = pd.to_numeric(m[c], errors="coerce")
    m = m.dropna().sort_values("Datetime").reset_index(drop=True)
    m["dkey"] = m["Datetime"].dt.normalize()

    # full 96-bar days only
    full_days = set(m.groupby("dkey").size().pipe(lambda s: s[s == 96]).index)
    m = m[m["dkey"].isin(full_days)].copy().reset_index(drop=True)

    i0_map = {
        r["dkey"]: int(r["index"])
        for _, r in m.groupby("dkey").head(1).reset_index()[["dkey", "index"]].iterrows()
    }

    info = {}
    for _, r in d.iterrows():
        k = r["dkey"]
        if not (START_DATE <= k <= END_DATE):
            continue
        if k not in i0_map:
            continue
        info[k] = {
            "open": float(r["Open"]),
            "pp": float(r["pp"]),
            "r1": float(r["r1"]),
            "s1": float(r["s1"]),
            "prev_range": float(r["prev_range"]),
            "i0": i0_map[k],
        }
    return m, info


def close_pips(side: str, entry: float, px: float) -> float:
    if side == "BUY":
        return ((px - entry) / PIP) - SPREAD_PIPS
    return ((entry - px) / PIP) - SPREAD_PIPS


def build_entries(info: dict):
    entries = []
    for d in sorted(info.keys()):
        v = info[d]
        entry = v["open"]
        pp = v["pp"]
        r1 = v["r1"]
        s1 = v["s1"]
        rng = v["prev_range"]
        if rng <= 0:
            continue

        # user rule: sell above pivot (within PP..R1), buy under (within S1..PP)
        if pp <= entry <= r1:
            side = "SELL"
            tp = entry - TP_MULT * rng
            sl = entry + SL_MULT * rng
        elif s1 <= entry <= pp:
            side = "BUY"
            tp = entry + TP_MULT * rng
            sl = entry - SL_MULT * rng
        else:
            continue

        entries.append(
            {
                "date": d,
                "i0": v["i0"],
                "side": side,
                "entry": entry,
                "tp": tp,
                "sl": sl,
            }
        )
    return entries


def run_backtest(m: pd.DataFrame, entries: list[dict]):
    hi = m["High"].to_numpy(float)
    lo = m["Low"].to_numpy(float)
    cl = m["Close"].to_numpy(float)

    total_pips = 0.0
    wins = losses = 0
    end_exits = 0
    buys = sells = 0

    for e in entries:
        side = e["side"]
        if side == "BUY":
            buys += 1
        else:
            sells += 1

        hit = False
        px = cl[-1]
        for j in range(e["i0"], len(hi)):
            if side == "BUY":
                hit_sl = lo[j] <= e["sl"]
                hit_tp = hi[j] >= e["tp"]
            else:
                hit_sl = hi[j] >= e["sl"]
                hit_tp = lo[j] <= e["tp"]

            if hit_sl and hit_tp:
                px = e["sl"]  # conservative tie-break
                hit = True
                break
            if hit_sl:
                px = e["sl"]
                hit = True
                break
            if hit_tp:
                px = e["tp"]
                hit = True
                break

        if not hit:
            end_exits += 1

        p = close_pips(side, e["entry"], px)
        total_pips += p
        if p > 0:
            wins += 1
        else:
            losses += 1

    trades = wins + losses
    days = len(set(e["date"] for e in entries))
    return {
        "pair": "EURJPY",
        "period": [str(START_DATE.date()), str(END_DATE.date())],
        "rule": "entry only if open in [S1,PP] buy or [PP,R1] sell",
        "tp_rule": "TP distance = 100% previous-day range",
        "sl_rule": "SL distance = 70% previous-day range",
        "no_day_end_close": True,
        "trades": trades,
        "buy_trades": buys,
        "sell_trades": sells,
        "wins": wins,
        "losses": losses,
        "win_rate": round(wins / max(trades, 1), 6),
        "total_pips": round(total_pips, 4),
        "cal_ppd": round(total_pips / max(days, 1), 4),
        "avg_pips_per_trade": round(total_pips / max(trades, 1), 4),
        "dataset_end_exits": end_exits,
    }


def main():
    m, info = load_data()
    entries = build_entries(info)
    out = run_backtest(m, entries)

    out_json = OUT_DIR / "dayopen_between_pp_s1r1_prev_range_tp100_sl70_2003_2015_eurjpy.json"
    out_json.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=2))
    print(f"saved={out_json}")


if __name__ == "__main__":
    main()

