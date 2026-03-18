from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(r"C:\Users\temen\5 joulu\TrailingStopLoss pivot peak trading 6 march 2026")
M15_PATH = ROOT / "forex raw data" / "Dukascopy" / "weekly_monthly_analysis" / "2003_20260306_15m" / "eurjpy" / "derived_15m" / "eurjpy_15m_20030804_20260306.csv"
DAILY_PATH = ROOT / "pair_workspaces" / "eurjpy_2003_20260306_15m_v1" / "eurjpy_daily_ohlcv_ver3.csv"
OUT_DIR = Path(__file__).parent

START_DATE = pd.Timestamp("2003-01-01")
END_DATE = pd.Timestamp("2015-12-31")
PIP = 0.01
SPREAD_PIPS = 1.0
MAX_BUCKET_NORM = 0.35
MIN_DENOM_PIPS = 10.0
TRAIL_PIPS_GRID = [20, 30, 40, 50, 60, 75, 90, 110, 130, 150, 175, 200, 225, 250, 300, 350, 400]


def load_data():
    d = pd.read_csv(DAILY_PATH, usecols=["Date", "Open", "High", "Low", "Close"], low_memory=False)
    d["Date"] = pd.to_datetime(d["Date"], errors="coerce")
    for c in ["Open", "High", "Low", "Close"]:
        d[c] = pd.to_numeric(d[c], errors="coerce")
    d = d.dropna().sort_values("Date").reset_index(drop=True)
    d["dkey"] = d["Date"].dt.normalize()

    d["yh"] = d["High"].shift(1)
    d["yl"] = d["Low"].shift(1)
    d["yc"] = d["Close"].shift(1)
    d["pp"] = (d["yh"] + d["yl"] + d["yc"]) / 3.0
    d["r2"] = d["pp"] + (d["yh"] - d["yl"])
    d["s2"] = d["pp"] - (d["yh"] - d["yl"])
    d = d.dropna(subset=["pp", "r2", "s2"]).copy()

    m = pd.read_csv(M15_PATH, usecols=["Datetime", "Open", "High", "Low", "Close"], low_memory=False)
    m["Datetime"] = pd.to_datetime(m["Datetime"], errors="coerce")
    for c in ["Open", "High", "Low", "Close"]:
        m[c] = pd.to_numeric(m[c], errors="coerce")
    m = m.dropna().sort_values("Datetime").reset_index(drop=True)
    m["dkey"] = m["Datetime"].dt.normalize()

    full_days = set(m.groupby("dkey").size().pipe(lambda s: s[s == 96]).index)
    m = m[m["dkey"].isin(full_days)].copy().reset_index(drop=True)

    i0_map = {
        r["dkey"]: int(r["index"])
        for _, r in m.groupby("dkey").head(1).reset_index()[["dkey", "index"]].iterrows()
    }

    info = {}
    for _, r in d.iterrows():
        k = r["dkey"]
        if START_DATE <= k <= END_DATE and k in i0_map:
            info[k] = {
                "open": float(r["Open"]),
                "pp": float(r["pp"]),
                "r2": float(r["r2"]),
                "s2": float(r["s2"]),
                "i0": i0_map[k],
            }
    return m, info


def close_pips(side: str, entry: float, px: float) -> float:
    if side == "BUY":
        return ((px - entry) / PIP) - SPREAD_PIPS
    return ((entry - px) / PIP) - SPREAD_PIPS


def build_entries(info: dict):
    entries = []
    min_denom = MIN_DENOM_PIPS * PIP
    for d in sorted(info.keys()):
        v = info[d]
        entry = v["open"]
        pp = v["pp"]
        if entry < pp:
            denom = pp - v["s2"]
            if denom < min_denom:
                continue
            dist_norm = (pp - entry) / denom
            if not (0.0 <= dist_norm <= MAX_BUCKET_NORM):
                continue
            side = "BUY"
        elif entry > pp:
            denom = v["r2"] - pp
            if denom < min_denom:
                continue
            dist_norm = (entry - pp) / denom
            if not (0.0 <= dist_norm <= MAX_BUCKET_NORM):
                continue
            side = "SELL"
        else:
            continue
        entries.append({"date": d, "i0": v["i0"], "side": side, "entry": entry})
    return entries


def run_one(m: pd.DataFrame, entries: list[dict], trail_pips: int):
    hi = m["High"].to_numpy(float)
    lo = m["Low"].to_numpy(float)
    cl = m["Close"].to_numpy(float)
    trail_px = trail_pips * PIP

    total_pips = 0.0
    wins = losses = 0
    end_exits = 0
    buy_n = sell_n = 0

    for e in entries:
        side = e["side"]
        entry = e["entry"]
        i0 = e["i0"]

        if side == "BUY":
            buy_n += 1
            best = entry
            sl = entry - trail_px
            hit = False
            for j in range(i0, len(hi)):
                best = max(best, hi[j])
                sl = max(sl, best - trail_px)  # trailing only
                if lo[j] <= sl:
                    px = sl
                    hit = True
                    break
            if not hit:
                px = cl[-1]
                end_exits += 1
            p = close_pips("BUY", entry, px)
        else:
            sell_n += 1
            best = entry
            sl = entry + trail_px
            hit = False
            for j in range(i0, len(hi)):
                best = min(best, lo[j])
                sl = min(sl, best + trail_px)  # trailing only
                if hi[j] >= sl:
                    px = sl
                    hit = True
                    break
            if not hit:
                px = cl[-1]
                end_exits += 1
            p = close_pips("SELL", entry, px)

        total_pips += p
        if p > 0:
            wins += 1
        else:
            losses += 1

    trades = wins + losses
    days = len(set(e["date"] for e in entries))
    return {
        "trail_pips": trail_pips,
        "days": days,
        "trades": trades,
        "buy_trades": buy_n,
        "sell_trades": sell_n,
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
    rows = []
    for tp in TRAIL_PIPS_GRID:
        r = run_one(m, entries, tp)
        rows.append(r)
        print(
            f"trail={tp:>3}p trades={r['trades']} wr={r['win_rate']:.4f} "
            f"ppd={r['cal_ppd']:+.4f} p/trade={r['avg_pips_per_trade']:+.4f}"
        )

    df = pd.DataFrame(rows).sort_values("cal_ppd", ascending=False).reset_index(drop=True)
    out = {
        "pair": "EURJPY",
        "period": [str(START_DATE.date()), str(END_DATE.date())],
        "entry_rule": "day open; open<PP BUY / open>PP SELL; keep only distance 0-35%",
        "risk_rule": "trailing stop only (no hard SL, no fixed TP)",
        "no_day_end_close": True,
        "trail_pips_grid": TRAIL_PIPS_GRID,
        "entries_used": len(entries),
        "best_by_cal_ppd": df.iloc[0].to_dict(),
        "all_results": df.to_dict(orient="records"),
    }

    out_json = OUT_DIR / "dayopen_pivotside_0_35_trailing_only_sweep_2003_2015_eurjpy.json"
    out_csv = OUT_DIR / "dayopen_pivotside_0_35_trailing_only_sweep_2003_2015_eurjpy.csv"
    out_json.write_text(json.dumps(out, indent=2), encoding="utf-8")
    df.to_csv(out_csv, index=False)

    print("\nBEST")
    print(df.iloc[0].to_dict())
    print(f"saved={out_json}")
    print(f"saved={out_csv}")


if __name__ == "__main__":
    main()

