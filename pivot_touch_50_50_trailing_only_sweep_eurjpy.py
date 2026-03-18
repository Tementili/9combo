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
BUY_SIZE = 50.0
SELL_SIZE = 50.0
MAX_HOLD_HOURS = 720  # one month
MAX_HOLD_BARS = MAX_HOLD_HOURS * 4
TRAIL_PIPS_GRID = [25, 40, 50, 60, 75, 100, 125, 150, 175, 200, 225, 250, 300, 350, 400, 500]


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
    d = d.dropna(subset=["pp"]).copy()
    d["dkey"] = d["Date"].dt.normalize()

    m = pd.read_csv(M15_PATH, usecols=["Datetime", "High", "Low", "Close"], low_memory=False)
    m["Datetime"] = pd.to_datetime(m["Datetime"], errors="coerce")
    for c in ["High", "Low", "Close"]:
        m[c] = pd.to_numeric(m[c], errors="coerce")
    m = m.dropna().sort_values("Datetime").reset_index(drop=True)
    m["dkey"] = m["Datetime"].dt.normalize()

    # Full 96-bar days only.
    full = m.groupby("dkey").size()
    full_days = set(full[full == 96].index)
    m = m[m["dkey"].isin(full_days)].copy().reset_index(drop=True)

    day_first = m.groupby("dkey").head(1).reset_index()[["dkey", "index"]]
    day_to_i0 = {r["dkey"]: int(r["index"]) for _, r in day_first.iterrows()}

    pp_map = {}
    for _, r in d.iterrows():
        k = r["dkey"]
        if k in day_to_i0:
            pp_map[k] = {"pp": float(r["pp"]), "i0": day_to_i0[k]}
    return m, pp_map


def close_pips(side: str, entry: float, px: float) -> float:
    if side == "BUY":
        return ((px - entry) / PIP) - SPREAD_PIPS
    return ((entry - px) / PIP) - SPREAD_PIPS


def run_one(m: pd.DataFrame, pp_map: dict, trail_pips: int):
    highs = m["High"].to_numpy(float)
    lows = m["Low"].to_numpy(float)
    closes = m["Close"].to_numpy(float)
    dkeys = m["dkey"].to_numpy()
    trail_px = trail_pips * PIP

    total_weighted = 0.0
    buy_weighted = 0.0
    sell_weighted = 0.0
    entries = 0
    no_touch_days = 0
    wins = losses = 0
    time_exits = 0

    days = sorted(pp_map.keys())
    for d in days:
        pp = pp_map[d]["pp"]
        i0 = pp_map[d]["i0"]

        # real touch of pivot from day open onward
        end_day = i0 + 95
        t_idx = None
        for j in range(i0, min(end_day, len(highs) - 1) + 1):
            if lows[j] <= pp <= highs[j]:
                t_idx = j
                break
        if t_idx is None:
            no_touch_days += 1
            continue
        entries += 1

        start_i = t_idx + 1
        if start_i >= len(highs):
            continue
        end_i = min(start_i + MAX_HOLD_BARS, len(highs) - 1)

        # BUY and SELL at pivot with trailing-stop only.
        pos = [
            {"side": "BUY", "entry": pp, "sl": pp - trail_px, "best": pp, "open": True},
            {"side": "SELL", "entry": pp, "sl": pp + trail_px, "best": pp, "open": True},
        ]

        for j in range(start_i, end_i + 1):
            hi = highs[j]
            lo = lows[j]
            for p in pos:
                if not p["open"]:
                    continue
                if p["side"] == "BUY":
                    p["best"] = max(p["best"], hi)
                    p["sl"] = max(p["sl"], p["best"] - trail_px)
                    hit = lo <= p["sl"]
                else:
                    p["best"] = min(p["best"], lo)
                    p["sl"] = min(p["sl"], p["best"] + trail_px)
                    hit = hi >= p["sl"]

                if hit:
                    px = p["sl"]
                    pips = close_pips(p["side"], p["entry"], px)
                    if p["side"] == "BUY":
                        w = pips * BUY_SIZE
                        buy_weighted += w
                    else:
                        w = pips * SELL_SIZE
                        sell_weighted += w
                    total_weighted += w
                    p["open"] = False
                    if pips >= 0:
                        wins += 1
                    else:
                        losses += 1

            if (not pos[0]["open"]) and (not pos[1]["open"]):
                break

        # month-end close for any still-open legs.
        px_end = closes[end_i]
        for p in pos:
            if p["open"]:
                pips = close_pips(p["side"], p["entry"], px_end)
                if p["side"] == "BUY":
                    w = pips * BUY_SIZE
                    buy_weighted += w
                else:
                    w = pips * SELL_SIZE
                    sell_weighted += w
                total_weighted += w
                p["open"] = False
                time_exits += 1
                if pips >= 0:
                    wins += 1
                else:
                    losses += 1

    cal_days = len(days)
    return {
        "trail_pips": trail_pips,
        "days": cal_days,
        "entries": entries,
        "no_touch_days": no_touch_days,
        "wins": wins,
        "losses": losses,
        "time_exits": time_exits,
        "win_rate": round(wins / max(wins + losses, 1), 6),
        "weighted_buy_pips": round(buy_weighted, 4),
        "weighted_sell_pips": round(sell_weighted, 4),
        "weighted_total_pips": round(total_weighted, 4),
        "cal_ppd_weighted": round(total_weighted / max(cal_days, 1), 4),
        "ppd_per_100_units": round((total_weighted / max(cal_days, 1)) / 100.0, 4),
    }


def main():
    m, pp_map = load_data()
    rows = []
    for tp in TRAIL_PIPS_GRID:
        r = run_one(m, pp_map, tp)
        rows.append(r)
        print(
            f"trail={tp:>3}p  cal_ppd_w={r['cal_ppd_weighted']:+.4f} "
            f"per100={r['ppd_per_100_units']:+.4f}  entries={r['entries']}"
        )

    df = pd.DataFrame(rows).sort_values("cal_ppd_weighted", ascending=False).reset_index(drop=True)
    out = {
        "pair": "EURJPY",
        "rule": "Real pivot touch entry; open BUY50+SELL50; trailing-stop only; run up to 1 month",
        "max_hold_hours": MAX_HOLD_HOURS,
        "spread_pips": SPREAD_PIPS,
        "trail_pips_grid": TRAIL_PIPS_GRID,
        "best": df.iloc[0].to_dict(),
        "all_results": df.to_dict(orient="records"),
    }

    out_json = OUT_DIR / "pivot_touch_50_50_trailing_only_sweep_eurjpy.json"
    out_csv = OUT_DIR / "pivot_touch_50_50_trailing_only_sweep_eurjpy.csv"
    out_json.write_text(json.dumps(out, indent=2), encoding="utf-8")
    df.to_csv(out_csv, index=False)
    print("\nBEST:")
    print(df.iloc[0].to_dict())
    print(f"saved={out_json}")
    print(f"saved={out_csv}")


if __name__ == "__main__":
    main()

