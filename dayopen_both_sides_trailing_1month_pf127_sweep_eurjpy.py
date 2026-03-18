from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

ROOT = Path(r"C:\Users\temen\5 joulu\TrailingStopLoss pivot peak trading 6 march 2026")
M15_PATH = ROOT / "forex raw data" / "Dukascopy" / "weekly_monthly_analysis" / "2003_20260306_15m" / "eurjpy" / "derived_15m" / "eurjpy_15m_20030804_20260306.csv"
OUT_DIR = Path(__file__).parent

PIP = 0.01
SPREAD_PIPS = 1.0
MAX_HOLD_HOURS = 720  # at least month
MAX_HOLD_BARS = MAX_HOLD_HOURS * 4
TARGET_PF = 1.27
TRAIL_PIPS_GRID = [50, 75, 100, 125, 150, 175, 200, 225, 250, 300, 350, 400, 500]


def load_data():
    m = pd.read_csv(M15_PATH, usecols=["Datetime", "Open", "High", "Low", "Close"], low_memory=False)
    m["Datetime"] = pd.to_datetime(m["Datetime"], errors="coerce")
    for c in ["Open", "High", "Low", "Close"]:
        m[c] = pd.to_numeric(m[c], errors="coerce")
    m = m.dropna().sort_values("Datetime").reset_index(drop=True)
    m["dkey"] = m["Datetime"].dt.normalize()

    # Full 96-bar days to keep day opens consistent.
    full = m.groupby("dkey").size()
    full_days = set(full[full == 96].index)
    m = m[m["dkey"].isin(full_days)].copy().reset_index(drop=True)

    first = m.groupby("dkey").head(1).reset_index()[["dkey", "index"]]
    day_i0 = [int(r["index"]) for _, r in first.sort_values("dkey").iterrows()]
    return m, day_i0


def close_pips(side: str, entry: float, px: float) -> float:
    if side == "BUY":
        return ((px - entry) / PIP) - SPREAD_PIPS
    return ((entry - px) / PIP) - SPREAD_PIPS


def run_one(m: pd.DataFrame, day_i0: list[int], trail_pips: int):
    highs = m["High"].to_numpy(float)
    lows = m["Low"].to_numpy(float)
    closes = m["Close"].to_numpy(float)
    opens = m["Open"].to_numpy(float)

    dpx = trail_pips * PIP
    realized = 0.0
    gross_win = 0.0
    gross_loss = 0.0
    wins = 0
    losses = 0
    time_exits = 0

    for i0 in day_i0:
        entry = opens[i0]
        end_i = min(i0 + MAX_HOLD_BARS, len(highs) - 1)

        # BUY and SELL from day open.
        positions = [
            {"side": "BUY", "entry": entry, "sl": entry - dpx, "best": entry, "open": True},
            {"side": "SELL", "entry": entry, "sl": entry + dpx, "best": entry, "open": True},
        ]

        for j in range(i0 + 1, end_i + 1):
            hi = highs[j]
            lo = lows[j]
            for p in positions:
                if not p["open"]:
                    continue
                if p["side"] == "BUY":
                    p["best"] = max(p["best"], hi)
                    p["sl"] = max(p["sl"], p["best"] - dpx)
                    hit = lo <= p["sl"]
                else:
                    p["best"] = min(p["best"], lo)
                    p["sl"] = min(p["sl"], p["best"] + dpx)
                    hit = hi >= p["sl"]

                if hit:
                    px = p["sl"]
                    pnl = close_pips(p["side"], p["entry"], px)
                    realized += pnl
                    if pnl >= 0:
                        wins += 1
                        gross_win += pnl
                    else:
                        losses += 1
                        gross_loss += -pnl
                    p["open"] = False

            if (not positions[0]["open"]) and (not positions[1]["open"]):
                break

        # month-end close for any still open
        px_close = closes[end_i]
        for p in positions:
            if p["open"]:
                pnl = close_pips(p["side"], p["entry"], px_close)
                realized += pnl
                if pnl >= 0:
                    wins += 1
                    gross_win += pnl
                else:
                    losses += 1
                    gross_loss += -pnl
                time_exits += 1
                p["open"] = False

    cal_days = len(day_i0)
    pf = (gross_win / gross_loss) if gross_loss > 0 else None
    avg_win = (gross_win / wins) if wins > 0 else 0.0
    avg_loss = (gross_loss / losses) if losses > 0 else 0.0
    wl = (wins / losses) if losses > 0 else None
    return {
        "trail_pips": trail_pips,
        "max_hold_hours": MAX_HOLD_HOURS,
        "wins": wins,
        "losses": losses,
        "time_exits": time_exits,
        "win_rate": round(wins / max(wins + losses, 1), 6),
        "profit_factor": round(pf, 6) if pf is not None else None,
        "avg_win_pips": round(avg_win, 4),
        "avg_loss_pips": round(avg_loss, 4),
        "win_loss_count_ratio": round(wl, 6) if wl is not None else None,
        "total_pips": round(realized, 4),
        "cal_ppd": round(realized / max(cal_days, 1), 4),
        "pf_distance_to_1p27": round(abs((pf if pf is not None else 0.0) - TARGET_PF), 6),
    }


def main():
    m, day_i0 = load_data()
    rows = []
    for tp in TRAIL_PIPS_GRID:
        r = run_one(m, day_i0, tp)
        rows.append(r)
        print(
            f"trail={tp:>3}p  PF={r['profit_factor']}  cal_ppd={r['cal_ppd']:+.4f} "
            f"wins={r['wins']} losses={r['losses']}"
        )

    df = pd.DataFrame(rows)
    best_pf = df.sort_values("pf_distance_to_1p27", ascending=True).iloc[0].to_dict()
    best_ppd = df.sort_values("cal_ppd", ascending=False).iloc[0].to_dict()

    out = {
        "pair": "EURJPY",
        "rule": "Open BUY+SELL at each day open, trailing stop only, run up to 1 month (720h)",
        "target_profit_factor": TARGET_PF,
        "spread_pips": SPREAD_PIPS,
        "trail_pips_grid": TRAIL_PIPS_GRID,
        "closest_to_pf_1p27": best_pf,
        "best_cal_ppd": best_ppd,
        "all_results": df.sort_values("trail_pips").to_dict(orient="records"),
    }

    out_json = OUT_DIR / "dayopen_both_sides_trailing_1month_pf127_sweep_eurjpy.json"
    out_csv = OUT_DIR / "dayopen_both_sides_trailing_1month_pf127_sweep_eurjpy.csv"
    out_json.write_text(json.dumps(out, indent=2), encoding="utf-8")
    df.to_csv(out_csv, index=False)
    print("\nclosest_to_pf_1.27:", best_pf)
    print("best_cal_ppd:", best_ppd)
    print(f"saved={out_json}")
    print(f"saved={out_csv}")


if __name__ == "__main__":
    main()

