from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


PAIR = "usdjpy"
START_DATE = "2014-01-01"
END_DATE = "2021-12-31"
ATR_PERIOD = 14
MAX_HOLD_HOURS = 432
WINDOW_STARTS = [0, 8, 16]

# Small neighborhood around found regime (do not change profile)
SL_GRID = [0.08, 0.10, 0.12]
TP_GRID = [0.45, 0.50, 0.55]

DATA_ROOT = Path(r"C:\Users\temen\5 joulu\TrailingStopLoss pivot peak trading 6 march 2026")
DAILY_PATH = DATA_ROOT / "NEW_AGENT_HOURLY_33PIPS_PACK" / "pair_datasets" / "usdjpy" / "processed_usdjpy_data_daily.csv"
M15_PATH = DATA_ROOT / "NEW_AGENT_HOURLY_33PIPS_PACK" / "pair_datasets" / "usdjpy" / "processed_usdjpy_data_15m.csv"
OUTPUT_DIR = Path(__file__).parent


def load_daily(path: Path) -> pd.DataFrame:
    d = pd.read_csv(path, low_memory=False)
    d["Date"] = pd.to_datetime(d["Date"], errors="coerce")
    for c in ["Open", "High", "Low", "Close"]:
        d[c] = pd.to_numeric(d[c], errors="coerce")
    return d.dropna(subset=["Date", "Open", "High", "Low", "Close"]).sort_values("Date").reset_index(drop=True)


def load_m15(path: Path):
    m = pd.read_csv(path, usecols=["Datetime", "Open", "High", "Low", "Close"], low_memory=False)
    m["Datetime"] = pd.to_datetime(m["Datetime"], errors="coerce")
    for c in ["Open", "High", "Low", "Close"]:
        m[c] = pd.to_numeric(m[c], errors="coerce")
    m = m.dropna(subset=["Datetime", "Open", "High", "Low", "Close"]).sort_values("Datetime").reset_index(drop=True)
    return (
        m["Datetime"].to_numpy(),
        m["Open"].to_numpy(float),
        m["High"].to_numpy(float),
        m["Low"].to_numpy(float),
        m["Close"].to_numpy(float),
    )


def atr_wilder(h: np.ndarray, l: np.ndarray, c: np.ndarray, period: int = 14) -> np.ndarray:
    n = len(h)
    tr = np.empty(n, dtype=float)
    tr[0] = h[0] - l[0]
    for i in range(1, n):
        tr[i] = max(h[i] - l[i], abs(h[i] - c[i - 1]), abs(l[i] - c[i - 1]))
    atr = np.full(n, np.nan)
    if n >= period:
        atr[period - 1] = np.mean(tr[:period])
        k = 1.0 / period
        for i in range(period, n):
            atr[i] = atr[i - 1] * (1.0 - k) + tr[i] * k
    return atr


def pip_size(pair: str) -> float:
    return 0.01 if pair.endswith("jpy") else 0.0001


def run_one(daily: pd.DataFrame, times, highs, lows, closes, atr: np.ndarray, sl_mult: float, tp_mult: float, pip: float, cost_pips: float) -> dict:
    start = pd.Timestamp(START_DATE)
    end = pd.Timestamp(END_DATE)
    cal_days = (end - start).days + 1

    total = 0.0
    buy_total = 0.0
    sell_total = 0.0
    initial_entries = 0
    reentry_pairs = 0
    legs_closed = 0

    for i in range(1, len(daily)):
        d = pd.Timestamp(daily.loc[i, "Date"])
        if d < start or d > end:
            continue

        yh = float(daily.loc[i - 1, "High"])
        yl = float(daily.loc[i - 1, "Low"])
        yc = float(daily.loc[i - 1, "Close"])
        atr_prev = float(atr[i - 1]) if np.isfinite(atr[i - 1]) else np.nan
        if not np.isfinite(atr_prev) or atr_prev <= 0:
            continue
        pp = (yh + yl + yc) / 3.0
        sl_dist = sl_mult * atr_prev
        tp_dist = tp_mult * atr_prev

        for wh in WINDOW_STARTS:
            wstart = np.datetime64(d + pd.Timedelta(hours=wh))
            i0 = int(np.searchsorted(times, wstart, side="left"))
            if i0 >= len(times):
                continue
            if pd.Timestamp(times[i0]).date() != d.date():
                continue

            end_bar = min(i0 + MAX_HOLD_HOURS * 4, len(highs))
            if end_bar <= i0:
                continue

            # each pos: [side, entry, tp, sl, best_extreme, active_from, open]
            pos = [
                ["BUY", pp, pp + tp_dist, pp - sl_dist, pp, i0, True],
                ["SELL", pp, pp - tp_dist, pp + sl_dist, pp, i0, True],
            ]
            initial_entries += 1

            for j in range(i0, end_bar):
                h_ = float(highs[j]); l_ = float(lows[j])
                tp_hits = []

                for p in pos:
                    if (not p[6]) or j < p[5]:
                        continue
                    side, entry, tp, sl, best, _, _ = p
                    if side == "BUY":
                        best = max(best, h_)
                        sl = max(sl, best - sl_dist)
                        hit_sl = l_ <= sl
                        hit_tp = h_ >= tp
                    else:
                        best = min(best, l_)
                        sl = min(sl, best + sl_dist)
                        hit_sl = h_ >= sl
                        hit_tp = l_ <= tp
                    p[3] = sl
                    p[4] = best

                    if hit_sl and hit_tp:
                        px = sl
                        pips = ((px - entry) / pip) - cost_pips if side == "BUY" else ((entry - px) / pip) - cost_pips
                        total += pips
                        if side == "BUY":
                            buy_total += pips
                        else:
                            sell_total += pips
                        p[6] = False
                        legs_closed += 1
                    elif hit_sl:
                        px = sl
                        pips = ((px - entry) / pip) - cost_pips if side == "BUY" else ((entry - px) / pip) - cost_pips
                        total += pips
                        if side == "BUY":
                            buy_total += pips
                        else:
                            sell_total += pips
                        p[6] = False
                        legs_closed += 1
                    elif hit_tp:
                        px = tp
                        pips = ((px - entry) / pip) - cost_pips if side == "BUY" else ((entry - px) / pip) - cost_pips
                        total += pips
                        if side == "BUY":
                            buy_total += pips
                        else:
                            sell_total += pips
                        p[6] = False
                        legs_closed += 1
                        tp_hits.append(px)

                for tp_px in tp_hits:
                    # next-bar activation to avoid same-bar recursion
                    pos.append(["BUY", tp_px, tp_px + tp_dist, tp_px - sl_dist, tp_px, j + 1, True])
                    pos.append(["SELL", tp_px, tp_px - tp_dist, tp_px + sl_dist, tp_px, j + 1, True])
                    reentry_pairs += 1

                if all(not p[6] for p in pos):
                    break

            last_c = float(closes[end_bar - 1])
            for p in pos:
                if p[6]:
                    side, entry = p[0], p[1]
                    pips = ((last_c - entry) / pip) - cost_pips if side == "BUY" else ((entry - last_c) / pip) - cost_pips
                    total += pips
                    if side == "BUY":
                        buy_total += pips
                    else:
                        sell_total += pips
                    legs_closed += 1

    return {
        "sl_mult": sl_mult,
        "tp_mult": tp_mult,
        "cal_ppd": round(total / cal_days, 4),
        "buy_cal_ppd": round(buy_total / cal_days, 4),
        "sell_cal_ppd": round(sell_total / cal_days, 4),
        "total_pips": round(total, 4),
        "initial_entries": initial_entries,
        "reentry_pairs": reentry_pairs,
        "total_legs_closed": legs_closed,
    }


def main() -> None:
    daily = load_daily(DAILY_PATH)
    times, opens, highs, lows, closes = load_m15(M15_PATH)
    atr = atr_wilder(
        daily["High"].to_numpy(float),
        daily["Low"].to_numpy(float),
        daily["Close"].to_numpy(float),
        ATR_PERIOD,
    )
    pip = pip_size(PAIR)
    cost_pips = 2.0

    rows = []
    for slm in SL_GRID:
        for tpm in TP_GRID:
            r = run_one(daily, times, highs, lows, closes, atr, slm, tpm, pip, cost_pips)
            rows.append(r)
            print(f"sl={slm:.2f} tp={tpm:.2f} -> cal_ppd={r['cal_ppd']:+.4f}")

    df = pd.DataFrame(rows).sort_values("cal_ppd", ascending=False).reset_index(drop=True)
    best = df.iloc[0].to_dict()
    out = {
        "pair": PAIR.upper(),
        "period": [START_DATE, END_DATE],
        "mode": "8h_open_tp_reentry_chain",
        "sl_grid": SL_GRID,
        "tp_grid": TP_GRID,
        "best": best,
        "top10": df.head(10).to_dict(orient="records"),
    }
    out_path = OUTPUT_DIR / f"pivot_8h_tp_reentry_finetune_{PAIR}_{START_DATE}_{END_DATE}.json"
    out_path.write_text(json.dumps(out, indent=2))
    print("\nBEST FINETUNE:")
    print(best)
    print(f"saved={out_path}")


if __name__ == "__main__":
    main()
