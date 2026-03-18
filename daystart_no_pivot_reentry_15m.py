from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

PAIR = "usdjpy"
START_DATE = "2014-01-01"
END_DATE = "2021-12-31"
ATR_PERIOD = 14
MAX_HOLD_HOURS = 432
SL_MULT = 0.08
TP_MULT = 0.55
SPREAD_PIPS = 2.0

ROOT = Path(r"C:\Users\temen\5 joulu\TrailingStopLoss pivot peak trading 6 march 2026")
DAILY_PATH = ROOT / "NEW_AGENT_HOURLY_33PIPS_PACK" / "pair_datasets" / "usdjpy" / "processed_usdjpy_data_daily.csv"
M15_PATH = ROOT / "NEW_AGENT_HOURLY_33PIPS_PACK" / "pair_datasets" / "usdjpy" / "processed_usdjpy_data_15m.csv"


@dataclass
class Leg:
    side: str
    entry: float
    sl: float
    tp: float
    open_i: int
    active: bool = True


def pip_size(pair: str) -> float:
    return 0.01 if pair.endswith("jpy") else 0.0001


def atr_wilder(h: np.ndarray, l: np.ndarray, c: np.ndarray, n: int) -> np.ndarray:
    tr = np.empty(len(h), float)
    tr[0] = h[0] - l[0]
    for i in range(1, len(h)):
        tr[i] = max(h[i] - l[i], abs(h[i] - c[i - 1]), abs(l[i] - c[i - 1]))
    out = np.full(len(h), np.nan)
    if len(h) >= n:
        out[n - 1] = tr[:n].mean()
        k = 1.0 / n
        for i in range(n, len(h)):
            out[i] = out[i - 1] * (1 - k) + tr[i] * k
    return out


def load_data():
    d = pd.read_csv(DAILY_PATH, usecols=["Date", "High", "Low", "Close"], low_memory=False)
    d["Date"] = pd.to_datetime(d["Date"], errors="coerce")
    for c in ["High", "Low", "Close"]:
        d[c] = pd.to_numeric(d[c], errors="coerce")
    d = d.dropna().sort_values("Date").reset_index(drop=True)
    d["ATR14"] = atr_wilder(d["High"].to_numpy(float), d["Low"].to_numpy(float), d["Close"].to_numpy(float), ATR_PERIOD)

    m = pd.read_csv(M15_PATH, usecols=["Datetime", "Open", "High", "Low"], low_memory=False)
    m["Datetime"] = pd.to_datetime(m["Datetime"], errors="coerce")
    for c in ["Open", "High", "Low"]:
        m[c] = pd.to_numeric(m[c], errors="coerce")
    m = m.dropna().sort_values("Datetime").reset_index(drop=True)
    return d, m


def close_leg(leg: Leg, px: float, pip: float) -> float:
    raw = (px - leg.entry) / pip if leg.side == "buy" else (leg.entry - px) / pip
    return raw - SPREAD_PIPS


def run():
    daily, m15 = load_data()
    pip = pip_size(PAIR)
    valid_days = daily[(daily["Date"] >= START_DATE) & (daily["Date"] <= END_DATE)].copy()
    valid_days = valid_days.dropna(subset=["ATR14"]).reset_index(drop=True)

    m15["date"] = m15["Datetime"].dt.date
    first_idx = m15.groupby("date").head(1).reset_index()[["date", "index"]]
    first_map = dict(zip(first_idx["date"], first_idx["index"]))

    times = m15["Datetime"].to_numpy()
    opens = m15["Open"].to_numpy(float)
    highs = m15["High"].to_numpy(float)
    lows = m15["Low"].to_numpy(float)

    total_pips = 0.0
    buy_pips = 0.0
    sell_pips = 0.0
    initial_entries = 0
    reentry_pairs = 0
    total_closed = 0

    for _, r in valid_days.iterrows():
        day = r["Date"].date()
        if day not in first_map:
            continue
        i0 = int(first_map[day])
        entry = opens[i0]
        dist_sl = float(r["ATR14"]) * SL_MULT
        dist_tp = float(r["ATR14"]) * TP_MULT
        if not np.isfinite(dist_sl) or not np.isfinite(dist_tp) or dist_sl <= 0 or dist_tp <= 0:
            continue

        expiry = times[i0] + pd.Timedelta(hours=MAX_HOLD_HOURS)
        legs = [
            Leg("buy", entry, entry - dist_sl, entry + dist_tp, i0),
            Leg("sell", entry, entry + dist_sl, entry - dist_tp, i0),
        ]
        initial_entries += 1

        i = i0 + 1
        while i < len(times) and times[i] <= expiry and any(x.active for x in legs):
            h = highs[i]
            l = lows[i]
            tp_hit = False
            tp_price = None

            for leg in legs:
                if not leg.active:
                    continue
                if leg.side == "buy":
                    sl_hit = l <= leg.sl
                    t_hit = h >= leg.tp
                else:
                    sl_hit = h >= leg.sl
                    t_hit = l <= leg.tp
                if not (sl_hit or t_hit):
                    continue
                px = leg.tp if t_hit else leg.sl
                p = close_leg(leg, px, pip)
                total_pips += p
                if leg.side == "buy":
                    buy_pips += p
                else:
                    sell_pips += p
                total_closed += 1
                leg.active = False
                if t_hit:
                    tp_hit = True
                    tp_price = px

            if tp_hit and tp_price is not None:
                legs.append(Leg("buy", tp_price, tp_price - dist_sl, tp_price + dist_tp, i))
                legs.append(Leg("sell", tp_price, tp_price + dist_sl, tp_price - dist_tp, i))
                reentry_pairs += 1

            i += 1

        end_i = min(i - 1, len(times) - 1)
        if end_i >= i0:
            exit_px = opens[end_i]
            for leg in legs:
                if not leg.active:
                    continue
                p = close_leg(leg, exit_px, pip)
                total_pips += p
                if leg.side == "buy":
                    buy_pips += p
                else:
                    sell_pips += p
                total_closed += 1
                leg.active = False

    cal_days = int((pd.Timestamp(END_DATE) - pd.Timestamp(START_DATE)).days + 1)
    out = {
        "pair": PAIR,
        "period": [START_DATE, END_DATE],
        "mode": "day-start-no-pivot-reentry",
        "sl_mult": SL_MULT,
        "tp_mult": TP_MULT,
        "max_hold_hours": MAX_HOLD_HOURS,
        "initial_entries": initial_entries,
        "reentry_pairs": reentry_pairs,
        "total_legs_closed": total_closed,
        "total_pips": round(total_pips, 4),
        "cal_ppd": round(total_pips / max(cal_days, 1), 4),
        "buy_cal_ppd": round(buy_pips / max(cal_days, 1), 4),
        "sell_cal_ppd": round(sell_pips / max(cal_days, 1), 4),
    }
    out_path = Path(__file__).parent / "daystart_no_pivot_reentry_15m_results.json"
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=2))
    print(f"Saved -> {out_path}")


if __name__ == "__main__":
    run()

