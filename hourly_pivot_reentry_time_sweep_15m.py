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


def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    d = pd.read_csv(DAILY_PATH, usecols=["Date", "High", "Low", "Close"], low_memory=False)
    d["Date"] = pd.to_datetime(d["Date"], errors="coerce")
    for c in ["High", "Low", "Close"]:
        d[c] = pd.to_numeric(d[c], errors="coerce")
    d = d.dropna().sort_values("Date").reset_index(drop=True)
    d["ATR14"] = atr_wilder(d["High"].to_numpy(float), d["Low"].to_numpy(float), d["Close"].to_numpy(float), ATR_PERIOD)
    d = d[(d["Date"] >= START_DATE) & (d["Date"] <= END_DATE)].dropna(subset=["ATR14"]).reset_index(drop=True)
    d["dkey"] = d["Date"].dt.date

    m = pd.read_csv(M15_PATH, usecols=["Datetime", "Open", "High", "Low", "Close"], low_memory=False)
    m["Datetime"] = pd.to_datetime(m["Datetime"], errors="coerce")
    for c in ["Open", "High", "Low", "Close"]:
        m[c] = pd.to_numeric(m[c], errors="coerce")
    m = m.dropna().sort_values("Datetime").reset_index(drop=True)
    m["dkey"] = m["Datetime"].dt.date
    m["hour"] = m["Datetime"].dt.hour
    m["minute"] = m["Datetime"].dt.minute
    m["hour_key"] = m["Datetime"].dt.floor("h")
    return d, m


def build_hourly_pivots(m15: pd.DataFrame) -> dict[pd.Timestamp, float]:
    h = (
        m15.groupby("hour_key")
        .agg(High=("High", "max"), Low=("Low", "min"), Close=("Close", "last"))
        .sort_index()
    )
    h["pivot"] = (h["High"] + h["Low"] + h["Close"]) / 3.0
    return h["pivot"].to_dict()


def close_leg(leg: Leg, px: float, pip: float) -> float:
    raw = (px - leg.entry) / pip if leg.side == "buy" else (leg.entry - px) / pip
    return raw - SPREAD_PIPS


def simulate_for_time(
    daily: pd.DataFrame,
    m15: pd.DataFrame,
    pivot_by_hour: dict[pd.Timestamp, float],
    entry_hour: int,
    entry_minute: int,
    pip: float,
) -> dict:
    entry_rows = m15[(m15["hour"] == entry_hour) & (m15["minute"] == entry_minute)][["dkey"]].copy()
    entry_rows["entry_i"] = entry_rows.index
    day_to_i = dict(zip(entry_rows["dkey"], entry_rows["entry_i"]))

    times = m15["Datetime"].to_numpy()
    highs = m15["High"].to_numpy(float)
    lows = m15["Low"].to_numpy(float)
    hour_keys = m15["hour_key"].to_numpy()

    total_pips = 0.0
    buy_pips = 0.0
    sell_pips = 0.0
    initial_entries = 0
    reentry_pairs = 0
    total_closed = 0

    for _, r in daily.iterrows():
        day = r["dkey"]
        i0 = day_to_i.get(day)
        if i0 is None:
            continue

        prev_hour_key = pd.Timestamp(hour_keys[i0]) - pd.Timedelta(hours=1)
        entry = pivot_by_hour.get(prev_hour_key)
        if entry is None or not np.isfinite(entry):
            continue

        dist_sl = float(r["ATR14"]) * SL_MULT
        dist_tp = float(r["ATR14"]) * TP_MULT
        if not np.isfinite(dist_sl) or not np.isfinite(dist_tp) or dist_sl <= 0 or dist_tp <= 0:
            continue

        expiry = times[i0] + pd.Timedelta(hours=MAX_HOLD_HOURS)
        legs = [
            Leg("buy", entry, entry - dist_sl, entry + dist_tp),
            Leg("sell", entry, entry + dist_sl, entry - dist_tp),
        ]
        initial_entries += 1

        i = i0 + 1
        while i < len(times) and times[i] <= expiry and any(x.active for x in legs):
            h = highs[i]
            l = lows[i]
            any_tp = False
            reentry_px = None

            for leg in legs:
                if not leg.active:
                    continue
                if leg.side == "buy":
                    sl_hit = l <= leg.sl
                    tp_hit = h >= leg.tp
                else:
                    sl_hit = h >= leg.sl
                    tp_hit = l <= leg.tp
                if not (sl_hit or tp_hit):
                    continue
                px = leg.tp if tp_hit else leg.sl
                p = close_leg(leg, px, pip)
                total_pips += p
                if leg.side == "buy":
                    buy_pips += p
                else:
                    sell_pips += p
                total_closed += 1
                leg.active = False
                if tp_hit:
                    any_tp = True
                    reentry_px = px

            if any_tp and reentry_px is not None:
                legs.append(Leg("buy", reentry_px, reentry_px - dist_sl, reentry_px + dist_tp))
                legs.append(Leg("sell", reentry_px, reentry_px + dist_sl, reentry_px - dist_tp))
                reentry_pairs += 1

            i += 1

        end_i = min(i - 1, len(times) - 1)
        if end_i >= i0:
            # Close leftovers at the last observed price in horizon.
            exit_px = float(m15.at[end_i, "Open"])
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
    return {
        "entry_time": f"{entry_hour:02d}:{entry_minute:02d}",
        "initial_entries": initial_entries,
        "reentry_pairs": reentry_pairs,
        "total_legs_closed": total_closed,
        "total_pips": round(total_pips, 4),
        "cal_ppd": round(total_pips / max(cal_days, 1), 4),
        "buy_cal_ppd": round(buy_pips / max(cal_days, 1), 4),
        "sell_cal_ppd": round(sell_pips / max(cal_days, 1), 4),
    }


def main() -> None:
    daily, m15 = load_data()
    pivot_by_hour = build_hourly_pivots(m15)
    pip = pip_size(PAIR)

    rows = []
    for h in range(24):
        rows.append(simulate_for_time(daily, m15, pivot_by_hour, h, 0, pip))

    df = pd.DataFrame(rows).sort_values("cal_ppd", ascending=False).reset_index(drop=True)
    out = {
        "pair": PAIR,
        "period": [START_DATE, END_DATE],
        "mode": "hourly-pivot-entry-time-sweep",
        "entry_pivot": "previous completed hour pivot (H+L+C)/3",
        "atr_source": "daily ATR14",
        "sl_mult": SL_MULT,
        "tp_mult": TP_MULT,
        "max_hold_hours": MAX_HOLD_HOURS,
        "tested_entry_times": [f"{h:02d}:00" for h in range(24)],
        "best": df.head(5).to_dict(orient="records"),
        "worst": df.tail(5).to_dict(orient="records"),
        "all_results": df.to_dict(orient="records"),
    }

    out_json = Path(__file__).parent / "hourly_pivot_reentry_time_sweep_15m_results.json"
    out_csv = Path(__file__).parent / "hourly_pivot_reentry_time_sweep_15m_results.csv"
    out_json.write_text(json.dumps(out, indent=2), encoding="utf-8")
    df.to_csv(out_csv, index=False)

    print("Top 5 entry times by cal_ppd:")
    print(df.head(5).to_string(index=False))
    print("\nBottom 5 entry times by cal_ppd:")
    print(df.tail(5).to_string(index=False))
    print(f"\nSaved -> {out_json}")
    print(f"Saved -> {out_csv}")


if __name__ == "__main__":
    main()

