from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(r"C:\Users\temen\5 joulu\TrailingStopLoss pivot peak trading 6 march 2026")
PAIR_DEFAULT = "usdjpy"
START_DEFAULT = "2014-01-01"
END_DEFAULT = "2021-12-31"


def pip_size(pair: str) -> float:
    return 0.01 if pair.endswith("jpy") else 0.0001


def m15_path(pair: str) -> Path:
    p = pair.lower()
    return ROOT / "NEW_AGENT_HOURLY_33PIPS_PACK" / "pair_datasets" / p / f"processed_{p}_data_15m.csv"


def load_m15(path: Path) -> pd.DataFrame:
    m = pd.read_csv(path, usecols=["Datetime", "Open", "Close"], low_memory=False)
    m["Datetime"] = pd.to_datetime(m["Datetime"], errors="coerce")
    for c in ["Open", "Close"]:
        m[c] = pd.to_numeric(m[c], errors="coerce")
    m = m.dropna(subset=["Datetime", "Open", "Close"]).sort_values("Datetime").reset_index(drop=True)
    m["dkey"] = m["Datetime"].dt.normalize()
    return m


def summarize_group(records: list[dict]) -> dict:
    if not records:
        return {
            "days": 0,
            "sum_up_pips": 0.0,
            "sum_down_pips": 0.0,
            "avg_up_pips_per_day": 0.0,
            "avg_down_pips_per_day": 0.0,
            "avg_net_pips_per_day": 0.0,
            "up_down_ratio": None,
        }
    df = pd.DataFrame(records)
    days = len(df)
    sum_up = float(df["up_pips"].sum())
    sum_down = float(df["down_pips"].sum())
    avg_up = sum_up / days
    avg_down = sum_down / days
    avg_net = float((df["up_pips"] - df["down_pips"]).mean())
    ratio = (sum_up / sum_down) if sum_down > 0 else None
    return {
        "days": days,
        "sum_up_pips": round(sum_up, 4),
        "sum_down_pips": round(sum_down, 4),
        "avg_up_pips_per_day": round(avg_up, 4),
        "avg_down_pips_per_day": round(avg_down, 4),
        "avg_net_pips_per_day": round(avg_net, 4),
        "up_down_ratio": round(ratio, 6) if ratio is not None else None,
    }


def run(pair: str, start_date: str, end_date: str) -> dict:
    m = load_m15(m15_path(pair))
    start = pd.Timestamp(start_date)
    end_next = pd.Timestamp(end_date) + pd.Timedelta(days=1)
    m = m[(m["Datetime"] >= start) & (m["Datetime"] < end_next)].copy()
    pip = pip_size(pair)

    up_days = []
    down_days = []
    flat_days = []

    for dkey, g in m.groupby("dkey", sort=True):
        if g.empty:
            continue
        day_open = float(g["Open"].iloc[0])
        day_close = float(g["Close"].iloc[-1])

        # Build a day path from open -> closes and decompose every move.
        path = np.concatenate(([day_open], g["Close"].to_numpy(float)))
        diffs = np.diff(path)
        up_pips = float(np.clip(diffs, 0.0, None).sum() / pip)
        down_pips = float(np.clip(-diffs, 0.0, None).sum() / pip)
        net_pips = (day_close - day_open) / pip

        rec = {
            "date": str(pd.Timestamp(dkey).date()),
            "open": day_open,
            "close": day_close,
            "net_pips": round(float(net_pips), 4),
            "up_pips": round(up_pips, 4),
            "down_pips": round(down_pips, 4),
        }
        if day_close > day_open:
            up_days.append(rec)
        elif day_close < day_open:
            down_days.append(rec)
        else:
            flat_days.append(rec)

    out = {
        "pair": pair.upper(),
        "period": [start_date, end_date],
        "timeframe": "15m",
        "definition": "For each day, sum all positive and all negative close-to-close pip moves from day open. Group by day close vs day open direction.",
        "up_days_summary": summarize_group(up_days),
        "down_days_summary": summarize_group(down_days),
        "flat_days_summary": summarize_group(flat_days),
        "examples": {
            "up_days_first5": up_days[:5],
            "down_days_first5": down_days[:5],
        },
    }

    out_path = Path(__file__).parent / f"day_direction_updown_pip_flow_{pair.lower()}_{start_date}_{end_date}.json"
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out["up_days_summary"], indent=2))
    print(json.dumps(out["down_days_summary"], indent=2))
    print(f"saved={out_path}")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pair", default=PAIR_DEFAULT)
    ap.add_argument("--start-date", default=START_DEFAULT)
    ap.add_argument("--end-date", default=END_DEFAULT)
    args = ap.parse_args()
    run(args.pair.lower(), args.start_date, args.end_date)


if __name__ == "__main__":
    main()

