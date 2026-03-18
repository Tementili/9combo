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
STEP_PIPS_DEFAULT = 1.0
MAX_LEVEL_DEFAULT = 20


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


def summarize(values: list[int]) -> dict:
    if not values:
        return {"count": 0, "avg": np.nan, "median": np.nan, "p90": np.nan}
    arr = np.asarray(values, dtype=float)
    return {
        "count": int(arr.size),
        "avg": round(float(arr.mean()), 4),
        "median": round(float(np.median(arr)), 4),
        "p90": round(float(np.quantile(arr, 0.9)), 4),
    }


def run(pair: str, start_date: str, end_date: str, step_pips: float, max_level: int) -> dict:
    m = load_m15(m15_path(pair))
    start = pd.Timestamp(start_date)
    end_next = pd.Timestamp(end_date) + pd.Timedelta(days=1)
    m = m[(m["Datetime"] >= start) & (m["Datetime"] < end_next)].copy()

    pip = pip_size(pair)
    step_px = step_pips * pip

    up_stats = {lvl: [] for lvl in range(0, max_level - 1)}
    dn_stats = {lvl: [] for lvl in range(0, max_level - 1)}
    traded_days = 0

    for _, g in m.groupby("dkey", sort=True):
        if g.empty:
            continue
        traded_days += 1

        day_open = float(g["Open"].iloc[0])
        # Step level from open using closes to avoid intrabar sequencing ambiguity.
        levels = np.floor((g["Close"].to_numpy(float) - day_open) / step_px).astype(int)
        n = len(levels)
        if n < 3:
            continue

        # Upward: from +L to +(L+2), how many steps back occurred in-between.
        for L in range(0, max_level - 1):
            hit_L = np.where(levels >= L)[0]
            if hit_L.size == 0:
                continue
            i0 = int(hit_L[0])
            hit_L2 = np.where(levels[i0:] >= (L + 2))[0]
            if hit_L2.size == 0:
                continue
            i1 = i0 + int(hit_L2[0])
            min_mid = int(levels[i0 : i1 + 1].min())
            retrace = max(0, L - min_mid)
            up_stats[L].append(retrace)

        # Downward: from -L to -(L+2), how many steps up bounce in-between.
        for L in range(0, max_level - 1):
            hit_L = np.where(levels <= -L)[0]
            if hit_L.size == 0:
                continue
            i0 = int(hit_L[0])
            hit_L2 = np.where(levels[i0:] <= -(L + 2))[0]
            if hit_L2.size == 0:
                continue
            i1 = i0 + int(hit_L2[0])
            max_mid = int(levels[i0 : i1 + 1].max())
            bounce = max(0, max_mid + L)
            dn_stats[L].append(bounce)

    up_rows = []
    dn_rows = []
    for L in range(0, max_level - 1):
        u = summarize(up_stats[L])
        d = summarize(dn_stats[L])
        up_rows.append(
            {
                "from_level_steps": L,
                "to_level_steps": L + 2,
                "samples": u["count"],
                "avg_steps_back_to_get_2_up": u["avg"],
                "median": u["median"],
                "p90": u["p90"],
            }
        )
        dn_rows.append(
            {
                "from_level_steps": -L,
                "to_level_steps": -(L + 2),
                "samples": d["count"],
                "avg_steps_back_to_get_2_down": d["avg"],
                "median": d["median"],
                "p90": d["p90"],
            }
        )

    up_df = pd.DataFrame(up_rows)
    dn_df = pd.DataFrame(dn_rows)

    # Weighted average by sample count, interpreted as "steps back per 2-step advance".
    def weighted_avg(df: pd.DataFrame, col: str) -> float:
        sub = df[(df["samples"] > 0) & df[col].notna()]
        if sub.empty:
            return float("nan")
        w = sub["samples"].to_numpy(float)
        x = sub[col].to_numpy(float)
        return float(np.average(x, weights=w))

    up_wavg = weighted_avg(up_df, "avg_steps_back_to_get_2_up")
    dn_wavg = weighted_avg(dn_df, "avg_steps_back_to_get_2_down")

    out = {
        "pair": pair.upper(),
        "period": [start_date, end_date],
        "timeframe": "15m_close_steps",
        "definition": "From day open, count how many opposite steps occur before gaining 2 more steps in same direction.",
        "step_pips": step_pips,
        "max_start_level_steps": max_level - 1,
        "days_in_sample": traded_days,
        "summary": {
            "up_weighted_avg_steps_back_for_2_steps_up": round(up_wavg, 4) if np.isfinite(up_wavg) else None,
            "down_weighted_avg_steps_back_for_2_steps_down": round(dn_wavg, 4) if np.isfinite(dn_wavg) else None,
        },
        "up_table": up_rows,
        "down_table": dn_rows,
    }

    out_json = Path(__file__).parent / f"daily_open_step_retrace_diag_{pair.lower()}_{start_date}_{end_date}_{step_pips:g}pip.json"
    up_csv = Path(__file__).parent / f"daily_open_step_retrace_up_{pair.lower()}_{start_date}_{end_date}_{step_pips:g}pip.csv"
    dn_csv = Path(__file__).parent / f"daily_open_step_retrace_down_{pair.lower()}_{start_date}_{end_date}_{step_pips:g}pip.csv"
    out_json.write_text(json.dumps(out, indent=2), encoding="utf-8")
    up_df.to_csv(up_csv, index=False)
    dn_df.to_csv(dn_csv, index=False)

    print(json.dumps(out["summary"], indent=2))
    print(f"saved={out_json}")
    print(f"saved={up_csv}")
    print(f"saved={dn_csv}")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pair", default=PAIR_DEFAULT)
    ap.add_argument("--start-date", default=START_DEFAULT)
    ap.add_argument("--end-date", default=END_DEFAULT)
    ap.add_argument("--step-pips", type=float, default=STEP_PIPS_DEFAULT)
    ap.add_argument("--max-level", type=int, default=MAX_LEVEL_DEFAULT)
    args = ap.parse_args()
    run(args.pair.lower(), args.start_date, args.end_date, args.step_pips, args.max_level)


if __name__ == "__main__":
    main()

