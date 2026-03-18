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
ENTRY_NORM = 0.225  # near the earlier 50/50 point (0.20..0.25 bin center)
MIN_SIDE_RANGE_PIPS = 10.0
PIP = 0.01


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

    m = pd.read_csv(M15_PATH, usecols=["Datetime", "Open", "High", "Low", "Close"], low_memory=False)
    m["Datetime"] = pd.to_datetime(m["Datetime"], errors="coerce")
    for c in ["Open", "High", "Low", "Close"]:
        m[c] = pd.to_numeric(m[c], errors="coerce")
    m = m.dropna().sort_values("Datetime").reset_index(drop=True)
    m["dkey"] = m["Datetime"].dt.normalize()

    full = m.groupby("dkey").size()
    full_days = set(full[full == 96].index)
    m = m[m["dkey"].isin(full_days)].copy()

    bars = {}
    for dkey, g in m.groupby("dkey", sort=True):
        bars[dkey] = {
            "o": g["Open"].to_numpy(float),
            "h": g["High"].to_numpy(float),
            "l": g["Low"].to_numpy(float),
            "c": g["Close"].to_numpy(float),
        }

    info = {}
    for _, r in d.iterrows():
        k = r["dkey"]
        if not (START_DATE <= k <= END_DATE):
            continue
        if k not in bars:
            continue
        info[k] = {"pp": float(r["pp"]), "r2": float(r["r2"]), "s2": float(r["s2"])}

    days = sorted(info.keys())
    return days, info, bars


def summarize_series(x: pd.Series) -> dict:
    if x.empty:
        return {}
    q = x.quantile([0.1, 0.25, 0.5, 0.75, 0.9]).to_dict()
    return {
        "mean": round(float(x.mean()), 4),
        "q10": round(float(q.get(0.1, np.nan)), 4),
        "q25": round(float(q.get(0.25, np.nan)), 4),
        "q50": round(float(q.get(0.5, np.nan)), 4),
        "q75": round(float(q.get(0.75, np.nan)), 4),
        "q90": round(float(q.get(0.9, np.nan)), 4),
    }


def run(days, info, bars) -> pd.DataFrame:
    out = []
    min_range_px = MIN_SIDE_RANGE_PIPS * PIP

    for d in days:
        lv = info[d]
        b = bars[d]
        pp = lv["pp"]
        up_range = lv["r2"] - pp
        dn_range = pp - lv["s2"]
        if up_range < min_range_px or dn_range < min_range_px:
            continue

        lower_touch = pp - ENTRY_NORM * dn_range
        upper_touch = pp + ENTRY_NORM * up_range

        trigger_idx = None
        side = None
        entry_px = None
        ambiguous = False

        for i in range(96):
            hit_lower = b["l"][i] <= lower_touch <= b["h"][i]
            hit_upper = b["l"][i] <= upper_touch <= b["h"][i]
            if hit_lower and hit_upper:
                ambiguous = True
                break
            if hit_lower:
                trigger_idx = i
                side = "BUY"
                entry_px = lower_touch
                break
            if hit_upper:
                trigger_idx = i
                side = "SELL"
                entry_px = upper_touch
                break

        if ambiguous or trigger_idx is None:
            continue

        # from trigger onward to day end: how far it reverts toward pivot (pivot == 0 scale)
        if side == "BUY":
            denom = dn_range
            # coordinate: (price - pp)/denom ; pivot is 0, below pivot is negative
            best_toward_pivot = np.max((b["h"][trigger_idx:] - pp) / denom)
            close_coord = (b["c"][-1] - pp) / denom
            entry_coord = -ENTRY_NORM
        else:
            denom = up_range
            # mirrored coordinate so pivot stays 0 and touch starts at -ENTRY_NORM
            best_toward_pivot = np.max((pp - b["l"][trigger_idx:]) / denom)
            close_coord = (pp - b["c"][-1]) / denom
            entry_coord = -ENTRY_NORM

        out.append(
            {
                "date": d,
                "side": side,
                "entry_coord": entry_coord,
                "best_toward_pivot_coord": float(best_toward_pivot),
                "close_coord": float(close_coord),
                "reached_pivot": int(best_toward_pivot >= 0.0),
            }
        )

    return pd.DataFrame(out)


def main():
    days, info, bars = load_data()
    df = run(days, info, bars)

    all_best = df["best_toward_pivot_coord"] if not df.empty else pd.Series(dtype=float)
    buy_best = df[df["side"] == "BUY"]["best_toward_pivot_coord"] if not df.empty else pd.Series(dtype=float)
    sell_best = df[df["side"] == "SELL"]["best_toward_pivot_coord"] if not df.empty else pd.Series(dtype=float)

    summary = {
        "pair": "EURJPY",
        "period": [str(START_DATE.date()), str(END_DATE.date())],
        "entry_norm_touch": ENTRY_NORM,
        "scale_definition": "pivot=0; touch entry starts at -entry_norm; values <0 mean pivot not reached",
        "triggers": int(len(df)),
        "buy_triggers": int((df["side"] == "BUY").sum()) if not df.empty else 0,
        "sell_triggers": int((df["side"] == "SELL").sum()) if not df.empty else 0,
        "pivot_reach_rate_all": round(float(df["reached_pivot"].mean()), 4) if not df.empty else None,
        "pivot_reach_rate_buy": round(float((df[df["side"] == "BUY"]["reached_pivot"]).mean()), 4) if not buy_best.empty else None,
        "pivot_reach_rate_sell": round(float((df[df["side"] == "SELL"]["reached_pivot"]).mean()), 4) if not sell_best.empty else None,
        "best_toward_pivot_coord_all": summarize_series(all_best),
        "best_toward_pivot_coord_buy": summarize_series(buy_best),
        "best_toward_pivot_coord_sell": summarize_series(sell_best),
    }

    out_json = OUT_DIR / "pivot_norm_touch_reversion_diag_2003_2015_eurjpy.json"
    out_csv = OUT_DIR / "pivot_norm_touch_reversion_diag_2003_2015_eurjpy_triggers.csv"
    out_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    df.to_csv(out_csv, index=False)

    print("SUMMARY")
    print(json.dumps(summary, indent=2))
    print(f"saved={out_json}")
    print(f"saved={out_csv}")


if __name__ == "__main__":
    main()

