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
SL_NORM = 0.55
MAX_BUCKET_NORM = 0.55
BUCKET_STEP = 0.05
MIN_DENOM_PIPS = 10.0


def load_data():
    d = pd.read_csv(DAILY_PATH, usecols=["Date", "Open", "High", "Low", "Close"], low_memory=False)
    d["Date"] = pd.to_datetime(d["Date"], errors="coerce")
    for c in ["Open", "High", "Low", "Close"]:
        d[c] = pd.to_numeric(d[c], errors="coerce")
    d = d.dropna().sort_values("Date").reset_index(drop=True)
    d["dkey"] = d["Date"].dt.normalize()

    # Previous-day pivot ladder for today's decision.
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

    # Keep only full 96-bar days.
    full_days = set(m.groupby("dkey").size().pipe(lambda s: s[s == 96]).index)
    m = m[m["dkey"].isin(full_days)].copy().reset_index(drop=True)
    day_first = m.groupby("dkey").head(1).reset_index()[["dkey", "index"]]
    day_last = m.groupby("dkey").tail(1).reset_index()[["dkey", "index"]]
    i0_map = {r["dkey"]: int(r["index"]) for _, r in day_first.iterrows()}
    i1_map = {r["dkey"]: int(r["index"]) for _, r in day_last.iterrows()}

    info = {}
    for _, r in d.iterrows():
        k = r["dkey"]
        if not (START_DATE <= k <= END_DATE):
            continue
        if k not in i0_map or k not in i1_map:
            continue
        info[k] = {
            "open": float(r["Open"]),
            "pp": float(r["pp"]),
            "r2": float(r["r2"]),
            "s2": float(r["s2"]),
            "i0": i0_map[k],
            "i1": i1_map[k],
        }
    return m, info


def pips(side: str, entry: float, px: float) -> float:
    if side == "BUY":
        return ((px - entry) / PIP) - SPREAD_PIPS
    return ((entry - px) / PIP) - SPREAD_PIPS


def run_diag(m: pd.DataFrame, info: dict):
    highs = m["High"].to_numpy(float)
    lows = m["Low"].to_numpy(float)
    closes = m["Close"].to_numpy(float)

    min_denom_px = MIN_DENOM_PIPS * PIP
    rows = []

    for d in sorted(info.keys()):
        v = info[d]
        entry = v["open"]  # open straight at day open
        pp = v["pp"]
        i0 = v["i0"]
        i1 = v["i1"]

        if entry < pp:
            side = "BUY"
            denom = pp - v["s2"]
            if denom < min_denom_px:
                continue
            entry_norm = (pp - entry) / denom
            sl = pp - SL_NORM * denom
        elif entry > pp:
            side = "SELL"
            denom = v["r2"] - pp
            if denom < min_denom_px:
                continue
            entry_norm = (entry - pp) / denom
            sl = pp + SL_NORM * denom
        else:
            continue

        if not (0.0 <= entry_norm <= MAX_BUCKET_NORM):
            continue

        # Track MFE and first passage to pivot/sl.
        best_fav = 0.0
        state = "none"
        exit_i = i1
        exit_px = closes[i1]

        for j in range(i0, len(highs)):
            hi = highs[j]
            lo = lows[j]

            if side == "BUY":
                fav = max(hi - entry, 0.0)
                if fav > best_fav:
                    best_fav = fav
                hit_sl = lo <= sl
                hit_pv = hi >= pp
            else:
                fav = max(entry - lo, 0.0)
                if fav > best_fav:
                    best_fav = fav
                hit_sl = hi >= sl
                hit_pv = lo <= pp

            if hit_sl and hit_pv:
                state = "sl_first_tie"
                exit_i = j
                exit_px = sl
                break
            if hit_sl:
                state = "sl_first"
                exit_i = j
                exit_px = sl
                break
            if hit_pv:
                state = "pivot_first"
                exit_i = j
                exit_px = pp
                break

        rows.append(
            {
                "date": d,
                "side": side,
                "entry_norm": float(entry_norm),
                "mfe_norm": float(best_fav / denom),
                "mfe_pips": float(best_fav / PIP),
                "pivot_hit_before_sl": int(state == "pivot_first"),
                "sl_hit_before_pivot": int(state in ("sl_first", "sl_first_tie")),
                "state": state,
                "pips_at_first_event": pips(side, entry, exit_px),
                "bars_to_first_event": int(exit_i - i0),
            }
        )

    df = pd.DataFrame(rows)
    if df.empty:
        return df, pd.DataFrame()

    bins = np.arange(0.0, MAX_BUCKET_NORM + BUCKET_STEP + 1e-9, BUCKET_STEP)
    labels = [f"{int(a*100):02d}-{int(b*100):02d}%" for a, b in zip(bins[:-1], bins[1:])]
    df["bucket"] = pd.cut(df["entry_norm"], bins=bins, right=False, labels=labels, include_lowest=True)

    grouped = (
        df.groupby(["side", "bucket"], observed=False)
        .agg(
            trades=("entry_norm", "size"),
            avg_entry_norm=("entry_norm", "mean"),
            avg_mfe_norm=("mfe_norm", "mean"),
            med_mfe_norm=("mfe_norm", "median"),
            avg_mfe_pips=("mfe_pips", "mean"),
            med_mfe_pips=("mfe_pips", "median"),
            pivot_first_rate=("pivot_hit_before_sl", "mean"),
            sl_first_rate=("sl_hit_before_pivot", "mean"),
            avg_pips_first_event=("pips_at_first_event", "mean"),
        )
        .reset_index()
    )
    grouped = grouped[grouped["trades"] > 0].copy()
    grouped["pivot_first_rate"] = grouped["pivot_first_rate"].round(6)
    grouped["sl_first_rate"] = grouped["sl_first_rate"].round(6)
    return df, grouped


def main():
    m, info = load_data()
    trades, bucket_tbl = run_diag(m, info)

    out = {
        "pair": "EURJPY",
        "period": [str(START_DATE.date()), str(END_DATE.date())],
        "entry_rule": "open straight at day open; open<PP BUY, open>PP SELL",
        "sl_rule": "hard stop at 55% away from pivot (PP->S2 for buys, PP->R2 for sells)",
        "bucket_rule": "entry distance 0..55% in 5% buckets",
        "trades": int(len(trades)),
        "buy_trades": int((trades["side"] == "BUY").sum()) if not trades.empty else 0,
        "sell_trades": int((trades["side"] == "SELL").sum()) if not trades.empty else 0,
        "bucket_results": bucket_tbl.to_dict(orient="records"),
    }

    out_json = OUT_DIR / "dayopen_pivotside_sl55_bucket5pct_diag_2003_2015_eurjpy.json"
    out_trades = OUT_DIR / "dayopen_pivotside_sl55_bucket5pct_diag_2003_2015_eurjpy_trades.csv"
    out_buckets = OUT_DIR / "dayopen_pivotside_sl55_bucket5pct_diag_2003_2015_eurjpy_buckets.csv"
    out_json.write_text(json.dumps(out, indent=2), encoding="utf-8")
    trades.to_csv(out_trades, index=False)
    bucket_tbl.to_csv(out_buckets, index=False)

    print("SUMMARY")
    print(json.dumps({k: v for k, v in out.items() if k != "bucket_results"}, indent=2))
    print("\nTOP BUCKETS BY AVG_MFE_PIPS")
    top = bucket_tbl.sort_values("avg_mfe_pips", ascending=False).head(8)
    print(top.to_string(index=False))
    print(f"saved={out_json}")
    print(f"saved={out_trades}")
    print(f"saved={out_buckets}")


if __name__ == "__main__":
    main()

