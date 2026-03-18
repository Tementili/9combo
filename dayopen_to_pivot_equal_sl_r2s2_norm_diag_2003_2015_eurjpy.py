from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(r"C:\Users\temen\5 joulu\TrailingStopLoss pivot peak trading 6 march 2026")
M15_PATH = ROOT / "forex raw data" / "Dukascopy" / "weekly_monthly_analysis" / "2003_20260306_15m" / "eurjpy" / "derived_15m" / "eurjpy_15m_20030804_20260306.csv"
DAILY_PATH = ROOT / "pair_workspaces" / "eurjpy_2003_20260306_15m_v1" / "eurjpy_daily_ohlcv_ver3.csv"
OUT_DIR = Path(__file__).parent

PIP = 0.01
SPREAD_PIPS = 1.0
MAX_HOLD_HOURS = 720
MAX_HOLD_BARS = MAX_HOLD_HOURS * 4
START_DATE = pd.Timestamp("2003-01-01")
END_DATE = pd.Timestamp("2015-12-31")
MIN_NORM_DENOM_PIPS = 10.0


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
    m = m[m["dkey"].isin(full_days)].copy().reset_index(drop=True)

    day_first = m.groupby("dkey").head(1).reset_index()[["dkey", "index"]]
    day_to_i0 = {r["dkey"]: int(r["index"]) for _, r in day_first.iterrows()}

    info = {}
    for _, r in d.iterrows():
        k = r["dkey"]
        if k in day_to_i0 and START_DATE <= k <= END_DATE:
            info[k] = {
                "pp": float(r["pp"]),
                "r2": float(r["r2"]),
                "s2": float(r["s2"]),
                "i0": day_to_i0[k],
            }
    return m, info


def close_pips(side: str, entry: float, px: float) -> float:
    if side == "BUY":
        return ((px - entry) / PIP) - SPREAD_PIPS
    return ((entry - px) / PIP) - SPREAD_PIPS


def run_diag(m: pd.DataFrame, info: dict) -> pd.DataFrame:
    opens = m["Open"].to_numpy(float)
    highs = m["High"].to_numpy(float)
    lows = m["Low"].to_numpy(float)
    closes = m["Close"].to_numpy(float)

    rows = []
    min_denom_px = MIN_NORM_DENOM_PIPS * PIP
    for d in sorted(info.keys()):
        v = info[d]
        pp = v["pp"]
        r2 = v["r2"]
        s2 = v["s2"]
        i0 = v["i0"]
        entry = opens[i0]

        if entry < pp:
            side = "BUY"
            tp = pp
            sl = entry - (pp - entry)
            denom = pp - s2
            if denom < min_denom_px:
                continue
            norm_entry = (pp - entry) / denom
        elif entry > pp:
            side = "SELL"
            tp = pp
            sl = entry + (entry - pp)
            denom = r2 - pp
            if denom < min_denom_px:
                continue
            norm_entry = (entry - pp) / denom
        else:
            continue

        start_i = i0 + 1
        if start_i >= len(opens):
            continue
        end_i = min(start_i + MAX_HOLD_BARS, len(opens) - 1)

        min_low = entry
        max_high = entry
        done = False
        outcome = "TIME"
        exit_px = closes[end_i]
        exit_i = end_i

        for j in range(start_i, end_i + 1):
            hi = highs[j]
            lo = lows[j]
            if lo < min_low:
                min_low = lo
            if hi > max_high:
                max_high = hi

            if side == "BUY":
                hit_sl = lo <= sl
                hit_tp = hi >= tp
            else:
                hit_sl = hi >= sl
                hit_tp = lo <= tp

            if hit_sl and hit_tp:
                done = True
                outcome = "LOSS"
                exit_px = sl
                exit_i = j
                break
            if hit_sl:
                done = True
                outcome = "LOSS"
                exit_px = sl
                exit_i = j
                break
            if hit_tp:
                done = True
                outcome = "WIN"
                exit_px = tp
                exit_i = j
                break

        if not done:
            outcome = "TIME"
            exit_px = closes[end_i]
            exit_i = end_i

        if side == "BUY":
            adverse_norm = max((entry - min_low) / denom, 0.0)
            from_pivot_extreme_norm = max((pp - min_low) / denom, 0.0)
        else:
            adverse_norm = max((max_high - entry) / denom, 0.0)
            from_pivot_extreme_norm = max((max_high - pp) / denom, 0.0)

        rows.append(
            {
                "date": d,
                "side": side,
                "entry": entry,
                "pp": pp,
                "r2": r2,
                "s2": s2,
                "norm_entry_dist_r2s2": norm_entry,
                "adverse_norm_from_entry": adverse_norm,
                "extreme_norm_from_pivot": from_pivot_extreme_norm,
                "outcome": outcome,
                "pips": close_pips(side, entry, exit_px),
                "bars_held": int(exit_i - i0),
            }
        )

    return pd.DataFrame(rows)


def summarize(df: pd.DataFrame) -> dict:
    wr_df = df[df["outcome"].isin(["WIN", "LOSS"])].copy()
    wr_df["is_win"] = (wr_df["outcome"] == "WIN").astype(int)

    # Bin by normalized entry distance and find closest to 50%.
    bins = np.arange(0.0, 2.05, 0.05)
    wr_df["norm_bin"] = pd.cut(wr_df["norm_entry_dist_r2s2"], bins=bins, right=False)
    gb = wr_df.groupby("norm_bin", observed=False).agg(
        trades=("is_win", "size"),
        wins=("is_win", "sum"),
        win_rate=("is_win", "mean"),
        avg_pips=("pips", "mean"),
    ).reset_index()
    gb = gb[gb["trades"] >= 50].copy()
    if not gb.empty:
        gb["dist_to_50"] = (gb["win_rate"] - 0.5).abs()
        near50 = gb.sort_values(["dist_to_50", "trades"], ascending=[True, False]).iloc[0].to_dict()
        near50["norm_bin"] = str(near50["norm_bin"])
    else:
        near50 = None

    wins_only = wr_df[wr_df["outcome"] == "WIN"]

    gb_out = gb.copy()
    if not gb_out.empty:
        gb_out["norm_bin"] = gb_out["norm_bin"].astype(str)

    return {
        "period": [str(START_DATE.date()), str(END_DATE.date())],
        "effective_first_day": str(df["date"].min().date()) if not df.empty else None,
        "effective_last_day": str(df["date"].max().date()) if not df.empty else None,
        "trades_total": int(len(df)),
        "wins": int((df["outcome"] == "WIN").sum()),
        "losses": int((df["outcome"] == "LOSS").sum()),
        "time_exits": int((df["outcome"] == "TIME").sum()),
        "win_rate_excl_time": float(round((df["outcome"] == "WIN").sum() / max(((df["outcome"] == "WIN") | (df["outcome"] == "LOSS")).sum(), 1), 6)),
        "cal_ppd": float(round(df["pips"].sum() / max(df["date"].nunique(), 1), 4)),
        "avg_extreme_norm_from_pivot_all": float(round(df["extreme_norm_from_pivot"].mean(), 4)),
        "avg_extreme_norm_from_pivot_wins": float(round(wins_only["extreme_norm_from_pivot"].mean(), 4)) if not wins_only.empty else None,
        "avg_adverse_norm_from_entry_all": float(round(df["adverse_norm_from_entry"].mean(), 4)),
        "avg_adverse_norm_from_entry_wins": float(round(wins_only["adverse_norm_from_entry"].mean(), 4)) if not wins_only.empty else None,
        "winrate_nearest_50_bin": near50,
        "bin_table": gb_out.to_dict(orient="records"),
    }


def main():
    m, info = load_data()
    df = run_diag(m, info)
    out = summarize(df)

    out_json = OUT_DIR / "dayopen_to_pivot_equal_sl_r2s2_norm_diag_2003_2015_eurjpy.json"
    out_csv = OUT_DIR / "dayopen_to_pivot_equal_sl_r2s2_norm_diag_2003_2015_eurjpy_trades.csv"
    out_json.write_text(json.dumps(out, indent=2), encoding="utf-8")
    df.to_csv(out_csv, index=False)

    print("SUMMARY")
    print(json.dumps({k: v for k, v in out.items() if k != "bin_table"}, indent=2))
    if out["winrate_nearest_50_bin"] is not None:
        b = out["winrate_nearest_50_bin"]
        print(
            f"nearest_50_bin={b['norm_bin']} trades={int(b['trades'])} "
            f"win_rate={b['win_rate']:.4f} avg_pips={b['avg_pips']:.4f}"
        )
    print(f"saved={out_json}")
    print(f"saved={out_csv}")


if __name__ == "__main__":
    main()

