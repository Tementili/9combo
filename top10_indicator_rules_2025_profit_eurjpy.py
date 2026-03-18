from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(r"C:\Users\temen\5 joulu\TrailingStopLoss pivot peak trading 6 march 2026")
DAILY_PATH = ROOT / "pair_workspaces" / "eurjpy_2003_20260306_15m_v1" / "eurjpy_daily_ohlcv_ver3.csv"
M15_PATH = ROOT / "forex raw data" / "Dukascopy" / "weekly_monthly_analysis" / "2003_20260306_15m" / "eurjpy" / "derived_15m" / "eurjpy_15m_20030804_20260306.csv"
RULES_PATH = Path(__file__).parent / "top10_indicators_sell_bucket_results_0_35.csv"
OUT_DIR = Path(__file__).parent

YEAR = 2025
PIP = 0.01
SPREAD_PIPS = 1.0
SL_NORM = 0.55
MAX_BUCKET_NORM = 0.55
MIN_DENOM_PIPS = 10.0
KEEP_BUCKETS = {"00-05%", "05-10%", "10-15%", "15-20%", "20-25%", "25-30%", "30-35%"}


def rsi_wilder(close: pd.Series, period: int) -> pd.Series:
    delta = close.diff()
    up = delta.clip(lower=0.0)
    down = -delta.clip(upper=0.0)
    roll_up = up.ewm(alpha=1.0 / period, adjust=False).mean()
    roll_down = down.ewm(alpha=1.0 / period, adjust=False).mean()
    rs = roll_up / roll_down.replace(0.0, float("nan"))
    return 100.0 - (100.0 / (1.0 + rs))


def bin_quintiles(s: pd.Series) -> pd.Series:
    try:
        return pd.qcut(s, q=5, labels=["Q1", "Q2", "Q3", "Q4", "Q5"], duplicates="drop").astype("object")
    except Exception:
        return pd.Series([np.nan] * len(s), index=s.index, dtype="object")


def build_daily_features() -> pd.DataFrame:
    d = pd.read_csv(DAILY_PATH, usecols=["Date", "Open", "High", "Low", "Close"], low_memory=False)
    d["Date"] = pd.to_datetime(d["Date"], errors="coerce")
    for c in ["Open", "High", "Low", "Close"]:
        d[c] = pd.to_numeric(d[c], errors="coerce")
    d = d.dropna().sort_values("Date").reset_index(drop=True)
    d["date"] = d["Date"].dt.normalize()

    # Pivot ladder from previous day.
    d["yh"] = d["High"].shift(1)
    d["yl"] = d["Low"].shift(1)
    d["yc"] = d["Close"].shift(1)
    d["pp"] = (d["yh"] + d["yl"] + d["yc"]) / 3.0
    d["r2"] = d["pp"] + (d["yh"] - d["yl"])
    d["s2"] = d["pp"] - (d["yh"] - d["yl"])

    prev_close = d["Close"].shift(1)
    tr = pd.concat(
        [
            (d["High"] - d["Low"]).abs(),
            (d["High"] - prev_close).abs(),
            (d["Low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    d["tr"] = tr
    d["atr14"] = tr.rolling(14).mean()
    d["natr14"] = 100.0 * d["atr14"] / d["Close"]

    ema12 = d["Close"].ewm(span=12, adjust=False).mean()
    ema26 = d["Close"].ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    d["macd_hist"] = macd - macd.ewm(span=9, adjust=False).mean()

    d["roc10"] = d["Close"].pct_change(10) * 100.0
    d["rsi14"] = rsi_wilder(d["Close"], 14)
    d["rsi2"] = rsi_wilder(d["Close"], 2)

    tp = (d["High"] + d["Low"] + d["Close"]) / 3.0
    sma_tp = tp.rolling(20).mean()
    md = (tp - sma_tp).abs().rolling(20).mean()
    d["cci20"] = (tp - sma_tp) / (0.015 * md.replace(0.0, float("nan")))

    sma20 = d["Close"].rolling(20).mean()
    d["sma20_dist"] = 100.0 * (d["Close"] - sma20) / sma20.replace(0.0, float("nan"))

    # ADX / DI-
    up_move = d["High"].diff()
    down_move = -d["Low"].diff()
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    atr_w = tr.ewm(alpha=1 / 14, adjust=False).mean()
    d["di_minus14"] = 100.0 * pd.Series(minus_dm).ewm(alpha=1 / 14, adjust=False).mean() / atr_w

    # EMA distance / keltner-like / hl-range
    ema20 = d["Close"].ewm(span=20, adjust=False).mean()
    d["ema20_dist"] = 100.0 * (d["Close"] - ema20) / ema20.replace(0.0, float("nan"))
    d["keltner_pos"] = (d["Close"] - ema20) / d["atr14"].replace(0.0, float("nan"))
    d["hl_range_pct"] = 100.0 * (d["High"] - d["Low"]) / d["Close"].replace(0.0, float("nan"))

    features = [
        "atr14",
        "cci20",
        "di_minus14",
        "ema20_dist",
        "hl_range_pct",
        "keltner_pos",
        "roc10",
        "rsi14",
        "rsi2",
        "sma20_dist",
        "tr",
        "natr14",
        "pp",
        "r2",
        "s2",
        "Open",
    ]
    out = d[["date"] + features].copy()
    for c in features:
        if c in {"pp", "r2", "s2", "Open"}:
            continue
        out[f"{c}_bin"] = bin_quintiles(d[c].shift(1))
    return out


def event_trades_2025(m15: pd.DataFrame, day_info: pd.DataFrame) -> pd.DataFrame:
    m15 = m15.copy()
    m15["Datetime"] = pd.to_datetime(m15["Datetime"], errors="coerce")
    for c in ["Open", "High", "Low", "Close"]:
        m15[c] = pd.to_numeric(m15[c], errors="coerce")
    m15 = m15.dropna().sort_values("Datetime").reset_index(drop=True)
    m15["dkey"] = m15["Datetime"].dt.normalize()

    full_days = set(m15.groupby("dkey").size().pipe(lambda s: s[s == 96]).index)
    m15 = m15[m15["dkey"].isin(full_days)].copy().reset_index(drop=True)
    first_idx = m15.groupby("dkey").head(1).reset_index()[["dkey", "index"]]
    last_idx = m15.groupby("dkey").tail(1).reset_index()[["dkey", "index"]]
    i0_map = {r["dkey"]: int(r["index"]) for _, r in first_idx.iterrows()}
    i1_map = {r["dkey"]: int(r["index"]) for _, r in last_idx.iterrows()}

    hi = m15["High"].to_numpy(float)
    lo = m15["Low"].to_numpy(float)
    cl = m15["Close"].to_numpy(float)

    out_rows = []
    for _, r in day_info.iterrows():
        d = r["date"]
        if d not in i0_map or d not in i1_map:
            continue
        i0 = i0_map[d]
        entry = float(r["Open"])
        pp = float(r["pp"])
        r2 = float(r["r2"])
        s2 = float(r["s2"])

        if entry < pp:
            side = "BUY"
            denom = pp - s2
            if denom < MIN_DENOM_PIPS * PIP:
                continue
            entry_norm = (pp - entry) / denom
            sl = pp - SL_NORM * denom
        elif entry > pp:
            side = "SELL"
            denom = r2 - pp
            if denom < MIN_DENOM_PIPS * PIP:
                continue
            entry_norm = (entry - pp) / denom
            sl = pp + SL_NORM * denom
        else:
            continue

        if not (0.0 <= entry_norm <= MAX_BUCKET_NORM):
            continue

        # 5% bucket label.
        b0 = int(np.floor(entry_norm / 0.05) * 5)
        b1 = b0 + 5
        bucket = f"{b0:02d}-{b1:02d}%"

        # First event: pivot or SL, multi-day until hit.
        state = "none"
        exit_px = cl[-1]
        for j in range(i0, len(hi)):
            if side == "BUY":
                hit_sl = lo[j] <= sl
                hit_pv = hi[j] >= pp
            else:
                hit_sl = hi[j] >= sl
                hit_pv = lo[j] <= pp
            if hit_sl and hit_pv:
                state = "sl_first_tie"
                exit_px = sl
                break
            if hit_sl:
                state = "sl_first"
                exit_px = sl
                break
            if hit_pv:
                state = "pivot_first"
                exit_px = pp
                break

        pips = ((entry - exit_px) / PIP) - SPREAD_PIPS if side == "SELL" else ((exit_px - entry) / PIP) - SPREAD_PIPS
        out_rows.append({"date": d, "side": side, "bucket": bucket, "pips_at_first_event": pips, "entry": entry, "pp": pp, "r2": r2, "s2": s2})

    return pd.DataFrame(out_rows)


def main():
    rules = pd.read_csv(RULES_PATH)
    # Keep only SELL rules and relevant columns.
    rules = rules[rules["side"] == "SELL"][["indicator", "bucket", "bin"]].copy()

    daily = build_daily_features()
    daily_2025 = daily[(daily["date"] >= pd.Timestamp(f"{YEAR}-01-01")) & (daily["date"] <= pd.Timestamp(f"{YEAR}-12-31"))].copy()

    m15 = pd.read_csv(M15_PATH, usecols=["Datetime", "Open", "High", "Low", "Close"], low_memory=False)
    trades = event_trades_2025(m15, daily_2025[["date", "Open", "pp", "r2", "s2"]])
    trades = trades[(trades["side"] == "SELL") & (trades["bucket"].isin(KEEP_BUCKETS))].copy()

    x = trades.merge(daily_2025, on="date", how="left")

    # Evaluate each indicator rule-map separately.
    rows = []
    selected_masks = []
    for ind in sorted(rules["indicator"].unique()):
        r = rules[rules["indicator"] == ind].copy()
        bin_col = f"{ind}_bin"
        if bin_col not in x.columns:
            continue
        mask = pd.Series(False, index=x.index)
        for _, rr in r.iterrows():
            mask = mask | ((x["bucket"] == rr["bucket"]) & (x[bin_col].astype(str) == str(rr["bin"])))
        g = x[mask].copy()
        selected_masks.append(mask)
        if g.empty:
            rows.append({"indicator": ind, "trades": 0})
            continue
        wins = (g["pips_at_first_event"] > 0).sum()
        losses = (g["pips_at_first_event"] <= 0).sum()
        sum_win = g.loc[g["pips_at_first_event"] > 0, "pips_at_first_event"].sum()
        sum_loss = abs(g.loc[g["pips_at_first_event"] <= 0, "pips_at_first_event"].sum())
        rows.append(
            {
                "indicator": ind,
                "trades": int(len(g)),
                "win_rate": float(wins / len(g)),
                "total_pips": float(g["pips_at_first_event"].sum()),
                "avg_pips_per_trade": float(g["pips_at_first_event"].mean()),
                "profit_factor": float(sum_win / sum_loss) if sum_loss > 0 else np.nan,
            }
        )

    per_indicator = pd.DataFrame(rows).sort_values("total_pips", ascending=False).reset_index(drop=True)

    # Combined union of all selected rules (no double-counting same trade).
    if selected_masks:
        union_mask = selected_masks[0].copy()
        for m in selected_masks[1:]:
            union_mask = union_mask | m
        u = x[union_mask].copy()
    else:
        u = x.iloc[0:0].copy()
    if not u.empty:
        uw = (u["pips_at_first_event"] > 0).sum()
        ul = (u["pips_at_first_event"] <= 0).sum()
        usumw = u.loc[u["pips_at_first_event"] > 0, "pips_at_first_event"].sum()
        usuml = abs(u.loc[u["pips_at_first_event"] <= 0, "pips_at_first_event"].sum())
        union = {
            "trades": int(len(u)),
            "win_rate": float(uw / len(u)),
            "total_pips": float(u["pips_at_first_event"].sum()),
            "avg_pips_per_trade": float(u["pips_at_first_event"].mean()),
            "profit_factor": float(usumw / usuml) if usuml > 0 else np.nan,
        }
    else:
        union = {"trades": 0}

    out_json = OUT_DIR / "top10_indicator_rules_2025_profit_eurjpy.json"
    out_csv = OUT_DIR / "top10_indicator_rules_2025_profit_eurjpy.csv"
    per_indicator.to_csv(out_csv, index=False)
    out_json.write_text(
        json.dumps(
            {
                "year": YEAR,
                "pair": "EURJPY",
                "rules_source": RULES_PATH.name,
                "per_indicator": per_indicator.to_dict(orient="records"),
                "combined_union": union,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print("PER INDICATOR 2025")
    print(per_indicator.to_string(index=False))
    print("\nCOMBINED UNION 2025")
    print(union)
    print(f"saved={out_json}")
    print(f"saved={out_csv}")


if __name__ == "__main__":
    main()

