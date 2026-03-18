from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(r"C:\Users\temen\5 joulu\TrailingStopLoss pivot peak trading 6 march 2026")
DAILY_PATH = ROOT / "pair_workspaces" / "eurjpy_2003_20260306_15m_v1" / "eurjpy_daily_ohlcv_ver3.csv"
TRADES_PATH = Path(__file__).parent / "dayopen_pivotside_sl55_bucket5pct_diag_2003_2015_eurjpy_trades.csv"
OUT_DIR = Path(__file__).parent

START_DATE = pd.Timestamp("2003-01-01")
END_DATE = pd.Timestamp("2015-12-31")
PIP = 0.01
SPREAD_PIPS = 1.0
SL_NORM = 0.55
MIN_GROUP_TRADES = 30
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
        b = pd.qcut(s, q=5, labels=["Q1", "Q2", "Q3", "Q4", "Q5"], duplicates="drop")
        return b.astype("object")
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

    # RSI family.
    d["rsi2"] = rsi_wilder(d["Close"], 2)
    d["rsi5"] = rsi_wilder(d["Close"], 5)
    d["rsi14"] = rsi_wilder(d["Close"], 14)

    # Stochastic.
    hh14 = d["High"].rolling(14).max()
    ll14 = d["Low"].rolling(14).min()
    d["stochk14"] = 100.0 * (d["Close"] - ll14) / (hh14 - ll14).replace(0.0, float("nan"))
    d["stochd14"] = d["stochk14"].rolling(3).mean()

    # Williams %R.
    d["wr14"] = -100.0 * (hh14 - d["Close"]) / (hh14 - ll14).replace(0.0, float("nan"))

    # CCI20.
    tp = (d["High"] + d["Low"] + d["Close"]) / 3.0
    sma_tp = tp.rolling(20).mean()
    md = (tp - sma_tp).abs().rolling(20).mean()
    d["cci20"] = (tp - sma_tp) / (0.015 * md.replace(0.0, float("nan")))

    # Bollinger zscore.
    sma20 = d["Close"].rolling(20).mean()
    std20 = d["Close"].rolling(20).std()
    d["bbz20"] = (d["Close"] - sma20) / std20.replace(0.0, float("nan"))

    # MACD.
    ema12 = d["Close"].ewm(span=12, adjust=False).mean()
    ema26 = d["Close"].ewm(span=26, adjust=False).mean()
    d["macd"] = ema12 - ema26
    d["macd_signal"] = d["macd"].ewm(span=9, adjust=False).mean()
    d["macd_hist"] = d["macd"] - d["macd_signal"]

    # ADX / DI.
    up_move = d["High"].diff()
    down_move = -d["Low"].diff()
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    atr_w = tr.ewm(alpha=1 / 14, adjust=False).mean()
    plus_di = 100.0 * pd.Series(plus_dm).ewm(alpha=1 / 14, adjust=False).mean() / atr_w
    minus_di = 100.0 * pd.Series(minus_dm).ewm(alpha=1 / 14, adjust=False).mean() / atr_w
    dx = 100.0 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0.0, float("nan"))
    d["adx14"] = dx.ewm(alpha=1 / 14, adjust=False).mean()
    d["di_plus14"] = plus_di
    d["di_minus14"] = minus_di
    d["di_spread14"] = plus_di - minus_di

    # ROC / momentum.
    d["roc1"] = d["Close"].pct_change(1) * 100.0
    d["roc3"] = d["Close"].pct_change(3) * 100.0
    d["roc5"] = d["Close"].pct_change(5) * 100.0
    d["roc10"] = d["Close"].pct_change(10) * 100.0

    # MA distances.
    sma50 = d["Close"].rolling(50).mean()
    ema20 = d["Close"].ewm(span=20, adjust=False).mean()
    d["sma20_dist"] = 100.0 * (d["Close"] - sma20) / sma20.replace(0.0, float("nan"))
    d["sma50_dist"] = 100.0 * (d["Close"] - sma50) / sma50.replace(0.0, float("nan"))
    d["ema20_dist"] = 100.0 * (d["Close"] - ema20) / ema20.replace(0.0, float("nan"))

    # Donchian / range position.
    hh20 = d["High"].rolling(20).max()
    ll20 = d["Low"].rolling(20).min()
    d["range_pos20"] = (d["Close"] - ll20) / (hh20 - ll20).replace(0.0, float("nan"))
    d["donchian_width20"] = 100.0 * (hh20 - ll20) / d["Close"].replace(0.0, float("nan"))

    # Parkinson volatility.
    ln_hl = np.log((d["High"] / d["Low"]).replace(0.0, float("nan")))
    d["park_vol10"] = (ln_hl.pow(2).rolling(10).mean() / (4.0 * np.log(2.0))).pow(0.5)

    # Keltner position-ish: distance from ema20 in ATR units.
    d["keltner_pos"] = (d["Close"] - ema20) / d["atr14"].replace(0.0, float("nan"))

    # HL range percent.
    d["hl_range_pct"] = 100.0 * (d["High"] - d["Low"]) / d["Close"].replace(0.0, float("nan"))

    feature_cols = [
        "atr14",
        "natr14",
        "rsi2",
        "rsi5",
        "rsi14",
        "stochk14",
        "stochd14",
        "wr14",
        "cci20",
        "bbz20",
        "macd",
        "macd_signal",
        "macd_hist",
        "adx14",
        "di_plus14",
        "di_minus14",
        "di_spread14",
        "roc1",
        "roc3",
        "roc5",
        "roc10",
        "sma20_dist",
        "sma50_dist",
        "ema20_dist",
        "range_pos20",
        "donchian_width20",
        "park_vol10",
        "tr",
        "keltner_pos",
        "hl_range_pct",
    ]

    # Shift to prior-day signal and create quintile bins.
    out = d[["date", "Open", "pp", "r2", "s2"]].copy()
    for c in feature_cols:
        sig = d[c].shift(1)
        out[f"{c}_bin"] = bin_quintiles(sig)

    return out


def main():
    trades = pd.read_csv(TRADES_PATH)
    trades["date"] = pd.to_datetime(trades["date"], errors="coerce").dt.normalize()
    trades = trades[(trades["date"] >= START_DATE) & (trades["date"] <= END_DATE)].copy()
    trades = trades[trades["bucket"].isin(KEEP_BUCKETS)].copy()

    daily = build_daily_features()
    x = trades.merge(daily, on="date", how="left")

    # Compute per-trade risk in pips from hard SL=55%.
    def risk_pips(row):
        entry = row["Open"]
        pp = row["pp"]
        if row["side"] == "BUY":
            denom = pp - row["s2"]
            sl = pp - SL_NORM * denom
            return max((entry - sl) / PIP + SPREAD_PIPS, 1e-9)
        denom = row["r2"] - pp
        sl = pp + SL_NORM * denom
        return max((sl - entry) / PIP + SPREAD_PIPS, 1e-9)

    x["risk_pips"] = x.apply(risk_pips, axis=1)
    x["r_multiple"] = x["pips_at_first_event"] / x["risk_pips"]

    indicator_bin_cols = [c for c in x.columns if c.endswith("_bin")]
    rows = []
    for ind_col in indicator_bin_cols:
        part = x.dropna(subset=[ind_col]).copy()
        g = (
            part.groupby(["side", "bucket", ind_col], observed=False)
            .agg(
                trades=("pips_at_first_event", "size"),
                wins=("pips_at_first_event", lambda s: int((s > 0).sum())),
                losses=("pips_at_first_event", lambda s: int((s <= 0).sum())),
                avg_win_pips=("pips_at_first_event", lambda s: float(s[s > 0].mean()) if (s > 0).any() else float("nan")),
                avg_loss_pips=("pips_at_first_event", lambda s: float(abs(s[s <= 0].mean())) if (s <= 0).any() else float("nan")),
                expectancy_pips=("pips_at_first_event", "mean"),
                avg_r=("r_multiple", "mean"),
                sum_win_pips=("pips_at_first_event", lambda s: float(s[s > 0].sum())),
                sum_loss_pips=("pips_at_first_event", lambda s: float(abs(s[s <= 0].sum()))),
            )
            .reset_index()
            .rename(columns={ind_col: "bin"})
        )
        g["indicator"] = ind_col.replace("_bin", "")
        g = g[g["trades"] >= MIN_GROUP_TRADES].copy()
        g["win_rate"] = g["wins"] / g["trades"]
        g["rr_ratio"] = g["avg_win_pips"] / g["avg_loss_pips"]
        g["profit_factor"] = g["sum_win_pips"] / g["sum_loss_pips"]
        rows.append(g)

    all_df = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    all_df = all_df.sort_values(["indicator", "side", "bucket", "avg_r"], ascending=[True, True, True, False]).reset_index(drop=True)

    # Best bin per indicator per side+bucket by ROI (avg_r), tie-break expectancy.
    best = (
        all_df.sort_values(["indicator", "side", "bucket", "avg_r", "expectancy_pips", "trades"], ascending=[True, True, True, False, False, False])
        .groupby(["indicator", "side", "bucket"], as_index=False)
        .head(1)
        .reset_index(drop=True)
    )

    # Indicator-level overall summary (weighted across buckets/states).
    summ_rows = []
    for ind, g in all_df.groupby("indicator"):
        w = g["trades"].sum()
        summ_rows.append(
            {
                "indicator": ind,
                "groups": int(len(g)),
                "trades_total": int(w),
                "avg_r_weighted": float((g["avg_r"] * g["trades"]).sum() / max(w, 1)),
                "rr_weighted": float((g["rr_ratio"] * g["trades"]).sum() / max(w, 1)),
                "expectancy_pips_weighted": float((g["expectancy_pips"] * g["trades"]).sum() / max(w, 1)),
                "profit_factor_weighted": float((g["profit_factor"] * g["trades"]).sum() / max(w, 1)),
            }
        )
    summ = pd.DataFrame(summ_rows).sort_values("avg_r_weighted", ascending=False).reset_index(drop=True)

    out_json = OUT_DIR / "bucket_0_35_indicator30_roi_rr_scan_2003_2015_eurjpy.json"
    out_all = OUT_DIR / "bucket_0_35_indicator30_roi_rr_scan_2003_2015_eurjpy_all.csv"
    out_best = OUT_DIR / "bucket_0_35_indicator30_roi_rr_scan_2003_2015_eurjpy_best_per_bucket.csv"
    out_summ = OUT_DIR / "bucket_0_35_indicator30_roi_rr_scan_2003_2015_eurjpy_indicator_summary.csv"
    all_df.to_csv(out_all, index=False)
    best.to_csv(out_best, index=False)
    summ.to_csv(out_summ, index=False)

    payload = {
        "pair": "EURJPY",
        "period": [str(START_DATE.date()), str(END_DATE.date())],
        "buckets_used": sorted(KEEP_BUCKETS),
        "sl_norm": SL_NORM,
        "min_group_trades": MIN_GROUP_TRADES,
        "indicators_count": 30,
        "files": {
            "all_groups": out_all.name,
            "best_per_bucket": out_best.name,
            "indicator_summary": out_summ.name,
        },
    }
    out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print("TOP INDICATORS BY ROI (avg_r_weighted)")
    print(summ.head(15).to_string(index=False))
    print("\nTOP BEST-PER-BUCKET ROWS BY ROI")
    print(best.sort_values("avg_r", ascending=False).head(20)[["indicator", "side", "bucket", "bin", "trades", "avg_r", "rr_ratio", "expectancy_pips", "profit_factor"]].to_string(index=False))
    print(f"saved={out_json}")
    print(f"saved={out_all}")
    print(f"saved={out_best}")
    print(f"saved={out_summ}")


if __name__ == "__main__":
    main()

