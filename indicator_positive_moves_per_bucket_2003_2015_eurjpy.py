from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

ROOT = Path(r"C:\Users\temen\5 joulu\TrailingStopLoss pivot peak trading 6 march 2026")
DAILY_PATH = ROOT / "pair_workspaces" / "eurjpy_2003_20260306_15m_v1" / "eurjpy_daily_ohlcv_ver3.csv"
OUT_DIR = Path(__file__).parent
TRADES_PATH = OUT_DIR / "dayopen_pivotside_sl55_bucket5pct_diag_2003_2015_eurjpy_trades.csv"

START_DATE = pd.Timestamp("2003-01-01")
END_DATE = pd.Timestamp("2015-12-31")
MIN_GROUP_TRADES = 30


def rsi_wilder(close: pd.Series, period: int) -> pd.Series:
    delta = close.diff()
    up = delta.clip(lower=0.0)
    down = -delta.clip(upper=0.0)
    roll_up = up.ewm(alpha=1.0 / period, adjust=False).mean()
    roll_down = down.ewm(alpha=1.0 / period, adjust=False).mean()
    rs = roll_up / roll_down.replace(0.0, float("nan"))
    return 100.0 - (100.0 / (1.0 + rs))


def build_daily_indicators() -> pd.DataFrame:
    d = pd.read_csv(DAILY_PATH, usecols=["Date", "High", "Low", "Close"], low_memory=False)
    d["Date"] = pd.to_datetime(d["Date"], errors="coerce")
    for c in ["High", "Low", "Close"]:
        d[c] = pd.to_numeric(d[c], errors="coerce")
    d = d.dropna().sort_values("Date").reset_index(drop=True)
    d["date"] = d["Date"].dt.normalize()

    prev_close = d["Close"].shift(1)
    tr = pd.concat(
        [
            (d["High"] - d["Low"]).abs(),
            (d["High"] - prev_close).abs(),
            (d["Low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    d["atr14"] = tr.rolling(14).mean()
    d["atr14_sig"] = d["atr14"].shift(1)
    d["atr_bucket"] = pd.qcut(d["atr14_sig"], q=5, labels=["ATR_Q1", "ATR_Q2", "ATR_Q3", "ATR_Q4", "ATR_Q5"])

    d["rsi2_sig"] = rsi_wilder(d["Close"], 2).shift(1)
    d["rsi_band"] = pd.cut(
        d["rsi2_sig"],
        bins=[-1, 5, 10, 20, 80, 90, 95, 101],
        labels=["RSI_0_5", "RSI_5_10", "RSI_10_20", "RSI_20_80", "RSI_80_90", "RSI_90_95", "RSI_95_100"],
        right=False,
    )

    hh14 = d["High"].rolling(14).max()
    ll14 = d["Low"].rolling(14).min()
    d["wr14"] = -100.0 * (hh14 - d["Close"]) / (hh14 - ll14).replace(0.0, float("nan"))
    d["wr14_sig"] = d["wr14"].shift(1)
    d["wr_band"] = pd.cut(
        d["wr14_sig"],
        bins=[-101, -95, -90, -80, -20, -10, -5, 1],
        labels=["WR_-100_-95", "WR_-95_-90", "WR_-90_-80", "WR_-80_-20", "WR_-20_-10", "WR_-10_-5", "WR_-5_0"],
        right=False,
    )

    sma20 = d["Close"].rolling(20).mean()
    std20 = d["Close"].rolling(20).std()
    d["bbz"] = (d["Close"] - sma20) / std20.replace(0.0, float("nan"))
    d["bbz_sig"] = d["bbz"].shift(1)
    d["bbz_band"] = pd.cut(
        d["bbz_sig"],
        bins=[-999, -2.5, -2.0, -1.5, 1.5, 2.0, 2.5, 999],
        labels=["BBZ_<-2.5", "BBZ_-2.5_-2.0", "BBZ_-2.0_-1.5", "BBZ_-1.5_1.5", "BBZ_1.5_2.0", "BBZ_2.0_2.5", "BBZ_>2.5"],
        right=False,
    )

    return d[["date", "atr_bucket", "rsi_band", "wr_band", "bbz_band"]].copy()


def summarize_for_indicator(x: pd.DataFrame, col: str) -> pd.DataFrame:
    g = (
        x.groupby(["side", "bucket", col], observed=False)
        .agg(
            trades=("pips_at_first_event", "size"),
            positive_trades=("pips_at_first_event", lambda s: int((s > 0).sum())),
            avg_positive_pips=("pips_at_first_event", lambda s: float(s[s > 0].mean()) if (s > 0).any() else float("nan")),
            median_positive_pips=("pips_at_first_event", lambda s: float(s[s > 0].median()) if (s > 0).any() else float("nan")),
            positive_rate=("pips_at_first_event", lambda s: float((s > 0).mean())),
            avg_all_pips=("pips_at_first_event", "mean"),
            pivot_first_rate=("pivot_hit_before_sl", "mean"),
        )
        .reset_index()
        .rename(columns={col: "indicator_bucket"})
    )
    g = g[g["trades"] >= MIN_GROUP_TRADES].copy()
    g["indicator"] = col
    return g


def main():
    trades = pd.read_csv(TRADES_PATH)
    trades["date"] = pd.to_datetime(trades["date"], errors="coerce").dt.normalize()
    trades = trades[(trades["date"] >= START_DATE) & (trades["date"] <= END_DATE)].copy()

    daily = build_daily_indicators()
    x = trades.merge(daily, on="date", how="left")

    indicators = ["atr_bucket", "rsi_band", "wr_band", "bbz_band"]
    out_tables = {}
    all_rows = []

    for ind in indicators:
        t = summarize_for_indicator(x.dropna(subset=[ind]).copy(), ind)
        t = t.sort_values(["side", "bucket", "avg_positive_pips"], ascending=[True, True, False]).reset_index(drop=True)
        out_tables[ind] = t
        all_rows.append(t)

        out_csv = OUT_DIR / f"indicator_positive_moves_{ind}_per_bucket_2003_2015_eurjpy.csv"
        t.to_csv(out_csv, index=False)
        print(f"{ind}: rows={len(t)} saved={out_csv}")

    all_df = pd.concat(all_rows, ignore_index=True) if all_rows else pd.DataFrame()
    all_csv = OUT_DIR / "indicator_positive_moves_all_per_bucket_2003_2015_eurjpy.csv"
    all_df.to_csv(all_csv, index=False)

    # Top 5 indicator states per side/bucket by avg positive pips
    top5 = (
        all_df.sort_values(["side", "bucket", "avg_positive_pips", "positive_trades"], ascending=[True, True, False, False])
        .groupby(["side", "bucket"], as_index=False)
        .head(5)
        .reset_index(drop=True)
    ) if not all_df.empty else pd.DataFrame()
    top5_csv = OUT_DIR / "indicator_positive_moves_top5_per_bucket_2003_2015_eurjpy.csv"
    top5.to_csv(top5_csv, index=False)

    out_json = OUT_DIR / "indicator_positive_moves_per_bucket_2003_2015_eurjpy.json"
    payload = {
        "pair": "EURJPY",
        "period": [str(START_DATE.date()), str(END_DATE.date())],
        "metric": "average positive pips (pips_at_first_event > 0)",
        "min_group_trades": MIN_GROUP_TRADES,
        "files": {
            "atr_bucket": "indicator_positive_moves_atr_bucket_per_bucket_2003_2015_eurjpy.csv",
            "rsi_band": "indicator_positive_moves_rsi_band_per_bucket_2003_2015_eurjpy.csv",
            "wr_band": "indicator_positive_moves_wr_band_per_bucket_2003_2015_eurjpy.csv",
            "bbz_band": "indicator_positive_moves_bbz_band_per_bucket_2003_2015_eurjpy.csv",
            "all": "indicator_positive_moves_all_per_bucket_2003_2015_eurjpy.csv",
            "top5": "indicator_positive_moves_top5_per_bucket_2003_2015_eurjpy.csv",
        },
    }
    out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(f"saved={all_csv}")
    print(f"saved={top5_csv}")
    print(f"saved={out_json}")
    if not top5.empty:
        print("\nTOP5 PER BUCKET")
        print(top5.to_string(index=False))


if __name__ == "__main__":
    main()

