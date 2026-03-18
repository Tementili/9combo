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

    keep = d[
        [
            "date",
            "atr_bucket",
            "rsi_band",
            "wr_band",
            "bbz_band",
        ]
    ].copy()
    return keep


def main():
    trades = pd.read_csv(TRADES_PATH)
    trades["date"] = pd.to_datetime(trades["date"], errors="coerce").dt.normalize()
    trades = trades[(trades["date"] >= START_DATE) & (trades["date"] <= END_DATE)].copy()

    # Winning buckets from prior distance-only analysis (stable winners).
    keep_buckets = {
        ("SELL", "10-15%"),
        ("SELL", "15-20%"),
        ("SELL", "20-25%"),
        ("SELL", "30-35%"),
    }
    trades = trades[trades.apply(lambda r: (r["side"], r["bucket"]) in keep_buckets, axis=1)].copy()

    daily = build_daily_indicators()
    x = trades.merge(daily, on="date", how="left")
    x = x.dropna(subset=["atr_bucket", "rsi_band", "wr_band", "bbz_band"]).copy()

    # Baseline per winning bucket.
    base = (
        x.groupby(["side", "bucket"], observed=False)
        .agg(
            trades=("pips_at_first_event", "size"),
            base_avg_pips=("pips_at_first_event", "mean"),
            base_pivot_first=("pivot_hit_before_sl", "mean"),
        )
        .reset_index()
    )

    indicator_cols = ["atr_bucket", "rsi_band", "wr_band", "bbz_band"]
    lifts = []
    for ind in indicator_cols:
        g = (
            x.groupby(["side", "bucket", ind], observed=False)
            .agg(
                grp_trades=("pips_at_first_event", "size"),
                avg_pips=("pips_at_first_event", "mean"),
                pivot_first_rate=("pivot_hit_before_sl", "mean"),
            )
            .reset_index()
            .rename(columns={ind: "indicator_bucket"})
        )
        g = g[g["grp_trades"] >= MIN_GROUP_TRADES].copy()
        if g.empty:
            continue
        g = g.merge(base, on=["side", "bucket"], how="left")
        g["indicator"] = ind
        g["lift_pips"] = g["avg_pips"] - g["base_avg_pips"]
        g["lift_pivot_first"] = g["pivot_first_rate"] - g["base_pivot_first"]
        lifts.append(g)

    if lifts:
        lift_df = pd.concat(lifts, ignore_index=True)
    else:
        lift_df = pd.DataFrame(
            columns=[
                "side",
                "bucket",
                "indicator",
                "indicator_bucket",
                "grp_trades",
                "avg_pips",
                "pivot_first_rate",
                "base_avg_pips",
                "base_pivot_first",
                "lift_pips",
                "lift_pivot_first",
            ]
        )

    top_lift = lift_df.sort_values("lift_pips", ascending=False).head(30).copy()

    out = {
        "pair": "EURJPY",
        "period": [str(START_DATE.date()), str(END_DATE.date())],
        "scope": "winning distance buckets only",
        "winning_buckets": [{"side": s, "bucket": b} for s, b in sorted(keep_buckets)],
        "min_group_trades": MIN_GROUP_TRADES,
        "base_bucket_stats": base.to_dict(orient="records"),
        "top_indicator_lifts": top_lift.to_dict(orient="records"),
    }

    out_json = OUT_DIR / "winning_buckets_indicator_lift_2003_2015_eurjpy.json"
    out_csv = OUT_DIR / "winning_buckets_indicator_lift_2003_2015_eurjpy.csv"
    out_json.write_text(json.dumps(out, indent=2), encoding="utf-8")
    lift_df.sort_values(["lift_pips", "grp_trades"], ascending=[False, False]).to_csv(out_csv, index=False)

    print("BASE BUCKETS")
    print(base.to_string(index=False))
    print("\nTOP INDICATOR LIFTS")
    print(top_lift[["side", "bucket", "indicator", "indicator_bucket", "grp_trades", "avg_pips", "lift_pips"]].to_string(index=False))
    print(f"saved={out_json}")
    print(f"saved={out_csv}")


if __name__ == "__main__":
    main()

