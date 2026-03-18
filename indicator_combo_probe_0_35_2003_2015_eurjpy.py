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
KEEP_BUCKETS = {"00-05%", "05-10%", "10-15%", "15-20%", "20-25%", "25-30%", "30-35%"}


def bin_quintiles(s: pd.Series) -> pd.Series:
    try:
        return pd.qcut(s, q=5, labels=["Q1", "Q2", "Q3", "Q4", "Q5"], duplicates="drop").astype("object")
    except Exception:
        return pd.Series([np.nan] * len(s), index=s.index, dtype="object")


def rsi_wilder(close: pd.Series, period: int) -> pd.Series:
    delta = close.diff()
    up = delta.clip(lower=0.0)
    down = -delta.clip(upper=0.0)
    roll_up = up.ewm(alpha=1.0 / period, adjust=False).mean()
    roll_down = down.ewm(alpha=1.0 / period, adjust=False).mean()
    rs = roll_up / roll_down.replace(0.0, float("nan"))
    return 100.0 - (100.0 / (1.0 + rs))


def build_daily_bins() -> pd.DataFrame:
    d = pd.read_csv(DAILY_PATH, usecols=["Date", "Open", "High", "Low", "Close"], low_memory=False)
    d["Date"] = pd.to_datetime(d["Date"], errors="coerce")
    for c in ["Open", "High", "Low", "Close"]:
        d[c] = pd.to_numeric(d[c], errors="coerce")
    d = d.dropna().sort_values("Date").reset_index(drop=True)
    d["date"] = d["Date"].dt.normalize()

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
    d["atr14"] = tr.rolling(14).mean()
    d["natr14"] = 100.0 * d["atr14"] / d["Close"]

    ema12 = d["Close"].ewm(span=12, adjust=False).mean()
    ema26 = d["Close"].ewm(span=26, adjust=False).mean()
    d["macd_hist"] = (ema12 - ema26) - (ema12 - ema26).ewm(span=9, adjust=False).mean()
    d["macd_signal"] = (ema12 - ema26).ewm(span=9, adjust=False).mean()

    tp = (d["High"] + d["Low"] + d["Close"]) / 3.0
    sma_tp = tp.rolling(20).mean()
    md = (tp - sma_tp).abs().rolling(20).mean()
    d["cci20"] = (tp - sma_tp) / (0.015 * md.replace(0.0, float("nan")))

    d["roc10"] = d["Close"].pct_change(10) * 100.0
    d["rsi14"] = rsi_wilder(d["Close"], 14)

    ln_hl = np.log((d["High"] / d["Low"]).replace(0.0, float("nan")))
    d["park_vol10"] = (ln_hl.pow(2).rolling(10).mean() / (4.0 * np.log(2.0))).pow(0.5)

    hh20 = d["High"].rolling(20).max()
    ll20 = d["Low"].rolling(20).min()
    d["donchian_width20"] = 100.0 * (hh20 - ll20) / d["Close"].replace(0.0, float("nan"))

    cols = ["atr14", "natr14", "macd_hist", "macd_signal", "cci20", "roc10", "rsi14", "park_vol10", "donchian_width20"]
    out = d[["date", "Open", "pp", "r2", "s2"]].copy()
    for c in cols:
        out[f"{c}_bin"] = bin_quintiles(d[c].shift(1))
    return out


def main():
    trades = pd.read_csv(TRADES_PATH)
    trades["date"] = pd.to_datetime(trades["date"], errors="coerce").dt.normalize()
    trades = trades[(trades["date"] >= START_DATE) & (trades["date"] <= END_DATE)].copy()
    trades = trades[(trades["side"] == "SELL") & (trades["bucket"].isin(KEEP_BUCKETS))].copy()

    daily = build_daily_bins()
    x = trades.merge(daily, on="date", how="left")

    # R multiple per trade.
    def risk_pips(row):
        entry = row["Open"]
        pp = row["pp"]
        denom = row["r2"] - pp
        sl = pp + SL_NORM * denom
        return max((sl - entry) / PIP + SPREAD_PIPS, 1e-9)

    x["risk_pips"] = x.apply(risk_pips, axis=1)
    x["r_multiple"] = x["pips_at_first_event"] / x["risk_pips"]

    # Candidate combos from robust top states.
    combos = [
        ("C1_macdh_Q5_15_20", (x["bucket"] == "15-20%") & (x["macd_hist_bin"] == "Q5")),
        ("C2_macdh_Q5_15_20_atr_Q5", (x["bucket"] == "15-20%") & (x["macd_hist_bin"] == "Q5") & (x["atr14_bin"] == "Q5")),
        ("C3_rsi14_Q4_25_30", (x["bucket"] == "25-30%") & (x["rsi14_bin"] == "Q4")),
        ("C4_rsi14_Q4_25_30_natr_Q5", (x["bucket"] == "25-30%") & (x["rsi14_bin"] == "Q4") & (x["natr14_bin"] == "Q5")),
        ("C5_parkvol_Q5_25_30", (x["bucket"] == "25-30%") & (x["park_vol10_bin"] == "Q5")),
        ("C6_combo_union_core", ((x["bucket"] == "15-20%") & (x["macd_hist_bin"] == "Q5")) | ((x["bucket"] == "25-30%") & (x["rsi14_bin"] == "Q4"))),
        ("C7_combo_union_strict", ((x["bucket"] == "15-20%") & (x["macd_hist_bin"] == "Q5") & (x["atr14_bin"] == "Q5")) | ((x["bucket"] == "25-30%") & (x["rsi14_bin"] == "Q4") & (x["natr14_bin"] == "Q5"))),
    ]

    rows = []
    for name, mask in combos:
        g = x[mask].copy()
        if g.empty:
            rows.append({"combo": name, "trades": 0})
            continue
        wins = (g["pips_at_first_event"] > 0).sum()
        losses = (g["pips_at_first_event"] <= 0).sum()
        avg_win = g.loc[g["pips_at_first_event"] > 0, "pips_at_first_event"].mean()
        avg_loss = g.loc[g["pips_at_first_event"] <= 0, "pips_at_first_event"].mean()
        sum_win = g.loc[g["pips_at_first_event"] > 0, "pips_at_first_event"].sum()
        sum_loss = abs(g.loc[g["pips_at_first_event"] <= 0, "pips_at_first_event"].sum())
        rows.append(
            {
                "combo": name,
                "trades": int(len(g)),
                "win_rate": float(wins / len(g)),
                "expectancy_pips": float(g["pips_at_first_event"].mean()),
                "avg_r": float(g["r_multiple"].mean()),
                "rr_ratio": float(abs(avg_win / avg_loss)) if pd.notna(avg_win) and pd.notna(avg_loss) and avg_loss != 0 else np.nan,
                "profit_factor": float(sum_win / sum_loss) if sum_loss > 0 else np.nan,
            }
        )

    out = pd.DataFrame(rows).sort_values("avg_r", ascending=False).reset_index(drop=True)
    out_csv = OUT_DIR / "indicator_combo_probe_0_35_2003_2015_eurjpy.csv"
    out_json = OUT_DIR / "indicator_combo_probe_0_35_2003_2015_eurjpy.json"
    out.to_csv(out_csv, index=False)
    out_json.write_text(
        json.dumps(
            {
                "pair": "EURJPY",
                "period": [str(START_DATE.date()), str(END_DATE.date())],
                "scope": "SELL side, buckets 0-35 only, hard SL55",
                "results": out.to_dict(orient="records"),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print(out.to_string(index=False))
    print(f"saved={out_csv}")
    print(f"saved={out_json}")


if __name__ == "__main__":
    main()

