from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


DATA_ROOT = Path(r"C:\Users\temen\5 joulu\TrailingStopLoss pivot peak trading 6 march 2026")
PAIR_DATA = {
    "usdjpy": {
        "daily": DATA_ROOT / "NEW_AGENT_HOURLY_33PIPS_PACK" / "pair_datasets" / "usdjpy" / "processed_usdjpy_data_daily.csv",
        "m15": DATA_ROOT / "NEW_AGENT_HOURLY_33PIPS_PACK" / "pair_datasets" / "usdjpy" / "processed_usdjpy_data_15m.csv",
        "cost_pips": 2.0,
        "pip": 0.01,
    }
}
OUTPUT_DIR = Path(__file__).parent
WINDOW_STARTS = [0, 8, 16]
MAX_HOLD_HOURS = 432


def load_daily(path: Path) -> pd.DataFrame:
    d = pd.read_csv(path, low_memory=False)
    d["Date"] = pd.to_datetime(d["Date"], errors="coerce")
    for c in ["Open", "High", "Low", "Close"]:
        d[c] = pd.to_numeric(d[c], errors="coerce")
    return d.dropna(subset=["Date", "Open", "High", "Low", "Close"]).sort_values("Date").reset_index(drop=True)


def load_m15(path: Path):
    m = pd.read_csv(path, usecols=["Datetime", "Open", "High", "Low", "Close"], low_memory=False)
    m["Datetime"] = pd.to_datetime(m["Datetime"], errors="coerce")
    for c in ["Open", "High", "Low", "Close"]:
        m[c] = pd.to_numeric(m[c], errors="coerce")
    m = m.dropna(subset=["Datetime", "Open", "High", "Low", "Close"]).sort_values("Datetime").reset_index(drop=True)
    return (
        m["Datetime"].to_numpy(),
        m["Open"].to_numpy(float),
        m["High"].to_numpy(float),
        m["Low"].to_numpy(float),
        m["Close"].to_numpy(float),
    )


def atr_wilder(h: np.ndarray, l: np.ndarray, c: np.ndarray, period: int = 14) -> np.ndarray:
    n = len(h)
    tr = np.empty(n, dtype=float)
    tr[0] = h[0] - l[0]
    for i in range(1, n):
        tr[i] = max(h[i] - l[i], abs(h[i] - c[i - 1]), abs(l[i] - c[i - 1]))
    atr = np.full(n, np.nan)
    if n >= period:
        atr[period - 1] = np.mean(tr[:period])
        k = 1.0 / period
        for i in range(period, n):
            atr[i] = atr[i - 1] * (1.0 - k) + tr[i] * k
    return atr


def adx_di(h: np.ndarray, l: np.ndarray, c: np.ndarray, period: int = 14):
    n = len(h)
    plus_dm = np.zeros(n)
    minus_dm = np.zeros(n)
    tr = np.zeros(n)
    for i in range(1, n):
        up = h[i] - h[i - 1]
        down = l[i - 1] - l[i]
        plus_dm[i] = up if (up > down and up > 0) else 0.0
        minus_dm[i] = down if (down > up and down > 0) else 0.0
        tr[i] = max(h[i] - l[i], abs(h[i] - c[i - 1]), abs(l[i] - c[i - 1]))

    atr = np.zeros(n)
    psm = np.zeros(n)
    msm = np.zeros(n)
    if n > period:
        atr[period] = tr[1 : period + 1].sum()
        psm[period] = plus_dm[1 : period + 1].sum()
        msm[period] = minus_dm[1 : period + 1].sum()
    for i in range(period + 1, n):
        atr[i] = atr[i - 1] - (atr[i - 1] / period) + tr[i]
        psm[i] = psm[i - 1] - (psm[i - 1] / period) + plus_dm[i]
        msm[i] = msm[i - 1] - (msm[i - 1] / period) + minus_dm[i]

    pdi = np.full(n, np.nan)
    ndi = np.full(n, np.nan)
    valid = atr > 0
    pdi[valid] = 100.0 * (psm[valid] / atr[valid])
    ndi[valid] = 100.0 * (msm[valid] / atr[valid])
    dx = np.full(n, np.nan)
    denom = pdi + ndi
    dvalid = denom > 0
    dx[dvalid] = 100.0 * np.abs(pdi[dvalid] - ndi[dvalid]) / denom[dvalid]

    adx = np.full(n, np.nan)
    start = period * 2
    if n > start:
        adx[start] = np.nanmean(dx[period + 1 : start + 1])
        for i in range(start + 1, n):
            adx[i] = ((adx[i - 1] * (period - 1)) + dx[i]) / period
    return adx, pdi, ndi


def run_entries(pair: str, start_date: str, end_date: str, sl_mult: float, tp_mult: float) -> pd.DataFrame:
    info = PAIR_DATA[pair]
    daily = load_daily(info["daily"])
    times, opens, highs, lows, closes = load_m15(info["m15"])
    pip = info["pip"]
    cost = info["cost_pips"]

    h = daily["High"].to_numpy(float)
    l = daily["Low"].to_numpy(float)
    c = daily["Close"].to_numpy(float)
    atr = atr_wilder(h, l, c, 14)
    adx, pdi, ndi = adx_di(h, l, c, 14)

    start = pd.Timestamp(start_date)
    end = pd.Timestamp(end_date)
    rows = []

    for i in range(1, len(daily)):
        d = pd.Timestamp(daily.loc[i, "Date"])
        if d < start or d > end:
            continue

        atr_prev = float(atr[i - 1]) if np.isfinite(atr[i - 1]) else np.nan
        adx_prev = float(adx[i - 1]) if np.isfinite(adx[i - 1]) else np.nan
        pdi_prev = float(pdi[i - 1]) if np.isfinite(pdi[i - 1]) else np.nan
        ndi_prev = float(ndi[i - 1]) if np.isfinite(ndi[i - 1]) else np.nan
        if not np.isfinite(atr_prev) or atr_prev <= 0:
            continue

        yh = float(daily.loc[i - 1, "High"])
        yl = float(daily.loc[i - 1, "Low"])
        yc = float(daily.loc[i - 1, "Close"])
        pp = (yh + yl + yc) / 3.0
        yr = yh - yl
        sl_dist = sl_mult * atr_prev
        tp_dist = tp_mult * atr_prev

        for wh in WINDOW_STARTS:
            wstart = np.datetime64(d + pd.Timedelta(hours=wh))
            i0 = int(np.searchsorted(times, wstart, side="left"))
            if i0 >= len(times):
                continue
            if pd.Timestamp(times[i0]).date() != d.date():
                continue
            end_bar = min(i0 + MAX_HOLD_HOURS * 4, len(highs))
            if end_bar <= i0:
                continue

            # buy/sell straddle at pivot
            buy_open = True
            sell_open = True
            buy_entry = pp
            sell_entry = pp
            buy_tp, sell_tp = pp + tp_dist, pp - tp_dist
            buy_sl, sell_sl = pp - sl_dist, pp + sl_dist
            buy_best, sell_best = pp, pp
            buy_exit = sell_exit = None
            buy_reason = sell_reason = None

            for j in range(i0, end_bar):
                h_ = float(highs[j]); l_ = float(lows[j])
                if buy_open:
                    buy_best = max(buy_best, h_)
                    buy_sl = max(buy_sl, buy_best - sl_dist)
                if sell_open:
                    sell_best = min(sell_best, l_)
                    sell_sl = min(sell_sl, sell_best + sl_dist)

                if buy_open:
                    hit_sl = l_ <= buy_sl
                    hit_tp = h_ >= buy_tp
                    if hit_sl and hit_tp:
                        buy_open = False; buy_exit = buy_sl; buy_reason = "BOTH_SL"
                    elif hit_sl:
                        buy_open = False; buy_exit = buy_sl; buy_reason = "SL"
                    elif hit_tp:
                        buy_open = False; buy_exit = buy_tp; buy_reason = "TP"

                if sell_open:
                    hit_sl = h_ >= sell_sl
                    hit_tp = l_ <= sell_tp
                    if hit_sl and hit_tp:
                        sell_open = False; sell_exit = sell_sl; sell_reason = "BOTH_SL"
                    elif hit_sl:
                        sell_open = False; sell_exit = sell_sl; sell_reason = "SL"
                    elif hit_tp:
                        sell_open = False; sell_exit = sell_tp; sell_reason = "TP"

                if not buy_open and not sell_open:
                    break

            if buy_open:
                buy_exit = float(closes[end_bar - 1]); buy_reason = "TIME_EXIT"
            if sell_open:
                sell_exit = float(closes[end_bar - 1]); sell_reason = "TIME_EXIT"

            buy_pips = ((buy_exit - buy_entry) / pip) - cost
            sell_pips = ((sell_entry - sell_exit) / pip) - cost
            net = buy_pips + sell_pips
            rows.append(
                {
                    "date": str(d.date()),
                    "year": int(d.year),
                    "session": int(wh),
                    "atr_pips": atr_prev / pip,
                    "adx": adx_prev,
                    "pdi": pdi_prev,
                    "ndi": ndi_prev,
                    "di_spread": abs(pdi_prev - ndi_prev) if np.isfinite(pdi_prev) and np.isfinite(ndi_prev) else np.nan,
                    "yr_atr_ratio": (yr / atr_prev) if atr_prev > 0 else np.nan,
                    "net_pips": net,
                    "win": 1 if net > 0 else 0,
                }
            )
    return pd.DataFrame(rows)


def bucket_report(df: pd.DataFrame) -> dict:
    rep = {}
    if df.empty:
        return {"empty": True}

    # quantile ATR buckets
    df = df.copy()
    df["atr_bucket"] = pd.qcut(df["atr_pips"], q=4, labels=["Q1_low", "Q2", "Q3", "Q4_high"], duplicates="drop")
    df["adx_bucket"] = pd.cut(df["adx"], bins=[-1, 15, 20, 25, 30, 100], labels=["<15", "15-20", "20-25", "25-30", "30+"])
    df["di_bucket"] = pd.cut(df["di_spread"], bins=[-1, 5, 10, 15, 100], labels=["<5", "5-10", "10-15", "15+"])

    def summarize(group_cols: list[str]) -> list[dict]:
        g = (
            df.groupby(group_cols, dropna=False)
            .agg(
                n=("net_pips", "count"),
                avg_pips=("net_pips", "mean"),
                total_pips=("net_pips", "sum"),
                win_rate=("win", "mean"),
            )
            .reset_index()
            .sort_values("avg_pips", ascending=False)
        )
        g["avg_pips"] = g["avg_pips"].round(4)
        g["total_pips"] = g["total_pips"].round(4)
        g["win_rate"] = g["win_rate"].round(4)
        return g.to_dict(orient="records")

    rep["overall"] = {
        "n": int(len(df)),
        "avg_pips": round(float(df["net_pips"].mean()), 4),
        "total_pips": round(float(df["net_pips"].sum()), 4),
        "win_rate": round(float(df["win"].mean()), 4),
    }
    rep["by_session"] = summarize(["session"])
    rep["by_atr_bucket"] = summarize(["atr_bucket"])
    rep["by_adx_bucket"] = summarize(["adx_bucket"])
    rep["by_di_bucket"] = summarize(["di_bucket"])
    rep["by_session_atr"] = summarize(["session", "atr_bucket"])
    rep["by_year"] = summarize(["year"])
    return rep


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pair", default="usdjpy")
    ap.add_argument("--start-date", default="2014-01-01")
    ap.add_argument("--end-date", default="2021-12-31")
    ap.add_argument("--sl-mult", type=float, default=0.10)
    ap.add_argument("--tp-mult", type=float, default=0.50)
    args = ap.parse_args()

    df = run_entries(args.pair, args.start_date, args.end_date, args.sl_mult, args.tp_mult)
    rep = bucket_report(df)

    tag = f"{args.pair}_{args.start_date}_{args.end_date}_sl{args.sl_mult}_tp{args.tp_mult}"
    csv_path = OUTPUT_DIR / f"pivot_bucket_diag_rows_{tag}.csv"
    json_path = OUTPUT_DIR / f"pivot_bucket_diag_report_{tag}.json"
    if not df.empty:
        df.to_csv(csv_path, index=False)
    json_path.write_text(json.dumps(rep, indent=2, default=str))

    print(f"overall={rep.get('overall')}")
    if "by_session" in rep:
        print(f"top_session={rep['by_session'][0]}")
        print(f"top_atr_bucket={rep['by_atr_bucket'][0]}")
        print(f"top_adx_bucket={rep['by_adx_bucket'][0]}")
        print(f"top_di_bucket={rep['by_di_bucket'][0]}")
    print(f"saved_json={json_path}")
    if not df.empty:
        print(f"saved_rows={csv_path}")


if __name__ == "__main__":
    main()

