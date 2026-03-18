from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(r"C:\Users\temen\5 joulu\TrailingStopLoss pivot peak trading 6 march 2026")
M15_PATH = ROOT / "forex raw data" / "Dukascopy" / "weekly_monthly_analysis" / "2003_20260306_15m" / "eurjpy" / "derived_15m" / "eurjpy_15m_20030804_20260306.csv"
DAILY_PATH = ROOT / "pair_workspaces" / "eurjpy_2003_20260306_15m_v1" / "eurjpy_daily_ohlcv_ver3.csv"
OUT_DIR = Path(__file__).parent

PAIR = "eurjpy"
PIP = 0.01
SPREAD_PIPS = 1.0
ATR_PERIOD = 14
TP_ATR = 1.0
SL_ATR = 1.0 / 1.27
N_SIM = 200


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


def load_data():
    d = pd.read_csv(DAILY_PATH, usecols=["Date", "High", "Low", "Close"], low_memory=False)
    d["Date"] = pd.to_datetime(d["Date"], errors="coerce")
    for c in ["High", "Low", "Close"]:
        d[c] = pd.to_numeric(d[c], errors="coerce")
    d = d.dropna().sort_values("Date").reset_index(drop=True)
    atr = atr_wilder(d["High"].to_numpy(float), d["Low"].to_numpy(float), d["Close"].to_numpy(float), ATR_PERIOD)
    d["atr_prev"] = pd.Series(atr).shift(1)
    d = d.dropna(subset=["atr_prev"]).copy()
    d["dkey"] = d["Date"].dt.normalize()

    m = pd.read_csv(M15_PATH, usecols=["Datetime", "Open", "High", "Low", "Close"], low_memory=False)
    m["Datetime"] = pd.to_datetime(m["Datetime"], errors="coerce")
    for c in ["Open", "High", "Low", "Close"]:
        m[c] = pd.to_numeric(m[c], errors="coerce")
    m = m.dropna().sort_values("Datetime").reset_index(drop=True)
    m["dkey"] = m["Datetime"].dt.normalize()
    full_days = m.groupby("dkey").size()
    full_days = set(full_days[full_days == 96].index)
    m = m[m["dkey"].isin(full_days)].copy()

    day_bars = {}
    for dkey, g in m.groupby("dkey", sort=True):
        day_bars[dkey] = {
            "o": float(g["Open"].iloc[0]),
            "h": g["High"].to_numpy(float),
            "l": g["Low"].to_numpy(float),
            "c_last": float(g["Close"].iloc[-1]),
        }

    daily_map = {}
    for _, r in d.iterrows():
        k = r["dkey"]
        if k in day_bars:
            daily_map[k] = float(r["atr_prev"])
    days = sorted(daily_map.keys())
    return days, daily_map, day_bars


def run_one(seed: int, days, atr_map, day_bars):
    rng = np.random.default_rng(seed)
    total_pips = 0.0
    wins = 0
    losses = 0
    time_exits = 0

    for d in days:
        atr = atr_map[d]
        bars = day_bars[d]
        entry = bars["o"]
        tp_dist = TP_ATR * atr
        sl_dist = SL_ATR * atr
        side = "BUY" if rng.random() < 0.5 else "SELL"

        if side == "BUY":
            tp = entry + tp_dist
            sl = entry - sl_dist
        else:
            tp = entry - tp_dist
            sl = entry + sl_dist

        done = False
        for h, l in zip(bars["h"], bars["l"]):
            if side == "BUY":
                hit_sl = l <= sl
                hit_tp = h >= tp
            else:
                hit_sl = h >= sl
                hit_tp = l <= tp

            if hit_sl and hit_tp:
                # conservative intrabar resolve
                px = sl
                done = True
                losses += 1
            elif hit_sl:
                px = sl
                done = True
                losses += 1
            elif hit_tp:
                px = tp
                done = True
                wins += 1
            if done:
                if side == "BUY":
                    pips = ((px - entry) / PIP) - SPREAD_PIPS
                else:
                    pips = ((entry - px) / PIP) - SPREAD_PIPS
                total_pips += pips
                break

        if not done:
            px = bars["c_last"]
            time_exits += 1
            if side == "BUY":
                pips = ((px - entry) / PIP) - SPREAD_PIPS
            else:
                pips = ((entry - px) / PIP) - SPREAD_PIPS
            total_pips += pips

    cal_days = len(days)
    return {
        "seed": seed,
        "days": cal_days,
        "cal_ppd": total_pips / max(cal_days, 1),
        "total_pips": total_pips,
        "wins": wins,
        "losses": losses,
        "time_exits": time_exits,
    }


def main():
    days, atr_map, day_bars = load_data()
    sims = [run_one(s, days, atr_map, day_bars) for s in range(N_SIM)]
    df = pd.DataFrame(sims)

    out = {
        "pair": PAIR.upper(),
        "period": [str(days[0].date()), str(days[-1].date())],
        "rule": "random BUY/SELL at day open; TP=1.0*ATR(prev day); SL=(1/1.27)*ATR(prev day)",
        "spread_pips": SPREAD_PIPS,
        "simulations": N_SIM,
        "summary": {
            "mean_cal_ppd": round(float(df["cal_ppd"].mean()), 4),
            "median_cal_ppd": round(float(df["cal_ppd"].median()), 4),
            "p10_cal_ppd": round(float(df["cal_ppd"].quantile(0.1)), 4),
            "p90_cal_ppd": round(float(df["cal_ppd"].quantile(0.9)), 4),
            "min_cal_ppd": round(float(df["cal_ppd"].min()), 4),
            "max_cal_ppd": round(float(df["cal_ppd"].max()), 4),
        },
        "top5": df.sort_values("cal_ppd", ascending=False).head(5).to_dict(orient="records"),
        "bottom5": df.sort_values("cal_ppd", ascending=True).head(5).to_dict(orient="records"),
    }

    out_json = OUT_DIR / "random_dayopen_direction_atr_tp1_sl1over127_eurjpy_full_days.json"
    out_csv = OUT_DIR / "random_dayopen_direction_atr_tp1_sl1over127_eurjpy_full_days.csv"
    out_json.write_text(json.dumps(out, indent=2), encoding="utf-8")
    df.to_csv(out_csv, index=False)

    print(json.dumps(out["summary"], indent=2))
    print(f"saved={out_json}")
    print(f"saved={out_csv}")


if __name__ == "__main__":
    main()

