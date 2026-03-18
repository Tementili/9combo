from __future__ import annotations
import json
from pathlib import Path

import numpy as np
import pandas as pd

DAILY_PATH = Path(r"C:\Users\temen\5 joulu\TrailingStopLoss pivot peak trading 6 march 2026\pair_workspaces\eurjpy_2003_20260306_15m_v1\eurjpy_daily_ohlcv_ver3.csv")
M15_PATH = Path(r"C:\Users\temen\5 joulu\TrailingStopLoss pivot peak trading 6 march 2026\forex raw data\Dukascopy\weekly_monthly_analysis\2003_20260306_15m\eurjpy\derived_15m\eurjpy_15m_20030804_20260306.csv")
OUT_DIR = Path(__file__).parent

PIP = 0.01
COST_PIPS = 1.0
SL_ATR_MULT = 0.70
TP_MULT = 1.3
N_RUNS = 20
SEED = 42


def atr_wilder(h, l, c, period=14):
    n = len(h)
    tr = np.empty(n)
    tr[0] = h[0] - l[0]
    for i in range(1, n):
        tr[i] = max(h[i] - l[i], abs(h[i] - c[i - 1]), abs(l[i] - c[i - 1]))
    atr = np.full(n, np.nan)
    if n >= period:
        atr[period - 1] = np.mean(tr[:period])
        k = 1.0 / period
        for i in range(period, n):
            atr[i] = atr[i - 1] * (1 - k) + tr[i] * k
    return atr


def run():
    daily = pd.read_csv(DAILY_PATH)
    daily["Date"] = pd.to_datetime(daily["Date"])
    for c in ["Open", "High", "Low", "Close"]:
        daily[c] = pd.to_numeric(daily[c], errors="coerce")
    daily = daily.dropna().sort_values("Date").reset_index(drop=True)

    h = daily["High"].values
    l = daily["Low"].values
    c = daily["Close"].values
    atr = atr_wilder(h, l, c, 14)

    daily["ATR14"] = atr
    daily["YH"] = daily["High"].shift(1)
    daily["YL"] = daily["Low"].shift(1)
    daily["YC"] = daily["Close"].shift(1)
    daily["PP"] = (daily["YH"] + daily["YL"] + daily["YC"]) / 3.0
    daily["S1"] = 2 * daily["PP"] - daily["YH"]
    daily["atr_prev"] = pd.Series(atr).shift(1)
    daily["date_key"] = daily["Date"].dt.date

    m15 = pd.read_csv(M15_PATH)
    m15["Datetime"] = pd.to_datetime(m15["Datetime"])
    for c in ["Open", "High", "Low", "Close"]:
        m15[c] = pd.to_numeric(m15[c], errors="coerce")
    m15 = m15.dropna().sort_values("Datetime").reset_index(drop=True)
    m15["date_key"] = m15["Datetime"].dt.date

    m15_highs = m15["High"].values.astype(float)
    m15_lows = m15["Low"].values.astype(float)
    m15_closes = m15["Close"].values.astype(float)
    m15_dates = m15["date_key"].values

    day_first_bar = {}
    day_last_bar = {}
    for i, dk in enumerate(m15_dates):
        if dk not in day_first_bar:
            day_first_bar[dk] = i
        day_last_bar[dk] = i

    trade_days = []
    for idx in range(1, len(daily)):
        row = daily.iloc[idx]
        dk = row["date_key"]
        pp = row["PP"]
        s1 = row["S1"]
        atr_val = row["atr_prev"]
        if not np.isfinite(pp) or not np.isfinite(s1) or not np.isfinite(atr_val):
            continue
        if atr_val <= 0 or dk not in day_first_bar:
            continue

        sl_dist = SL_ATR_MULT * atr_val
        tp_dist = sl_dist * TP_MULT
        i_start = day_first_bar[dk]
        i_end = day_last_bar[dk]

        entry_bar = None
        for j in range(i_start, i_end + 1):
            if m15_lows[j] <= pp:
                entry_bar = j
                break
        if entry_bar is None:
            continue

        trade_days.append({
            "dk": dk,
            "entry_bar": entry_bar,
            "entry_px": pp,
            "sl_dist": sl_dist,
            "tp_dist": tp_dist,
            "atr_val": atr_val,
            "pp": pp,
            "s1": s1,
        })

    total_bars = len(m15_highs)
    n_trade_days = len(trade_days)
    rng = np.random.default_rng(SEED)

    all_run_results = []

    for run_id in range(N_RUNS):
        sides = rng.choice(["BUY", "SELL"], size=n_trade_days)
        run_pips = 0.0
        run_wins = 0
        run_losses = 0
        run_tp = 0
        run_sl = 0

        for t_idx, td in enumerate(trade_days):
            side = sides[t_idx]
            entry_px = td["entry_px"]
            sl_dist = td["sl_dist"]
            tp_dist = td["tp_dist"]
            entry_bar = td["entry_bar"]

            if side == "BUY":
                sl_px = entry_px - sl_dist
                tp_px = entry_px + tp_dist
            else:
                sl_px = entry_px + sl_dist
                tp_px = entry_px - tp_dist

            exit_px = None
            exit_reason = None

            for j in range(entry_bar + 1, total_bars):
                bh = m15_highs[j]
                bl = m15_lows[j]

                if side == "BUY":
                    hit_sl = bl <= sl_px
                    hit_tp = bh >= tp_px
                else:
                    hit_sl = bh >= sl_px
                    hit_tp = bl <= tp_px

                if hit_sl and hit_tp:
                    exit_px = sl_px
                    exit_reason = "BOTH_SL"
                    break
                elif hit_sl:
                    exit_px = sl_px
                    exit_reason = "SL"
                    break
                elif hit_tp:
                    exit_px = tp_px
                    exit_reason = "TP"
                    break

            if exit_px is None:
                exit_px = float(m15_closes[total_bars - 1])
                exit_reason = "DATASET_END"

            if side == "BUY":
                raw = (exit_px - entry_px) / PIP
            else:
                raw = (entry_px - exit_px) / PIP
            net = raw - COST_PIPS
            run_pips += net

            if net > 0:
                run_wins += 1
            else:
                run_losses += 1
            if exit_reason == "TP":
                run_tp += 1
            elif exit_reason in ("SL", "BOTH_SL"):
                run_sl += 1

        n_total = run_wins + run_losses
        first_date = str(trade_days[0]["dk"])
        last_date = str(trade_days[-1]["dk"])
        cal_days = (pd.Timestamp(last_date) - pd.Timestamp(first_date)).days + 1

        all_run_results.append({
            "run": run_id,
            "trades": n_total,
            "wins": run_wins,
            "losses": run_losses,
            "win_rate": round(run_wins / n_total, 4) if n_total > 0 else 0,
            "total_pips": round(run_pips, 2),
            "cal_ppd": round(run_pips / cal_days, 4) if cal_days > 0 else 0,
            "tp_exits": run_tp,
            "sl_exits": run_sl,
        })

    df_runs = pd.DataFrame(all_run_results)
    avg_pips = df_runs["total_pips"].mean()
    avg_ppd = df_runs["cal_ppd"].mean()
    avg_wr = df_runs["win_rate"].mean()
    best = df_runs.loc[df_runs["total_pips"].idxmax()]
    worst = df_runs.loc[df_runs["total_pips"].idxmin()]

    summary = {
        "pair": "EURJPY",
        "strategy": "RANDOM BUY/SELL at PP (price touches PP-S1 zone), 15m bars",
        "sl_rule": f"fixed {SL_ATR_MULT}*ATR14",
        "tp_rule": f"fixed {SL_ATR_MULT}*{TP_MULT}*ATR14 (={SL_ATR_MULT * TP_MULT:.3f}*ATR14)",
        "cost_pips": COST_PIPS,
        "n_runs": N_RUNS,
        "trade_days_per_run": n_trade_days,
        "avg_total_pips": round(avg_pips, 2),
        "avg_cal_ppd": round(avg_ppd, 4),
        "avg_win_rate": round(avg_wr, 4),
        "best_run_pips": round(float(best["total_pips"]), 2),
        "worst_run_pips": round(float(worst["total_pips"]), 2),
        "std_pips": round(float(df_runs["total_pips"].std()), 2),
        "all_runs": all_run_results,
    }

    out_csv = OUT_DIR / "random_side_pp_s1_fixed_sl70_tp91_15m_eurjpy_runs.csv"
    out_json = OUT_DIR / "random_side_pp_s1_fixed_sl70_tp91_15m_eurjpy_summary.json"
    df_runs.to_csv(out_csv, index=False)
    out_json.write_text(json.dumps(summary, indent=2))

    print("\n=== RANDOM BUY/SELL at PP | Fixed SL=0.70*ATR14 | Fixed TP=0.91*ATR14 | 15m EURJPY ===")
    print(f"Trade opportunities: {n_trade_days} days  |  {N_RUNS} random runs")
    print(f"\nPer-run results:")
    for _, r in df_runs.iterrows():
        print(f"  Run {int(r['run']):2d}: pips={r['total_pips']:+10.2f}  cal_ppd={r['cal_ppd']:+.4f}  wr={r['win_rate']:.2%}  TP={int(r['tp_exits'])}  SL={int(r['sl_exits'])}")
    print(f"\nSummary across {N_RUNS} runs:")
    print(f"  Avg total pips:  {avg_pips:+.2f}")
    print(f"  Avg cal_ppd:     {avg_ppd:+.4f}")
    print(f"  Avg win rate:    {avg_wr:.2%}")
    print(f"  Best run:        {float(best['total_pips']):+.2f} pips")
    print(f"  Worst run:       {float(worst['total_pips']):+.2f} pips")
    print(f"  Std dev:         {float(df_runs['total_pips'].std()):.2f} pips")
    print(f"\nBUY-only baseline (prev test): -47,745 pips / cal_ppd -5.7964")
    print(f"\nSaved: {out_csv}")
    print(f"Saved: {out_json}")


if __name__ == "__main__":
    run()
