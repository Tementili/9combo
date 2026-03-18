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


def atr_wilder(h, l, c, period=2):
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
    atr = atr_wilder(h, l, c, 2)  # ATR2 now

    daily["ATR2"] = atr
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

    ledger = []
    total_bars = len(m15_highs)

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

        entry_px = pp
        sl_px = entry_px + sl_dist
        tp_px = entry_px - tp_dist

        exit_px = None
        exit_reason = None
        exit_bar = None

        for j in range(entry_bar + 1, total_bars):
            bh = m15_highs[j]
            bl = m15_lows[j]

            hit_sl = bh >= sl_px
            hit_tp = bl <= tp_px

            if hit_sl and hit_tp:
                exit_bar = j
                exit_px = sl_px
                exit_reason = "BOTH_SL"
                break
            elif hit_sl:
                exit_bar = j
                exit_px = sl_px
                exit_reason = "SL"
                break
            elif hit_tp:
                exit_bar = j
                exit_px = tp_px
                exit_reason = "TP"
                break

        if exit_px is None:
            exit_bar = total_bars - 1
            exit_px = float(m15_closes[exit_bar])
            exit_reason = "DATASET_END"

        raw_pips = (entry_px - exit_px) / PIP
        net_pips = raw_pips - COST_PIPS
        hold_bars = exit_bar - entry_bar
        hold_hours = hold_bars * 0.25

        ledger.append({
            "date": str(dk),
            "entry_px": round(entry_px, 5),
            "sl_px": round(sl_px, 5),
            "tp_px": round(tp_px, 5),
            "exit_px": round(exit_px, 5),
            "exit_reason": exit_reason,
            "sl_dist_pips": round(sl_dist / PIP, 2),
            "tp_dist_pips": round(tp_dist / PIP, 2),
            "atr_pips": round(atr_val / PIP, 2),
            "net_pips": round(net_pips, 4),
            "hold_hours": round(hold_hours, 2),
        })

    if not ledger:
        print("No trades.")
        return

    df = pd.DataFrame(ledger)
    total_pips = df["net_pips"].sum()
    n_trades = len(df)
    wins = (df["net_pips"] > 0).sum()
    losses = (df["net_pips"] <= 0).sum()
    win_rate = wins / n_trades

    first_date = df["date"].min()
    last_date = df["date"].max()
    cal_days = (pd.Timestamp(last_date) - pd.Timestamp(first_date)).days + 1
    cal_ppd = total_pips / cal_days

    tp_exits = (df["exit_reason"] == "TP").sum()
    sl_exits = (df["exit_reason"].isin(["SL", "BOTH_SL"])).sum()
    dataset_end = (df["exit_reason"] == "DATASET_END").sum()
    avg_hold = df["hold_hours"].mean()
    avg_win = df.loc[df["net_pips"] > 0, "net_pips"].mean() if wins > 0 else 0
    avg_loss = df.loc[df["net_pips"] <= 0, "net_pips"].mean() if losses > 0 else 0

    df["year"] = pd.to_datetime(df["date"]).dt.year
    by_year = {}
    for yr, g in df.groupby("year"):
        yr_days = (pd.Timestamp(g["date"].max()) - pd.Timestamp(g["date"].min())).days + 1
        by_year[int(yr)] = {
            "trades": len(g),
            "total_pips": round(float(g["net_pips"].sum()), 2),
            "cal_ppd": round(float(g["net_pips"].sum()) / max(yr_days, 1), 4),
            "win_rate": round(float((g["net_pips"] > 0).sum() / len(g)), 4),
        }

    summary = {
        "pair": "EURJPY",
        "strategy": "SELL at PP when price drops to PP-S1 zone, 15m bars (ATR2 sizing)",
        "sl_rule": f"fixed {SL_ATR_MULT}*ATR2 ABOVE entry",
        "tp_rule": f"fixed {SL_ATR_MULT}*{TP_MULT}*ATR2 BELOW entry (={SL_ATR_MULT * TP_MULT:.3f}*ATR2)",
        "cost_pips": COST_PIPS,
        "period": [first_date, last_date],
        "cal_days": cal_days,
        "n_trades": n_trades,
        "wins": int(wins),
        "losses": int(losses),
        "win_rate": round(win_rate, 4),
        "total_pips": round(total_pips, 2),
        "cal_ppd": round(cal_ppd, 4),
        "avg_pips_per_trade": round(total_pips / n_trades, 4),
        "avg_win_pips": round(float(avg_win), 4),
        "avg_loss_pips": round(float(avg_loss), 4),
        "avg_hold_hours": round(float(avg_hold), 2),
        "exit_counts": {"TP": int(tp_exits), "SL": int(sl_exits), "DATASET_END": int(dataset_end)},
        "yearly": by_year,
    }

    ledger_path = OUT_DIR / "sell_pp_s1_fixed_sl70_tp91_atr2_15m_eurjpy_ledger.csv"
    summary_path = OUT_DIR / "sell_pp_s1_fixed_sl70_tp91_atr2_15m_eurjpy_summary.json"
    df.drop(columns=["year"]).to_csv(ledger_path, index=False)
    summary_path.write_text(json.dumps(summary, indent=2))

    print("\n=== SELL at PP (price drops to PP-S1) | Fixed SL=0.70*ATR2 | Fixed TP=0.91*ATR2 | 15m EURJPY ===")
    print(f"Period:   {first_date} .. {last_date}  ({cal_days} cal days)")
    print(f"Trades:   {n_trades}  (wins={wins} losses={losses})")
    print(f"Win rate: {win_rate:.2%}")
    print(f"Total:    {total_pips:+.2f} pips")
    print(f"Cal PPD:  {cal_ppd:+.4f}")
    print(f"Avg win:  {avg_win:+.2f} pips   Avg loss: {avg_loss:+.2f} pips")
    print(f"Avg hold: {avg_hold:.1f} hours")
    print(f"Exits:    TP={tp_exits}  SL={sl_exits}  DATASET_END={dataset_end}")
    print("\nYearly breakdown:")
    for yr in sorted(by_year):
        y = by_year[yr]
        print(f"  {yr}: trades={y['trades']:4d}  pips={y['total_pips']:+9.2f}  cal_ppd={y['cal_ppd']:+.4f}  wr={y['win_rate']:.2%}")
    print(f"\nComparison:")
    print(f"  BUY only:    -47,745 pips  cal_ppd=-5.7964  wr=38.96%")
    print(f"  Random side:    -675 pips  cal_ppd=-0.0819  wr=44.16%")
    print(f"  SELL only:   {total_pips:+,.0f} pips  cal_ppd={cal_ppd:+.4f}  wr={win_rate:.2%}")


if __name__ == "__main__":
    run()
 
