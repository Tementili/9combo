from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from bl_runner import PAIR_CFG, _load_daily, _load_hourly


PAIR = "usdjpy"
START_DATE = "2022-01-01"
END_DATE = "2023-12-31"
ATR_PERIOD = 14
ATR_MULT = 1.00
MAX_HOLD_HOURS = 432
OUTPUT_DIR = Path(__file__).parent


@dataclass
class Leg:
    tag: str
    side: str  # BUY/SELL
    entry: float
    tp: float
    sl: float
    open_bar: int
    is_open: bool = True
    exit_bar: Optional[int] = None
    exit_px: Optional[float] = None
    exit_reason: Optional[str] = None
    best_extreme: Optional[float] = None


def pip_size(pair: str) -> float:
    return 0.01 if pair.endswith("jpy") else 0.0001


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


def close_leg(leg: Leg, bar_idx: int, px: float, reason: str) -> None:
    leg.is_open = False
    leg.exit_bar = bar_idx
    leg.exit_px = float(px)
    leg.exit_reason = reason


def hit_check(leg: Leg, high: float, low: float) -> tuple[bool, Optional[str], Optional[float]]:
    if leg.side == "BUY":
        hit_sl = low <= leg.sl
        hit_tp = high >= leg.tp
    else:
        hit_sl = high >= leg.sl
        hit_tp = low <= leg.tp
    if hit_sl and hit_tp:
        return True, "BOTH_SL", leg.sl  # conservative
    if hit_sl:
        return True, "SL", leg.sl
    if hit_tp:
        return True, "TP", leg.tp
    return False, None, None


def pnl_pips(side: str, entry: float, exit_px: float, pip: float, cost_pips: float) -> float:
    raw = (exit_px - entry) / pip if side == "BUY" else (entry - exit_px) / pip
    return raw - cost_pips


def first_hourly_bar_index_per_day(times) -> dict:
    out = {}
    for i, ts in enumerate(times):
        d = pd.Timestamp(ts).date()
        if d not in out:
            out[d] = i
    return out


def run(atr_mult: float) -> dict:
    cfg = PAIR_CFG[PAIR]
    cost_pips = float(cfg["cost_pips"])
    pip = pip_size(PAIR)

    daily = _load_daily(Path(cfg["daily"]))
    times, opens, highs, lows, closes = _load_hourly(Path(cfg["h1"]))
    first_bar = first_hourly_bar_index_per_day(times)

    h = daily["High"].to_numpy(float)
    l = daily["Low"].to_numpy(float)
    c = daily["Close"].to_numpy(float)
    atr = atr_wilder(h, l, c, ATR_PERIOD)

    start = pd.Timestamp(START_DATE)
    end = pd.Timestamp(END_DATE)

    ledger = []
    traded_days = 0
    for i in range(1, len(daily)):
        d = pd.Timestamp(daily.loc[i, "Date"])
        if d < start or d > end:
            continue
        if d.date() not in first_bar:
            continue

        # Causal prior-day values
        yh = float(daily.loc[i - 1, "High"])
        yl = float(daily.loc[i - 1, "Low"])
        yc = float(daily.loc[i - 1, "Close"])
        atr_prev = float(atr[i - 1])
        if not np.isfinite(atr_prev) or atr_prev <= 0:
            continue

        pp = (yh + yl + yc) / 3.0
        dist = atr_mult * atr_prev
        if dist <= 0:
            continue
        atr_pips = atr_prev / pip

        day_start = first_bar[d.date()]
        day_end = min(day_start + MAX_HOLD_HOURS, len(highs))
        if day_end <= day_start:
            continue

        # Require pivot touch to activate entries (both legs at same pivot price).
        entry_bar = None
        for j in range(day_start, day_end):
            if float(lows[j]) <= pp <= float(highs[j]):
                entry_bar = j
                break
        if entry_bar is None:
            continue

        traded_days += 1
        buy = Leg(
            tag="BUY",
            side="BUY",
            entry=pp,
            tp=pp + dist,
            sl=pp - dist,
            open_bar=entry_bar,
            best_extreme=pp,
        )
        sell = Leg(
            tag="SELL",
            side="SELL",
            entry=pp,
            tp=pp - dist,
            sl=pp + dist,
            open_bar=entry_bar,
            best_extreme=pp,
        )
        legs = [buy, sell]

        for j in range(entry_bar, day_end):
            hj = float(highs[j])
            lj = float(lows[j])

            # Update trailing stops with same ATR distance.
            if buy.is_open:
                buy.best_extreme = max(float(buy.best_extreme), hj)
                trail_sl = float(buy.best_extreme) - dist
                buy.sl = max(buy.sl, trail_sl)
            if sell.is_open:
                sell.best_extreme = min(float(sell.best_extreme), lj)
                trail_sl = float(sell.best_extreme) + dist
                sell.sl = min(sell.sl, trail_sl)

            for leg in legs:
                if not leg.is_open:
                    continue
                hit, reason, px = hit_check(leg, hj, lj)
                if hit:
                    close_leg(leg, j, float(px), str(reason))

            if all(not lg.is_open for lg in legs):
                break

        last_close = float(closes[day_end - 1])
        for leg in legs:
            if leg.is_open:
                close_leg(leg, day_end - 1, last_close, "TIME_EXIT")

        for leg in legs:
            pips = pnl_pips(leg.side, leg.entry, float(leg.exit_px), pip, cost_pips)
            ledger.append(
                {
                    "pair": PAIR.upper(),
                    "date": str(d.date()),
                    "leg": leg.tag,
                    "entry": round(leg.entry, 6),
                    "exit": round(float(leg.exit_px), 6),
                    "reason": leg.exit_reason,
                    "pips": round(pips, 4),
                    "atr_pips": round(float(atr_pips), 6),
                    "dist_pips": round(float(dist / pip), 6),
                }
            )

    if not ledger:
        return {
            "pair": PAIR.upper(),
            "period": [START_DATE, END_DATE],
            "atr_mult": atr_mult,
            "traded_days": 0,
            "cal_ppd": 0.0,
            "buy_cal_ppd": 0.0,
            "sell_cal_ppd": 0.0,
        }

    df = pd.DataFrame(ledger)
    days = (pd.Timestamp(END_DATE) - pd.Timestamp(START_DATE)).days + 1
    total = float(df["pips"].sum())
    buy_total = float(df[df["leg"] == "BUY"]["pips"].sum())
    sell_total = float(df[df["leg"] == "SELL"]["pips"].sum())

    loss_df = df[df["pips"] < 0]
    buy_loss_df = df[(df["leg"] == "BUY") & (df["pips"] < 0)]
    sell_loss_df = df[(df["leg"] == "SELL") & (df["pips"] < 0)]
    loss_abs = loss_df["pips"].abs()
    loss_buy_abs = buy_loss_df["pips"].abs()
    loss_sell_abs = sell_loss_df["pips"].abs()

    summary = {
        "pair": PAIR.upper(),
        "period": [START_DATE, END_DATE],
        "atr_period": ATR_PERIOD,
        "atr_mult": atr_mult,
        "max_hold_hours": MAX_HOLD_HOURS,
        "traded_days": int(traded_days),
        "n_legs": int(len(df)),
        "cal_ppd": round(total / days, 4),
        "buy_cal_ppd": round(buy_total / days, 4),
        "sell_cal_ppd": round(sell_total / days, 4),
        "total_pips": round(total, 4),
        "buy_total_pips": round(buy_total, 4),
        "sell_total_pips": round(sell_total, 4),
        "avg_loss_all_legs": round(float(loss_df["pips"].mean()), 4) if len(loss_df) else 0.0,
        "avg_loss_buy_legs": round(float(buy_loss_df["pips"].mean()), 4) if len(buy_loss_df) else 0.0,
        "avg_loss_sell_legs": round(float(sell_loss_df["pips"].mean()), 4) if len(sell_loss_df) else 0.0,
        "avg_loss_abs_all_legs": round(float(loss_abs.mean()), 4) if len(loss_abs) else 0.0,
        "avg_loss_abs_buy_legs": round(float(loss_buy_abs.mean()), 4) if len(loss_buy_abs) else 0.0,
        "avg_loss_abs_sell_legs": round(float(loss_sell_abs.mean()), 4) if len(loss_sell_abs) else 0.0,
        "avg_atr_pips_losing_legs": round(float(loss_df["atr_pips"].mean()), 4) if len(loss_df) else 0.0,
        "avg_atr_pips_losing_buy_legs": round(float(buy_loss_df["atr_pips"].mean()), 4) if len(buy_loss_df) else 0.0,
        "avg_atr_pips_losing_sell_legs": round(float(sell_loss_df["atr_pips"].mean()), 4) if len(sell_loss_df) else 0.0,
        "avg_loss_pct_of_atr_all_legs": round(float((loss_abs / loss_df["atr_pips"]).mean() * 100.0), 2) if len(loss_df) else 0.0,
        "avg_loss_pct_of_atr_buy_legs": round(float((loss_buy_abs / buy_loss_df["atr_pips"]).mean() * 100.0), 2) if len(buy_loss_df) else 0.0,
        "avg_loss_pct_of_atr_sell_legs": round(float((loss_sell_abs / sell_loss_df["atr_pips"]).mean() * 100.0), 2) if len(sell_loss_df) else 0.0,
        "n_losing_legs": int(len(loss_df)),
        "exit_counts": df["reason"].value_counts().to_dict(),
    }

    mult_tag = str(atr_mult).replace(".", "p")
    ledger_path = OUTPUT_DIR / f"pivot_atr_straddle_ledger_{PAIR}_{mult_tag}_{START_DATE}_{END_DATE}.csv"
    summary_path = OUTPUT_DIR / f"pivot_atr_straddle_summary_{PAIR}_{mult_tag}_{START_DATE}_{END_DATE}.json"
    df.to_csv(ledger_path, index=False)
    summary_path.write_text(json.dumps(summary, indent=2))

    return summary


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--atr-mult", type=float, default=ATR_MULT)
    args = ap.parse_args()
    s = run(args.atr_mult)
    print("\nPivot ATR65 straddle (hard TP + trailing SL) results")
    print(f"pair={s['pair']} period={s['period'][0]}..{s['period'][1]}")
    print(f"traded_days={s['traded_days']} n_legs={s['n_legs']}")
    print(f"cal_ppd={s['cal_ppd']:+.4f} buy={s['buy_cal_ppd']:+.4f} sell={s['sell_cal_ppd']:+.4f}")
    print(f"total_pips={s['total_pips']:+.4f}")
    print(
        f"avg_loss_all={s['avg_loss_all_legs']:+.4f} "
        f"avg_loss_buy={s['avg_loss_buy_legs']:+.4f} "
        f"avg_loss_sell={s['avg_loss_sell_legs']:+.4f} "
        f"n_losing_legs={s['n_losing_legs']}"
    )
    print(
        f"avg_loss_pct_of_atr_all={s['avg_loss_pct_of_atr_all_legs']:.2f}% "
        f"buy={s['avg_loss_pct_of_atr_buy_legs']:.2f}% "
        f"sell={s['avg_loss_pct_of_atr_sell_legs']:.2f}%"
    )
    print(f"exit_counts={s['exit_counts']}")


if __name__ == "__main__":
    main()

