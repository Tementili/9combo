from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(r"C:\Users\temen\5 joulu\TrailingStopLoss pivot peak trading 6 march 2026")
PAIR_DEFAULT = "usdjpy"
START_DEFAULT = "2014-01-01"
END_DEFAULT = "2021-12-31"
ATR_PERIOD = 14
SPREAD_PIPS = 2.0
STEP_ATR_FRAC = 0.50   # add new buy each -50% ATR drop
TARGET_ATR_FRAC = 0.50 # close basket at +50% ATR net


@dataclass
class Pos:
    entry: float
    tp: float
    active_from_bar: int
    open: bool = True
    exit_px: float = np.nan
    exit_reason: str = ""


def pip_size(pair: str) -> float:
    return 0.01 if pair.endswith("jpy") else 0.0001


def pair_paths(pair: str) -> tuple[Path, Path]:
    p = pair.lower()
    daily = ROOT / "NEW_AGENT_HOURLY_33PIPS_PACK" / "pair_datasets" / p / f"processed_{p}_data_daily.csv"
    m15 = ROOT / "NEW_AGENT_HOURLY_33PIPS_PACK" / "pair_datasets" / p / f"processed_{p}_data_15m.csv"
    return daily, m15


def load_daily(path: Path) -> pd.DataFrame:
    d = pd.read_csv(path, low_memory=False)
    d["Date"] = pd.to_datetime(d["Date"], errors="coerce")
    for c in ["Open", "High", "Low", "Close"]:
        d[c] = pd.to_numeric(d[c], errors="coerce")
    d = d.dropna(subset=["Date", "Open", "High", "Low", "Close"]).sort_values("Date").reset_index(drop=True)
    return d


def load_m15(path: Path):
    m = pd.read_csv(path, usecols=["Datetime", "Open", "High", "Low", "Close"], low_memory=False)
    m["Datetime"] = pd.to_datetime(m["Datetime"], errors="coerce")
    for c in ["Open", "High", "Low", "Close"]:
        m[c] = pd.to_numeric(m[c], errors="coerce")
    m = m.dropna(subset=["Datetime", "Open", "High", "Low", "Close"]).sort_values("Datetime").reset_index(drop=True)
    return m


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


def close_buy_pips(entry: float, exit_px: float, pip: float) -> float:
    return ((exit_px - entry) / pip) - SPREAD_PIPS


def run(pair: str, start_date: str, end_date: str) -> dict:
    daily_path, m15_path = pair_paths(pair)
    daily = load_daily(daily_path)
    m15 = load_m15(m15_path)

    pip = pip_size(pair)
    atr = atr_wilder(
        daily["High"].to_numpy(float),
        daily["Low"].to_numpy(float),
        daily["Close"].to_numpy(float),
        ATR_PERIOD,
    )

    # Daily map with previous-day pivot + previous-day ATR14
    day_info: dict = {}
    for i in range(1, len(daily)):
        d = pd.Timestamp(daily.loc[i, "Date"]).date()
        atr_prev = float(atr[i - 1]) if np.isfinite(atr[i - 1]) else np.nan
        if not np.isfinite(atr_prev) or atr_prev <= 0:
            continue
        yh = float(daily.loc[i - 1, "High"])
        yl = float(daily.loc[i - 1, "Low"])
        yc = float(daily.loc[i - 1, "Close"])
        pp = (yh + yl + yc) / 3.0
        day_info[d] = {"pp": pp, "atr_prev": atr_prev}

    start = pd.Timestamp(start_date)
    end = pd.Timestamp(end_date)
    m15 = m15[(m15["Datetime"] >= start) & (m15["Datetime"] < (end + pd.Timedelta(days=1)))].copy()
    m15["dkey"] = m15["Datetime"].dt.date

    total_pips = 0.0
    total_closed = 0
    days_traded = 0
    days_target_hit = 0
    max_positions_open = 0

    grouped = m15.groupby("dkey", sort=True)
    for dkey, g in grouped:
        if dkey not in day_info:
            continue
        info = day_info[dkey]
        pp = float(info["pp"])
        atr_prev = float(info["atr_prev"])
        step = STEP_ATR_FRAC * atr_prev
        target_pips = (TARGET_ATR_FRAC * atr_prev) / pip
        if step <= 0 or target_pips <= 0:
            continue

        highs = g["High"].to_numpy(float)
        lows = g["Low"].to_numpy(float)
        closes = g["Close"].to_numpy(float)
        idxs = g.index.to_numpy()

        positions: list[Pos] = []
        started = False
        realized_pips = 0.0
        last_entry = np.nan

        for k in range(len(g)):
            h_ = highs[k]
            l_ = lows[k]
            c_ = closes[k]
            bar_index = int(idxs[k])

            # Start basket when market is under daily pivot.
            if not started and c_ < pp:
                e = c_
                positions.append(Pos(entry=e, tp=e + step, active_from_bar=bar_index + 1))
                started = True
                last_entry = e

            if not started:
                continue

            # Add ladder buys every additional -50% ATR drop from last entry.
            # Allow multi-level adds in one bar if move is large.
            while l_ <= (last_entry - step):
                new_entry = last_entry - step
                positions.append(Pos(entry=new_entry, tp=last_entry, active_from_bar=bar_index + 1))
                last_entry = new_entry

            # Close TP hits for active legs.
            for p in positions:
                if (not p.open) or bar_index < p.active_from_bar:
                    continue
                if h_ >= p.tp:
                    p.open = False
                    p.exit_px = p.tp
                    p.exit_reason = "TP"
                    pips = close_buy_pips(p.entry, p.tp, pip)
                    realized_pips += pips
                    total_pips += pips
                    total_closed += 1

            # Basket-level target check (realized + floating) >= +50% ATR.
            floating = 0.0
            open_count = 0
            for p in positions:
                if p.open:
                    floating += close_buy_pips(p.entry, c_, pip)
                    open_count += 1
            max_positions_open = max(max_positions_open, open_count)
            equity_pips = realized_pips + floating

            if equity_pips >= target_pips:
                for p in positions:
                    if p.open:
                        p.open = False
                        p.exit_px = c_
                        p.exit_reason = "BASKET_TARGET"
                        pips = close_buy_pips(p.entry, c_, pip)
                        realized_pips += pips
                        total_pips += pips
                        total_closed += 1
                days_target_hit += 1
                break

        if started:
            days_traded += 1
            # EOD force-close remaining open legs.
            last_c = closes[-1]
            for p in positions:
                if p.open:
                    p.open = False
                    p.exit_px = last_c
                    p.exit_reason = "EOD"
                    pips = close_buy_pips(p.entry, last_c, pip)
                    realized_pips += pips
                    total_pips += pips
                    total_closed += 1

    cal_days = int((pd.Timestamp(end_date) - pd.Timestamp(start_date)).days + 1)
    out = {
        "pair": pair.upper(),
        "period": [start_date, end_date],
        "timeframe": "15m",
        "mode": "buy_ladder_under_daily_pivot",
        "entry_rule": "start first buy when close < daily pivot",
        "add_rule": "add buy every additional 0.50*ATR drop from last entry",
        "tp_rule": "new buy TP = previous buy entry",
        "basket_target_rule": "close all when realized+floating >= +0.50*ATR (in pips)",
        "atr_source": "previous daily ATR14",
        "step_atr_frac": STEP_ATR_FRAC,
        "target_atr_frac": TARGET_ATR_FRAC,
        "spread_pips_per_leg": SPREAD_PIPS,
        "days_traded": days_traded,
        "days_target_hit": days_target_hit,
        "target_hit_rate_on_traded_days": round(days_target_hit / max(days_traded, 1), 4),
        "max_positions_open_same_time": int(max_positions_open),
        "total_legs_closed": int(total_closed),
        "total_pips": round(total_pips, 4),
        "cal_ppd": round(total_pips / max(cal_days, 1), 4),
    }
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pair", default=PAIR_DEFAULT, help="e.g. usdjpy, eurusd")
    ap.add_argument("--start-date", default=START_DEFAULT)
    ap.add_argument("--end-date", default=END_DEFAULT)
    args = ap.parse_args()

    out = run(args.pair.lower(), args.start_date, args.end_date)
    out_path = Path(__file__).parent / f"buy_ladder_under_pivot_target50atr_{args.pair.lower()}_{args.start_date}_{args.end_date}.json"
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=2))
    print(f"saved={out_path}")


if __name__ == "__main__":
    main()

