from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

ROOT = Path(r"C:\Users\temen\5 joulu\TrailingStopLoss pivot peak trading 6 march 2026")
M15_PATH = ROOT / "forex raw data" / "Dukascopy" / "weekly_monthly_analysis" / "2003_20260306_15m" / "eurjpy" / "derived_15m" / "eurjpy_15m_20030804_20260306.csv"
DAILY_PATH = ROOT / "pair_workspaces" / "eurjpy_2003_20260306_15m_v1" / "eurjpy_daily_ohlcv_ver3.csv"
OUT_DIR = Path(__file__).parent

PIP = 0.01
SPREAD_PIPS = 1.0
START_DATE = pd.Timestamp("2003-01-01")
END_DATE = pd.Timestamp("2015-12-31")
ENTRY_MIN_NORM = 0.225
ENTRY_MAX_NORM = 0.50
SL_NORM = 0.55
TP_NORM = 0.5097
MIN_SIDE_RANGE_PIPS = 10.0


@dataclass
class PendingEntry:
    day: pd.Timestamp
    side: str
    entry_idx: int
    entry_px: float
    sl_px: float
    tp_px: float


@dataclass
class OpenPos:
    day: pd.Timestamp
    side: str
    entry_idx: int
    entry_px: float
    sl_px: float
    tp_px: float


def load_data():
    d = pd.read_csv(DAILY_PATH, usecols=["Date", "High", "Low", "Close"], low_memory=False)
    d["Date"] = pd.to_datetime(d["Date"], errors="coerce")
    for c in ["High", "Low", "Close"]:
        d[c] = pd.to_numeric(d[c], errors="coerce")
    d = d.dropna().sort_values("Date").reset_index(drop=True)
    d["yh"] = d["High"].shift(1)
    d["yl"] = d["Low"].shift(1)
    d["yc"] = d["Close"].shift(1)
    d["pp"] = (d["yh"] + d["yl"] + d["yc"]) / 3.0
    d["r2"] = d["pp"] + (d["yh"] - d["yl"])
    d["s2"] = d["pp"] - (d["yh"] - d["yl"])
    d = d.dropna(subset=["pp", "r2", "s2"]).copy()
    d["dkey"] = d["Date"].dt.normalize()

    m = pd.read_csv(M15_PATH, usecols=["Datetime", "Open", "High", "Low", "Close"], low_memory=False)
    m["Datetime"] = pd.to_datetime(m["Datetime"], errors="coerce")
    for c in ["Open", "High", "Low", "Close"]:
        m[c] = pd.to_numeric(m[c], errors="coerce")
    m = m.dropna().sort_values("Datetime").reset_index(drop=True)
    m["dkey"] = m["Datetime"].dt.normalize()

    # keep full days for clean day slicing
    full = m.groupby("dkey").size()
    full_days = set(full[full == 96].index)
    m = m[m["dkey"].isin(full_days)].copy().reset_index(drop=True)

    day_idx = m.groupby("dkey").indices
    day_bounds = {k: (int(v[0]), int(v[-1])) for k, v in day_idx.items()}

    day_info = {}
    for _, r in d.iterrows():
        k = r["dkey"]
        if k in day_bounds and START_DATE <= k <= END_DATE:
            day_info[k] = {"pp": float(r["pp"]), "r2": float(r["r2"]), "s2": float(r["s2"])}

    return m, day_bounds, day_info


def pips(side: str, entry_px: float, exit_px: float) -> float:
    if side == "BUY":
        return ((exit_px - entry_px) / PIP) - SPREAD_PIPS
    return ((entry_px - exit_px) / PIP) - SPREAD_PIPS


def find_entries(m: pd.DataFrame, day_bounds: dict, day_info: dict):
    min_side_px = MIN_SIDE_RANGE_PIPS * PIP
    opens = m["Open"].to_numpy(float)
    highs = m["High"].to_numpy(float)
    lows = m["Low"].to_numpy(float)

    pendings: list[PendingEntry] = []
    counters = {
        "days_in_period": 0,
        "no_entry_days": 0,
        "skipped_ambiguous_touch": 0,
        "entry_open_in_band": 0,
        "entry_touch_in_band": 0,
    }

    for d in sorted(day_info.keys()):
        counters["days_in_period"] += 1
        i0, i1 = day_bounds[d]
        pp = day_info[d]["pp"]
        up_rng = day_info[d]["r2"] - pp
        dn_rng = pp - day_info[d]["s2"]
        if up_rng < min_side_px or dn_rng < min_side_px:
            counters["no_entry_days"] += 1
            continue

        buy_lo = pp - ENTRY_MAX_NORM * dn_rng
        buy_hi = pp - ENTRY_MIN_NORM * dn_rng
        sell_lo = pp + ENTRY_MIN_NORM * up_rng
        sell_hi = pp + ENTRY_MAX_NORM * up_rng

        # SL/TP anchored to pivot geometry (entry-day fixed).
        buy_sl = pp - SL_NORM * dn_rng
        buy_tp = pp + TP_NORM * dn_rng
        sell_sl = pp + SL_NORM * up_rng
        sell_tp = pp - TP_NORM * up_rng

        side = None
        trig_idx = None
        entry_px = None

        o0 = opens[i0]
        if buy_lo <= o0 <= buy_hi:
            side = "BUY"
            trig_idx = i0
            entry_px = o0
            counters["entry_open_in_band"] += 1
        elif sell_lo <= o0 <= sell_hi:
            side = "SELL"
            trig_idx = i0
            entry_px = o0
            counters["entry_open_in_band"] += 1
        else:
            for j in range(i0, i1 + 1):
                hit_buy = (lows[j] <= buy_hi) and (highs[j] >= buy_lo)
                hit_sell = (lows[j] <= sell_hi) and (highs[j] >= sell_lo)
                if hit_buy and hit_sell:
                    counters["skipped_ambiguous_touch"] += 1
                    side = None
                    break
                if hit_buy:
                    side = "BUY"
                    trig_idx = j
                    # conservative fill in band: edge nearest pivot
                    entry_px = buy_hi
                    counters["entry_touch_in_band"] += 1
                    break
                if hit_sell:
                    side = "SELL"
                    trig_idx = j
                    # conservative fill in band: edge nearest pivot
                    entry_px = sell_lo
                    counters["entry_touch_in_band"] += 1
                    break

        if side is None:
            counters["no_entry_days"] += 1
            continue

        entry_idx = trig_idx + 1
        if entry_idx >= len(opens):
            counters["no_entry_days"] += 1
            continue

        if side == "BUY":
            pendings.append(PendingEntry(d, side, entry_idx, entry_px, buy_sl, buy_tp))
        else:
            pendings.append(PendingEntry(d, side, entry_idx, entry_px, sell_sl, sell_tp))

    return pendings, counters


def run(m: pd.DataFrame, pendings: list[PendingEntry], counters: dict):
    highs = m["High"].to_numpy(float)
    lows = m["Low"].to_numpy(float)
    closes = m["Close"].to_numpy(float)

    by_bar = {}
    for p in pendings:
        by_bar.setdefault(p.entry_idx, []).append(p)

    active: list[OpenPos] = []
    closed_rows = []
    wins = losses = 0
    buy_count = sell_count = 0
    total_pips = 0.0

    for i in range(len(highs)):
        for p in by_bar.get(i, []):
            active.append(OpenPos(p.day, p.side, p.entry_idx, p.entry_px, p.sl_px, p.tp_px))
            if p.side == "BUY":
                buy_count += 1
            else:
                sell_count += 1

        if not active:
            continue

        hi = highs[i]
        lo = lows[i]
        nxt = []
        for p in active:
            if p.side == "BUY":
                hit_sl = lo <= p.sl_px
                hit_tp = hi >= p.tp_px
            else:
                hit_sl = hi >= p.sl_px
                hit_tp = lo <= p.tp_px

            if hit_sl and hit_tp:
                exit_px = p.sl_px
                outcome = "SL_BOTH"
            elif hit_sl:
                exit_px = p.sl_px
                outcome = "SL"
            elif hit_tp:
                exit_px = p.tp_px
                outcome = "TP"
            else:
                nxt.append(p)
                continue

            pp = pips(p.side, p.entry_px, exit_px)
            total_pips += pp
            if outcome.startswith("TP"):
                wins += 1
            else:
                losses += 1
            closed_rows.append(
                {
                    "entry_day": p.day,
                    "side": p.side,
                    "entry_idx": p.entry_idx,
                    "exit_idx": i,
                    "outcome": outcome,
                    "pips": pp,
                }
            )
        active = nxt

    # no day-end close; close only at dataset end for unresolved positions
    dataset_end_exits = 0
    last_close = closes[-1]
    for p in active:
        pp = pips(p.side, p.entry_px, last_close)
        total_pips += pp
        dataset_end_exits += 1
        if pp >= 0:
            wins += 1
            outcome = "END_WIN"
        else:
            losses += 1
            outcome = "END_LOSS"
        closed_rows.append(
            {
                "entry_day": p.day,
                "side": p.side,
                "entry_idx": p.entry_idx,
                "exit_idx": len(highs) - 1,
                "outcome": outcome,
                "pips": pp,
            }
        )

    trades = wins + losses
    days = counters["days_in_period"]
    summary = {
        "pair": "EURJPY",
        "period": [str(START_DATE.date()), str(END_DATE.date())],
        "entry_band_norm": [ENTRY_MIN_NORM, ENTRY_MAX_NORM],
        "sl_norm": SL_NORM,
        "tp_norm": TP_NORM,
        "rule": "enter when price in 22.5%-50% away-from-pivot band; no day-end close; SL/TP hard from entry-day pivot geometry",
        "days": days,
        "entries": len(pendings),
        "buys": buy_count,
        "sells": sell_count,
        "wins": wins,
        "losses": losses,
        "dataset_end_exits": dataset_end_exits,
        "win_rate": round(wins / max(trades, 1), 6),
        "no_entry_days": counters["no_entry_days"],
        "skipped_ambiguous_touch": counters["skipped_ambiguous_touch"],
        "entry_open_in_band": counters["entry_open_in_band"],
        "entry_touch_in_band": counters["entry_touch_in_band"],
        "total_pips": round(total_pips, 4),
        "cal_ppd": round(total_pips / max(days, 1), 4),
        "avg_pips_per_trade": round(total_pips / max(trades, 1), 4),
    }
    return summary, closed_rows


def main():
    m, day_bounds, day_info = load_data()
    pendings, counters = find_entries(m, day_bounds, day_info)
    summary, rows = run(m, pendings, counters)

    out_json = OUT_DIR / "pivot_norm_entry_band0225_050_sl055_tp05097_nodayclose_2003_2015_eurjpy.json"
    out_csv = OUT_DIR / "pivot_norm_entry_band0225_050_sl055_tp05097_nodayclose_2003_2015_eurjpy_trades.csv"
    out_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    pd.DataFrame(rows).to_csv(out_csv, index=False)

    print("RESULT")
    print(json.dumps(summary, indent=2))
    print(f"saved={out_json}")
    print(f"saved={out_csv}")


if __name__ == "__main__":
    main()

