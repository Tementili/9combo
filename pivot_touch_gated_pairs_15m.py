from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

PAIR = "usdjpy"
START_DATE = "2014-01-01"
END_DATE = "2021-12-31"
SL_MULT = 0.08
TP_MULT = 0.55
ATR_PERIOD = 14
MAX_HOLD_HOURS = 432
SPREAD_PIPS = 2.0

DATA_ROOT = Path(r"C:\Users\temen\5 joulu\TrailingStopLoss pivot peak trading 6 march 2026")
DAILY_PATH = DATA_ROOT / "NEW_AGENT_HOURLY_33PIPS_PACK" / "pair_datasets" / "usdjpy" / "processed_usdjpy_data_daily.csv"
M15_PATH = DATA_ROOT / "NEW_AGENT_HOURLY_33PIPS_PACK" / "pair_datasets" / "usdjpy" / "processed_usdjpy_data_15m.csv"
OUTPUT_DIR = Path(__file__).parent


@dataclass
class Pos:
    side: str  # BUY/SELL
    entry: float
    tp: float
    sl: float
    best_extreme: float
    open_bar: int
    active_from_bar: int
    pair_id: int
    is_initial_leg: bool
    open: bool = True
    exit_bar: int = -1
    exit_px: float = np.nan
    reason: str = ""


@dataclass
class PairState:
    pair_id: int
    init_total: int = 2
    init_closed: int = 0
    init_tp_hits: int = 0
    init_loss_hits: int = 0
    qualified: bool = False


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
    return (
        m["Datetime"].to_numpy(),
        m["Open"].to_numpy(float),
        m["High"].to_numpy(float),
        m["Low"].to_numpy(float),
        m["Close"].to_numpy(float),
    )


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


def new_buy(entry: float, sl_dist: float, tp_dist: float, bar: int, pair_id: int, is_initial_leg: bool) -> Pos:
    return Pos(
        side="BUY",
        entry=entry,
        tp=entry + tp_dist,
        sl=entry - sl_dist,
        best_extreme=entry,
        open_bar=bar,
        active_from_bar=bar,
        pair_id=pair_id,
        is_initial_leg=is_initial_leg,
    )


def new_sell(entry: float, sl_dist: float, tp_dist: float, bar: int, pair_id: int, is_initial_leg: bool) -> Pos:
    return Pos(
        side="SELL",
        entry=entry,
        tp=entry - tp_dist,
        sl=entry + sl_dist,
        best_extreme=entry,
        open_bar=bar,
        active_from_bar=bar,
        pair_id=pair_id,
        is_initial_leg=is_initial_leg,
    )


def close_pos(p: Pos, bar: int, px: float, reason: str) -> None:
    p.open = False
    p.exit_bar = bar
    p.exit_px = float(px)
    p.reason = reason


def update_pair_state(pair_states: dict[int, PairState], p: Pos) -> None:
    if not p.is_initial_leg:
        return
    st = pair_states[p.pair_id]
    st.init_closed += 1
    if p.reason == "TP":
        st.init_tp_hits += 1
    if p.reason in ("SL", "BOTH_SL"):
        st.init_loss_hits += 1

    # Qualification rule requested:
    # - at least one initial leg TP, OR
    # - both initial legs lost (SL/BOTH_SL)
    if st.init_tp_hits >= 1:
        st.qualified = True
    elif st.init_closed == st.init_total and st.init_loss_hits == st.init_total:
        st.qualified = True


def run() -> dict:
    daily = load_daily(DAILY_PATH)
    times, opens, highs, lows, closes = load_m15(M15_PATH)
    pip = pip_size(PAIR)
    atr = atr_wilder(
        daily["High"].to_numpy(float),
        daily["Low"].to_numpy(float),
        daily["Close"].to_numpy(float),
        ATR_PERIOD,
    )

    daily_map = {}
    for i in range(1, len(daily)):
        d = pd.Timestamp(daily.loc[i, "Date"]).date()
        atr_prev = float(atr[i - 1]) if np.isfinite(atr[i - 1]) else np.nan
        if not np.isfinite(atr_prev) or atr_prev <= 0:
            continue
        yh = float(daily.loc[i - 1, "High"])
        yl = float(daily.loc[i - 1, "Low"])
        yc = float(daily.loc[i - 1, "Close"])
        pp = (yh + yl + yc) / 3.0
        daily_map[d] = (pp, SL_MULT * atr_prev, TP_MULT * atr_prev)

    start = pd.Timestamp(START_DATE)
    end = pd.Timestamp(END_DATE)
    cal_days = (end - start).days + 1

    open_positions: list[Pos] = []
    all_closed: list[Pos] = []
    pair_states: dict[int, PairState] = {}
    pair_counter = 0
    last_opened_pair_id = -1
    gate_open = True  # first pair can open on first pivot touch in range

    exit_counts = {"SL": 0, "TP": 0, "BOTH_SL": 0, "TIME_EXIT": 0}
    pivot_touch_count = 0
    opened_pair_count = 0

    for j in range(len(times)):
        t = pd.Timestamp(times[j])
        if t < start or t > (end + pd.Timedelta(days=1)):
            continue
        day = t.date()
        if day not in daily_map:
            continue
        pp, sl_dist, tp_dist = daily_map[day]
        h_ = float(highs[j])
        l_ = float(lows[j])
        c_ = float(closes[j])

        # 1) Update all open positions on this bar.
        for p in open_positions:
            if not p.open or j < p.active_from_bar:
                continue

            # time-exit first if max horizon exceeded
            max_bars = MAX_HOLD_HOURS * 4
            if j - p.open_bar >= max_bars:
                close_pos(p, j, c_, "TIME_EXIT")
                if p.reason in exit_counts:
                    exit_counts[p.reason] += 1
                update_pair_state(pair_states, p)
                continue

            if p.side == "BUY":
                p.best_extreme = max(p.best_extreme, h_)
                p.sl = max(p.sl, p.best_extreme - sl_dist)
                hit_sl = l_ <= p.sl
                hit_tp = h_ >= p.tp
            else:
                p.best_extreme = min(p.best_extreme, l_)
                p.sl = min(p.sl, p.best_extreme + sl_dist)
                hit_sl = h_ >= p.sl
                hit_tp = l_ <= p.tp

            if hit_sl and hit_tp:
                close_pos(p, j, p.sl, "BOTH_SL")
            elif hit_sl:
                close_pos(p, j, p.sl, "SL")
            elif hit_tp:
                close_pos(p, j, p.tp, "TP")

            if not p.open:
                if p.reason in exit_counts:
                    exit_counts[p.reason] += 1
                update_pair_state(pair_states, p)

        # 2) If latest opened pair qualified, open gate for next pivot touch.
        if last_opened_pair_id >= 0 and pair_states[last_opened_pair_id].qualified:
            gate_open = True

        # 3) Pivot-touch trigger for opening a new pair.
        touched_pivot = l_ <= pp <= h_
        if touched_pivot:
            pivot_touch_count += 1
            if gate_open:
                pair_counter += 1
                pair_id = pair_counter
                pair_states[pair_id] = PairState(pair_id=pair_id)
                open_positions.append(new_buy(pp, sl_dist, tp_dist, j, pair_id, True))
                open_positions.append(new_sell(pp, sl_dist, tp_dist, j, pair_id, True))
                opened_pair_count += 1
                last_opened_pair_id = pair_id
                gate_open = False

        # Compact occasionally
        if j % 10000 == 0:
            still_open = []
            for p in open_positions:
                if p.open:
                    still_open.append(p)
                else:
                    all_closed.append(p)
            open_positions = still_open

    # Force-close leftovers at final close.
    final_close = float(closes[-1])
    final_bar = len(closes) - 1
    for p in open_positions:
        if p.open:
            close_pos(p, final_bar, final_close, "TIME_EXIT")
            exit_counts["TIME_EXIT"] += 1
            update_pair_state(pair_states, p)
        all_closed.append(p)

    total = 0.0
    total_buy = 0.0
    total_sell = 0.0
    for p in all_closed:
        if p.side == "BUY":
            pips = ((p.exit_px - p.entry) / pip) - SPREAD_PIPS
            total_buy += pips
        else:
            pips = ((p.entry - p.exit_px) / pip) - SPREAD_PIPS
            total_sell += pips
        total += pips

    qualified_pairs = sum(1 for s in pair_states.values() if s.qualified)
    out = {
        "pair": PAIR.upper(),
        "period": [START_DATE, END_DATE],
        "timeframe": "15m",
        "mode": "daily_pivot_touch_gated_pairs",
        "entry_anchor": "daily PP=(YH+YL+YC)/3 from previous day",
        "gate_rule": "new pair only after latest pair has >=1 TP on initial legs OR both initial legs lost",
        "sl_mult": SL_MULT,
        "tp_mult": TP_MULT,
        "max_hold_hours": MAX_HOLD_HOURS,
        "pivot_touches": pivot_touch_count,
        "opened_pairs": opened_pair_count,
        "qualified_pairs": qualified_pairs,
        "total_legs_closed": len(all_closed),
        "cal_ppd": round(total / max(cal_days, 1), 4),
        "buy_cal_ppd": round(total_buy / max(cal_days, 1), 4),
        "sell_cal_ppd": round(total_sell / max(cal_days, 1), 4),
        "total_pips": round(total, 4),
        "exit_counts": exit_counts,
    }
    out_path = OUTPUT_DIR / f"pivot_touch_gated_pairs_{PAIR}_{START_DATE}_{END_DATE}.json"
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=2))
    print(f"saved={out_path}")
    return out


if __name__ == "__main__":
    run()

