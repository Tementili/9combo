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
ENTRY_NORM = 0.225
SL_NORM = 0.55
TP_GRID = [round(x, 3) for x in [0.00, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80]]
MIN_SIDE_RANGE_PIPS = 10.0


@dataclass
class Pending:
    side: str
    entry_idx: int
    entry_px: float
    sl_px: float
    tp_px: float


@dataclass
class Pos:
    side: str
    entry_idx: int
    entry_px: float
    sl_px: float
    tp_px: float


def load():
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

    m = pd.read_csv(M15_PATH, usecols=["Datetime", "High", "Low", "Close"], low_memory=False)
    m["Datetime"] = pd.to_datetime(m["Datetime"], errors="coerce")
    for c in ["High", "Low", "Close"]:
        m[c] = pd.to_numeric(m[c], errors="coerce")
    m = m.dropna().sort_values("Datetime").reset_index(drop=True)
    m["dkey"] = m["Datetime"].dt.normalize()
    full_days = set(m.groupby("dkey").size().pipe(lambda s: s[s == 96]).index)
    m = m[m["dkey"].isin(full_days)].copy().reset_index(drop=True)

    day_idx = m.groupby("dkey").indices
    day_bounds = {k: (int(v[0]), int(v[-1])) for k, v in day_idx.items()}

    info = {}
    for _, r in d.iterrows():
        k = r["dkey"]
        if START_DATE <= k <= END_DATE and k in day_bounds:
            info[k] = {"pp": float(r["pp"]), "r2": float(r["r2"]), "s2": float(r["s2"]), "i0": day_bounds[k][0], "i1": day_bounds[k][1]}
    return m, info


def calc_pips(side: str, entry: float, exit_px: float) -> float:
    if side == "BUY":
        return ((exit_px - entry) / PIP) - SPREAD_PIPS
    return ((entry - exit_px) / PIP) - SPREAD_PIPS


def build_pendings(m: pd.DataFrame, info: dict, tp_norm: float):
    highs = m["High"].to_numpy(float)
    lows = m["Low"].to_numpy(float)
    min_side = MIN_SIDE_RANGE_PIPS * PIP

    pendings: list[Pending] = []
    stats = {"days": 0, "entries": 0, "no_entry_days": 0, "ambiguous": 0, "buys": 0, "sells": 0}
    for d in sorted(info.keys()):
        stats["days"] += 1
        pp = info[d]["pp"]
        r2 = info[d]["r2"]
        s2 = info[d]["s2"]
        dn = pp - s2
        up = r2 - pp
        if dn < min_side or up < min_side:
            stats["no_entry_days"] += 1
            continue

        buy_entry = pp - ENTRY_NORM * dn
        sell_entry = pp + ENTRY_NORM * up
        buy_sl = pp - SL_NORM * dn
        sell_sl = pp + SL_NORM * up
        buy_tp = pp + tp_norm * dn
        sell_tp = pp - tp_norm * up

        i0, i1 = info[d]["i0"], info[d]["i1"]
        side = None
        trig = None
        for i in range(i0, i1 + 1):
            hit_b = lows[i] <= buy_entry <= highs[i]
            hit_s = lows[i] <= sell_entry <= highs[i]
            if hit_b and hit_s:
                side = None
                stats["ambiguous"] += 1
                break
            if hit_b:
                side = "BUY"
                trig = i
                break
            if hit_s:
                side = "SELL"
                trig = i
                break

        if side is None or trig is None:
            stats["no_entry_days"] += 1
            continue

        entry_idx = trig + 1
        if entry_idx >= len(highs):
            stats["no_entry_days"] += 1
            continue

        if side == "BUY":
            pendings.append(Pending(side, entry_idx, buy_entry, buy_sl, buy_tp))
            stats["buys"] += 1
        else:
            pendings.append(Pending(side, entry_idx, sell_entry, sell_sl, sell_tp))
            stats["sells"] += 1
        stats["entries"] += 1

    return pendings, stats


def simulate(m: pd.DataFrame, pendings: list[Pending], stats: dict):
    highs = m["High"].to_numpy(float)
    lows = m["Low"].to_numpy(float)
    closes = m["Close"].to_numpy(float)

    by_idx = {}
    for p in pendings:
        by_idx.setdefault(p.entry_idx, []).append(p)

    active: list[Pos] = []
    wins = losses = 0
    total_pips = 0.0
    end_exits = 0

    for i in range(len(highs)):
        for p in by_idx.get(i, []):
            active.append(Pos(p.side, p.entry_idx, p.entry_px, p.sl_px, p.tp_px))

        if not active:
            continue

        hi = highs[i]
        lo = lows[i]
        keep = []
        for p in active:
            if p.side == "BUY":
                hit_sl = lo <= p.sl_px
                hit_tp = hi >= p.tp_px
            else:
                hit_sl = hi >= p.sl_px
                hit_tp = lo <= p.tp_px

            if hit_sl and hit_tp:
                px = p.sl_px
                is_win = False
            elif hit_sl:
                px = p.sl_px
                is_win = False
            elif hit_tp:
                px = p.tp_px
                is_win = True
            else:
                keep.append(p)
                continue

            pp = calc_pips(p.side, p.entry_px, px)
            total_pips += pp
            if is_win:
                wins += 1
            else:
                losses += 1
        active = keep

    # no day-end close: only close leftovers at final dataset bar
    end_px = closes[-1]
    for p in active:
        pp = calc_pips(p.side, p.entry_px, end_px)
        total_pips += pp
        end_exits += 1
        if pp >= 0:
            wins += 1
        else:
            losses += 1

    trades = wins + losses
    return {
        "days": stats["days"],
        "entries": stats["entries"],
        "buys": stats["buys"],
        "sells": stats["sells"],
        "no_entry_days": stats["no_entry_days"],
        "ambiguous_touch_days": stats["ambiguous"],
        "wins": wins,
        "losses": losses,
        "dataset_end_exits": end_exits,
        "win_rate": round(wins / max(trades, 1), 6),
        "total_pips": round(total_pips, 4),
        "cal_ppd": round(total_pips / max(stats["days"], 1), 4),
        "avg_pips_per_trade": round(total_pips / max(trades, 1), 4),
    }


def main():
    m, info = load()
    rows = []
    for tp in TP_GRID:
        pendings, stats = build_pendings(m, info, tp)
        r = simulate(m, pendings, stats)
        r["tp_norm"] = tp
        # from entry -0.225 to tp +tp_norm
        r["tp_distance_from_entry_norm"] = round(tp + ENTRY_NORM, 4)
        rows.append(r)
        print(f"tp={tp:>4.2f} cal_ppd={r['cal_ppd']:+.4f} wr={r['win_rate']:.4f} trades={r['wins']+r['losses']}")

    df = pd.DataFrame(rows).sort_values("tp_norm").reset_index(drop=True)
    pos = df[df["cal_ppd"] > 0].copy()
    if not pos.empty:
        min_profitable_tp = float(pos["tp_norm"].min())
        max_profitable_tp = float(pos["tp_norm"].max())
    else:
        min_profitable_tp = None
        max_profitable_tp = None

    best = df.sort_values("cal_ppd", ascending=False).iloc[0].to_dict()

    out = {
        "pair": "EURJPY",
        "period": [str(START_DATE.date()), str(END_DATE.date())],
        "entry_norm": ENTRY_NORM,
        "sl_norm": SL_NORM,
        "tp_grid": TP_GRID,
        "no_day_end_close": True,
        "best_by_cal_ppd": best,
        "min_profitable_tp_norm": min_profitable_tp,
        "max_profitable_tp_norm": max_profitable_tp,
        "all_results": df.to_dict(orient="records"),
    }

    out_json = OUT_DIR / "pivot_norm_entry0225_sl055_nodayclose_tp_sweep_2003_2015_eurjpy.json"
    out_csv = OUT_DIR / "pivot_norm_entry0225_sl055_nodayclose_tp_sweep_2003_2015_eurjpy.csv"
    out_json.write_text(json.dumps(out, indent=2), encoding="utf-8")
    df.to_csv(out_csv, index=False)

    print("\nBEST")
    print(best)
    print(f"min_profitable_tp_norm={min_profitable_tp}")
    print(f"max_profitable_tp_norm={max_profitable_tp}")
    print(f"saved={out_json}")
    print(f"saved={out_csv}")


if __name__ == "__main__":
    main()

