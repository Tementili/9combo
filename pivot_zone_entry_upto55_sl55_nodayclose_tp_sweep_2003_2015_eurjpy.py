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
ENTRY_MAX_NORM = 0.55
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

    full = m.groupby("dkey").size()
    full_days = set(full[full == 96].index)
    m = m[m["dkey"].isin(full_days)].copy().reset_index(drop=True)

    day_idx = m.groupby("dkey").indices
    bounds = {k: (int(v[0]), int(v[-1])) for k, v in day_idx.items()}

    info = {}
    for _, r in d.iterrows():
        k = r["dkey"]
        if k in bounds and START_DATE <= k <= END_DATE:
            info[k] = {
                "pp": float(r["pp"]),
                "r2": float(r["r2"]),
                "s2": float(r["s2"]),
                "i0": bounds[k][0],
                "i1": bounds[k][1],
            }
    return m, info


def pips(side: str, entry_px: float, exit_px: float) -> float:
    if side == "BUY":
        return ((exit_px - entry_px) / PIP) - SPREAD_PIPS
    return ((entry_px - exit_px) / PIP) - SPREAD_PIPS


def build_entries(m: pd.DataFrame, info: dict, tp_norm: float):
    opens = m["Open"].to_numpy(float)
    highs = m["High"].to_numpy(float)
    lows = m["Low"].to_numpy(float)
    min_side = MIN_SIDE_RANGE_PIPS * PIP

    pendings: list[Pending] = []
    st = {"days": 0, "entries": 0, "buys": 0, "sells": 0, "no_entry_days": 0, "ambiguous_days": 0, "blocked_under55_before_entry": 0}

    for d in sorted(info.keys()):
        st["days"] += 1
        pp = info[d]["pp"]
        up = info[d]["r2"] - pp
        dn = pp - info[d]["s2"]
        if up < min_side or dn < min_side:
            st["no_entry_days"] += 1
            continue

        i0, i1 = info[d]["i0"], info[d]["i1"]
        buy_22 = pp - ENTRY_MIN_NORM * dn
        buy_55 = pp - ENTRY_MAX_NORM * dn
        sell_22 = pp + ENTRY_MIN_NORM * up
        sell_55 = pp + ENTRY_MAX_NORM * up
        buy_tp = pp + tp_norm * dn
        sell_tp = pp - tp_norm * up

        side = None
        trig = None
        blocked = False

        o0 = opens[i0]
        if buy_55 <= o0 <= buy_22:
            side = "BUY"
            trig = i0
        elif sell_22 <= o0 <= sell_55:
            side = "SELL"
            trig = i0
        else:
            for i in range(i0, i1 + 1):
                buy_beyond = lows[i] < buy_55
                sell_beyond = highs[i] > sell_55
                buy_in_zone = (lows[i] <= buy_22) and (highs[i] >= buy_55) and (not buy_beyond)
                sell_in_zone = (lows[i] <= sell_55) and (highs[i] >= sell_22) and (not sell_beyond)

                if buy_beyond and not sell_in_zone:
                    blocked = True
                if sell_beyond and not buy_in_zone:
                    blocked = True
                if buy_in_zone and sell_in_zone:
                    st["ambiguous_days"] += 1
                    side = None
                    trig = None
                    break
                if buy_in_zone:
                    side = "BUY"
                    trig = i
                    break
                if sell_in_zone:
                    side = "SELL"
                    trig = i
                    break

        if side is None or trig is None:
            st["no_entry_days"] += 1
            if blocked:
                st["blocked_under55_before_entry"] += 1
            continue

        entry_idx = trig + 1
        if entry_idx >= len(opens):
            st["no_entry_days"] += 1
            continue
        entry_px = opens[entry_idx]
        if side == "BUY":
            pendings.append(Pending(side, entry_idx, entry_px, buy_55, buy_tp))
            st["buys"] += 1
        else:
            pendings.append(Pending(side, entry_idx, entry_px, sell_55, sell_tp))
            st["sells"] += 1
        st["entries"] += 1

    return pendings, st


def simulate(m: pd.DataFrame, pendings: list[Pending], st: dict):
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

            pp = pips(p.side, p.entry_px, px)
            total_pips += pp
            if is_win:
                wins += 1
            else:
                losses += 1
        active = keep

    end_px = closes[-1]
    for p in active:
        pp = pips(p.side, p.entry_px, end_px)
        total_pips += pp
        end_exits += 1
        if pp >= 0:
            wins += 1
        else:
            losses += 1

    trades = wins + losses
    return {
        "days": st["days"],
        "entries": st["entries"],
        "buys": st["buys"],
        "sells": st["sells"],
        "no_entry_days": st["no_entry_days"],
        "ambiguous_days": st["ambiguous_days"],
        "blocked_under55_before_entry": st["blocked_under55_before_entry"],
        "wins": wins,
        "losses": losses,
        "dataset_end_exits": end_exits,
        "win_rate": round(wins / max(trades, 1), 6),
        "total_pips": round(total_pips, 4),
        "cal_ppd": round(total_pips / max(st["days"], 1), 4),
        "avg_pips_per_trade": round(total_pips / max(trades, 1), 4),
    }


def main():
    m, info = load_data()
    rows = []
    for tp in TP_GRID:
        pendings, st = build_entries(m, info, tp)
        r = simulate(m, pendings, st)
        r["tp_norm"] = tp
        rows.append(r)
        print(f"tp={tp:>4.2f} cal_ppd={r['cal_ppd']:+.4f} wr={r['win_rate']:.4f} trades={r['wins']+r['losses']}")

    df = pd.DataFrame(rows).sort_values("tp_norm").reset_index(drop=True)
    best = df.sort_values("cal_ppd", ascending=False).iloc[0].to_dict()
    pos = df[df["cal_ppd"] > 0]
    min_pos = float(pos["tp_norm"].min()) if not pos.empty else None
    max_pos = float(pos["tp_norm"].max()) if not pos.empty else None

    out = {
        "pair": "EURJPY",
        "period": [str(START_DATE.date()), str(END_DATE.date())],
        "entry_zone_norm": [ENTRY_MIN_NORM, ENTRY_MAX_NORM],
        "sl_norm": SL_NORM,
        "tp_grid": TP_GRID,
        "no_day_end_close": True,
        "best_by_cal_ppd": best,
        "min_profitable_tp_norm": min_pos,
        "max_profitable_tp_norm": max_pos,
        "all_results": df.to_dict(orient="records"),
    }

    out_json = OUT_DIR / "pivot_zone_entry_upto55_sl55_nodayclose_tp_sweep_2003_2015_eurjpy.json"
    out_csv = OUT_DIR / "pivot_zone_entry_upto55_sl55_nodayclose_tp_sweep_2003_2015_eurjpy.csv"
    out_json.write_text(json.dumps(out, indent=2), encoding="utf-8")
    df.to_csv(out_csv, index=False)

    print("\nBEST")
    print(best)
    print(f"min_profitable_tp_norm={min_pos}")
    print(f"max_profitable_tp_norm={max_pos}")
    print(f"saved={out_json}")
    print(f"saved={out_csv}")


if __name__ == "__main__":
    main()

