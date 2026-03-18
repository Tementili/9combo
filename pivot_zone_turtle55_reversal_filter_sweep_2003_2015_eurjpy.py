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
TP_NORM = 0.65
MIN_SIDE_RANGE_PIPS = 10.0
TURTLE_N = 55


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


def rsi_wilder(close: pd.Series, period: int) -> pd.Series:
    delta = close.diff()
    up = delta.clip(lower=0.0)
    down = -delta.clip(upper=0.0)
    roll_up = up.ewm(alpha=1.0 / period, adjust=False).mean()
    roll_down = down.ewm(alpha=1.0 / period, adjust=False).mean()
    rs = roll_up / roll_down.replace(0.0, float("nan"))
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return pd.to_numeric(rsi, errors="coerce")


def load_data():
    d = pd.read_csv(DAILY_PATH, usecols=["Date", "Open", "High", "Low", "Close"], low_memory=False)
    d["Date"] = pd.to_datetime(d["Date"], errors="coerce")
    for c in ["Open", "High", "Low", "Close"]:
        d[c] = pd.to_numeric(d[c], errors="coerce")
    d = d.dropna().sort_values("Date").reset_index(drop=True)
    d["dkey"] = d["Date"].dt.normalize()

    # Pivot ladder for current day decisions from previous day values.
    d["yh"] = d["High"].shift(1)
    d["yl"] = d["Low"].shift(1)
    d["yc"] = d["Close"].shift(1)
    d["pp"] = (d["yh"] + d["yl"] + d["yc"]) / 3.0
    d["r2"] = d["pp"] + (d["yh"] - d["yl"])
    d["s2"] = d["pp"] - (d["yh"] - d["yl"])

    # Turtle channel from prior N days only.
    d["turtle_hi"] = d["High"].shift(1).rolling(TURTLE_N).max()
    d["turtle_lo"] = d["Low"].shift(1).rolling(TURTLE_N).min()

    # Reversal indicators evaluated at day-open using info up to previous close.
    d["rsi2_sig"] = rsi_wilder(d["Close"], 2).shift(1)
    hh14 = d["High"].rolling(14).max()
    ll14 = d["Low"].rolling(14).min()
    d["wr14_raw"] = -100.0 * (hh14 - d["Close"]) / (hh14 - ll14).replace(0.0, pd.NA)
    d["wr14_sig"] = d["wr14_raw"].shift(1)
    sma20 = d["Close"].rolling(20).mean()
    std20 = d["Close"].rolling(20).std()
    d["bbz_raw"] = (d["Close"] - sma20) / std20.replace(0.0, pd.NA)
    d["bbz_sig"] = d["bbz_raw"].shift(1)

    m = pd.read_csv(M15_PATH, usecols=["Datetime", "Open", "High", "Low", "Close"], low_memory=False)
    m["Datetime"] = pd.to_datetime(m["Datetime"], errors="coerce")
    for c in ["Open", "High", "Low", "Close"]:
        m[c] = pd.to_numeric(m[c], errors="coerce")
    m = m.dropna().sort_values("Datetime").reset_index(drop=True)
    m["dkey"] = m["Datetime"].dt.normalize()
    full_days = set(m.groupby("dkey").size().pipe(lambda s: s[s == 96]).index)
    m = m[m["dkey"].isin(full_days)].copy().reset_index(drop=True)
    bounds = {k: (int(v[0]), int(v[-1])) for k, v in m.groupby("dkey").indices.items()}

    d = d.dropna(subset=["pp", "r2", "s2", "turtle_hi", "turtle_lo", "rsi2_sig", "wr14_sig", "bbz_sig"]).copy()

    info = {}
    for _, r in d.iterrows():
        k = r["dkey"]
        if k in bounds and START_DATE <= k <= END_DATE:
            i0, i1 = bounds[k]
            info[k] = {
                "pp": float(r["pp"]),
                "r2": float(r["r2"]),
                "s2": float(r["s2"]),
                "t_hi": float(r["turtle_hi"]),
                "t_lo": float(r["turtle_lo"]),
                "rsi2": float(r["rsi2_sig"]),
                "wr14": float(r["wr14_sig"]),
                "bbz": float(r["bbz_sig"]),
                "i0": i0,
                "i1": i1,
            }
    return m, info


def pips(side: str, entry_px: float, exit_px: float) -> float:
    if side == "BUY":
        return ((exit_px - entry_px) / PIP) - SPREAD_PIPS
    return ((entry_px - exit_px) / PIP) - SPREAD_PIPS


def passes_filter(side: str, v: dict, cfg: dict) -> bool:
    f = cfg["filter"]
    if f == "none":
        return True
    if f == "rsi2":
        x = cfg["x"]
        return (side == "BUY" and v["rsi2"] <= x) or (side == "SELL" and v["rsi2"] >= 100.0 - x)
    if f == "wr14":
        x = cfg["x"]
        return (side == "BUY" and v["wr14"] <= -100.0 + x) or (side == "SELL" and v["wr14"] >= -x)
    if f == "bbz":
        z = cfg["z"]
        return (side == "BUY" and v["bbz"] <= -z) or (side == "SELL" and v["bbz"] >= z)
    return False


def build_entries(m: pd.DataFrame, info: dict, cfg: dict):
    opens = m["Open"].to_numpy(float)
    highs = m["High"].to_numpy(float)
    lows = m["Low"].to_numpy(float)
    min_side = MIN_SIDE_RANGE_PIPS * PIP

    st = {
        "days": 0,
        "entries": 0,
        "buys": 0,
        "sells": 0,
        "no_entry_days": 0,
        "ambiguous_days": 0,
        "blocked_under55_before_entry": 0,
        "filtered_out_by_turtle": 0,
        "filtered_out_by_reversal": 0,
    }
    pendings: list[Pending] = []

    for d in sorted(info.keys()):
        st["days"] += 1
        v = info[d]
        pp = v["pp"]
        up = v["r2"] - pp
        dn = pp - v["s2"]
        if up < min_side or dn < min_side:
            st["no_entry_days"] += 1
            continue

        i0, i1 = v["i0"], v["i1"]
        buy_22 = pp - ENTRY_MIN_NORM * dn
        buy_55 = pp - ENTRY_MAX_NORM * dn
        sell_22 = pp + ENTRY_MIN_NORM * up
        sell_55 = pp + ENTRY_MAX_NORM * up
        buy_tp = pp + TP_NORM * dn
        sell_tp = pp - TP_NORM * up

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

        # Turtle filter.
        if v["t_hi"] <= v["t_lo"]:
            st["filtered_out_by_turtle"] += 1
            st["no_entry_days"] += 1
            continue
        t_mid = 0.5 * (v["t_hi"] + v["t_lo"])
        if (side == "BUY" and o0 > t_mid) or (side == "SELL" and o0 < t_mid):
            st["filtered_out_by_turtle"] += 1
            st["no_entry_days"] += 1
            continue

        # Reversal filter.
        if not passes_filter(side, v, cfg):
            st["filtered_out_by_reversal"] += 1
            st["no_entry_days"] += 1
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
        "filtered_out_by_turtle": st["filtered_out_by_turtle"],
        "filtered_out_by_reversal": st["filtered_out_by_reversal"],
        "wins": wins,
        "losses": losses,
        "dataset_end_exits": end_exits,
        "win_rate": round(wins / max(trades, 1), 6),
        "total_pips": round(total_pips, 4),
        "cal_ppd": round(total_pips / max(st["days"], 1), 4),
        "avg_pips_per_trade": round(total_pips / max(trades, 1), 4),
    }


def cfg_name(cfg: dict) -> str:
    f = cfg["filter"]
    if f == "none":
        return "turtle_only"
    if f in ("rsi2", "wr14"):
        return f"{f}_x{cfg['x']}"
    return f"{f}_z{cfg['z']}"


def main():
    m, info = load_data()
    configs = [{"filter": "none"}]
    for x in [2.0, 5.0, 8.0, 10.0]:
        configs.append({"filter": "rsi2", "x": x})
    for x in [2.0, 5.0, 10.0]:
        configs.append({"filter": "wr14", "x": x})
    for z in [1.5, 2.0, 2.5]:
        configs.append({"filter": "bbz", "z": z})

    rows = []
    for cfg in configs:
        pendings, st = build_entries(m, info, cfg)
        r = simulate(m, pendings, st)
        r["config"] = cfg_name(cfg)
        rows.append(r)
        print(f"{r['config']:<18} cal_ppd={r['cal_ppd']:+.4f} entries={r['entries']}")

    df = pd.DataFrame(rows).sort_values("cal_ppd", ascending=False).reset_index(drop=True)
    out = {
        "pair": "EURJPY",
        "period": [str(START_DATE.date()), str(END_DATE.date())],
        "base_rule": "zone 22.5..55, sl55, tp0.65, turtle55, no day-end close",
        "best": df.iloc[0].to_dict(),
        "all_results": df.to_dict(orient="records"),
    }
    out_json = OUT_DIR / "pivot_zone_turtle55_reversal_filter_sweep_2003_2015_eurjpy.json"
    out_csv = OUT_DIR / "pivot_zone_turtle55_reversal_filter_sweep_2003_2015_eurjpy.csv"
    out_json.write_text(json.dumps(out, indent=2), encoding="utf-8")
    df.to_csv(out_csv, index=False)
    print("\nBEST")
    print(df.iloc[0].to_dict())
    print(f"saved={out_json}")
    print(f"saved={out_csv}")


if __name__ == "__main__":
    main()

