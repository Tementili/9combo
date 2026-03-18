from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(r"C:\Users\temen\5 joulu\TrailingStopLoss pivot peak trading 6 march 2026")
M15_PATH = ROOT / "forex raw data" / "Dukascopy" / "weekly_monthly_analysis" / "2003_20260306_15m" / "eurjpy" / "derived_15m" / "eurjpy_15m_20030804_20260306.csv"
DAILY_PATH = ROOT / "pair_workspaces" / "eurjpy_2003_20260306_15m_v1" / "eurjpy_daily_ohlcv_ver3.csv"
OUT_DIR = Path(__file__).parent

PIP = 0.01
SPREAD_PIPS = 1.0
ATR_PERIOD = 14


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


def close_pips(side: str, entry: float, px: float) -> float:
    if side == "BUY":
        return ((px - entry) / PIP) - SPREAD_PIPS
    return ((entry - px) / PIP) - SPREAD_PIPS


def load_data():
    d = pd.read_csv(DAILY_PATH, usecols=["Date", "High", "Low", "Close", "Open"], low_memory=False)
    d["Date"] = pd.to_datetime(d["Date"], errors="coerce")
    for c in ["Open", "High", "Low", "Close"]:
        d[c] = pd.to_numeric(d[c], errors="coerce")
    d = d.dropna().sort_values("Date").reset_index(drop=True)

    atr = atr_wilder(d["High"].to_numpy(float), d["Low"].to_numpy(float), d["Close"].to_numpy(float), ATR_PERIOD)
    d["atr14"] = atr
    d["atr_prev"] = d["atr14"].shift(1)
    d["yh"] = d["High"].shift(1)
    d["yl"] = d["Low"].shift(1)
    d["yc"] = d["Close"].shift(1)
    d["pp"] = (d["yh"] + d["yl"] + d["yc"]) / 3.0
    d["range_prev"] = d["yh"] - d["yl"]
    d = d.dropna(subset=["atr_prev", "pp", "range_prev"]).copy()
    d["dkey"] = d["Date"].dt.normalize()

    m = pd.read_csv(M15_PATH, usecols=["Datetime", "Open", "High", "Low", "Close"], low_memory=False)
    m["Datetime"] = pd.to_datetime(m["Datetime"], errors="coerce")
    for c in ["Open", "High", "Low", "Close"]:
        m[c] = pd.to_numeric(m[c], errors="coerce")
    m = m.dropna().sort_values("Datetime").reset_index(drop=True)
    m["dkey"] = m["Datetime"].dt.normalize()
    return d, m


def main():
    d, m = load_data()
    # Average daily ATR in pips (target threshold).
    avg_atr_pips = float(np.nanmean(d["atr14"].to_numpy(float)) / PIP)

    # Daily parameters by date.
    day_info = {}
    for _, r in d.iterrows():
        day_info[r["dkey"]] = {
            "pp": float(r["pp"]),
            "yh": float(r["yh"]),
            "yl": float(r["yl"]),
            "range_prev": float(r["range_prev"]),
        }

    # First bar index per day
    first = m.groupby("dkey").head(1).reset_index()[["dkey", "index"]]
    day_to_i0 = {r["dkey"]: int(r["index"]) for _, r in first.iterrows()}

    highs = m["High"].to_numpy(float)
    lows = m["Low"].to_numpy(float)
    closes = m["Close"].to_numpy(float)
    opens = m["Open"].to_numpy(float)
    dkeys = m["dkey"].to_numpy()

    realized = 0.0
    primary_tp_hits = 0
    basket_target_hits = 0
    campaigns_started = 0
    campaigns_closed = 0
    ladder_trades_opened = 0

    active = None  # single active campaign at a time

    for i in range(len(m)):
        dkey = dkeys[i]
        params = day_info.get(dkey)
        if params is None:
            continue

        # Start new campaign at day open only if none active.
        if active is None and day_to_i0.get(dkey, -1) == i:
            o = opens[i]
            pp = params["pp"]
            rng = params["range_prev"]
            yh = params["yh"]
            yl = params["yl"]
            if rng > 0:
                campaigns_started += 1
                if o < pp:
                    # Under pivot: primary BUY with original TP at YH.
                    active = {
                        "mode": "buy_primary",
                        "range": rng,
                        "realized": 0.0,
                        "primary": {"side": "BUY", "entry": o, "tp": yh, "open": True},
                        "ladders": [],
                        "next_ladder_trigger": yl,  # open SELL when touches YL
                    }
                elif o > pp:
                    # Above pivot: mirror.
                    active = {
                        "mode": "sell_primary",
                        "range": rng,
                        "realized": 0.0,
                        "primary": {"side": "SELL", "entry": o, "tp": yl, "open": True},
                        "ladders": [],
                        "next_ladder_trigger": yh,  # open BUY when touches YH
                    }

        if active is None:
            continue

        hi = highs[i]
        lo = lows[i]
        cl = closes[i]
        rng = active["range"]

        # Primary TP only (no forced SL per your instruction set).
        p = active["primary"]
        if p["open"]:
            if p["side"] == "BUY" and hi >= p["tp"]:
                pips = close_pips("BUY", p["entry"], p["tp"])
                active["realized"] += pips
                realized += pips
                p["open"] = False
                primary_tp_hits += 1
            elif p["side"] == "SELL" and lo <= p["tp"]:
                pips = close_pips("SELL", p["entry"], p["tp"])
                active["realized"] += pips
                realized += pips
                p["open"] = False
                primary_tp_hits += 1

        # Ladder logic
        if active["mode"] == "buy_primary":
            # Open SELL ladder each time price drops by one previous-day range.
            while lo <= active["next_ladder_trigger"]:
                entry = active["next_ladder_trigger"]
                tp = entry - rng
                active["ladders"].append({"side": "SELL", "entry": entry, "tp": tp, "open": True})
                ladder_trades_opened += 1
                active["next_ladder_trigger"] = entry - rng
        else:
            # Mirror: open BUY ladder each time price rises by one range.
            while hi >= active["next_ladder_trigger"]:
                entry = active["next_ladder_trigger"]
                tp = entry + rng
                active["ladders"].append({"side": "BUY", "entry": entry, "tp": tp, "open": True})
                ladder_trades_opened += 1
                active["next_ladder_trigger"] = entry + rng

        # Close ladder TPs.
        for leg in active["ladders"]:
            if not leg["open"]:
                continue
            if leg["side"] == "SELL" and lo <= leg["tp"]:
                pips = close_pips("SELL", leg["entry"], leg["tp"])
                active["realized"] += pips
                realized += pips
                leg["open"] = False
            elif leg["side"] == "BUY" and hi >= leg["tp"]:
                pips = close_pips("BUY", leg["entry"], leg["tp"])
                active["realized"] += pips
                realized += pips
                leg["open"] = False

        # Basket target: realized + floating >= one average daily ATR.
        floating = 0.0
        if p["open"]:
            floating += close_pips(p["side"], p["entry"], cl)
        for leg in active["ladders"]:
            if leg["open"]:
                floating += close_pips(leg["side"], leg["entry"], cl)
        if active["realized"] + floating >= avg_atr_pips:
            # Close all open legs at current close and end campaign.
            if p["open"]:
                pips = close_pips(p["side"], p["entry"], cl)
                realized += pips
                p["open"] = False
            for leg in active["ladders"]:
                if leg["open"]:
                    pips = close_pips(leg["side"], leg["entry"], cl)
                    realized += pips
                    leg["open"] = False
            basket_target_hits += 1
            campaigns_closed += 1
            active = None

    # No forced close at dataset end.
    open_positions_end = 0
    unrealized_end = 0.0
    if active is not None:
        cl = closes[-1]
        p = active["primary"]
        if p["open"]:
            open_positions_end += 1
            unrealized_end += close_pips(p["side"], p["entry"], cl)
        for leg in active["ladders"]:
            if leg["open"]:
                open_positions_end += 1
                unrealized_end += close_pips(leg["side"], leg["entry"], cl)

    cal_days = len(set(dkeys))
    out = {
        "pair": "EURJPY",
        "rule": "Open BUY under pivot / SELL above pivot at day open; ladder opposite side every 1x yesterday range; ladder TP at +1x range; close basket when realized+floating >= average ATR pips",
        "forced_close": "none",
        "avg_daily_atr_target_pips": round(avg_atr_pips, 4),
        "days": cal_days,
        "campaigns_started": campaigns_started,
        "campaigns_closed_on_target": campaigns_closed,
        "primary_tp_hits": primary_tp_hits,
        "ladder_trades_opened": ladder_trades_opened,
        "realized_total_pips": round(realized, 4),
        "realized_cal_ppd": round(realized / max(cal_days, 1), 4),
        "open_positions_at_end": open_positions_end,
        "unrealized_pips_at_last_close": round(unrealized_end, 4),
    }
    out_json = OUT_DIR / "pivot_open_primary_with_range_ladder_target_atr_eurjpy.json"
    out_json.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=2))
    print(f"saved={out_json}")


if __name__ == "__main__":
    main()

