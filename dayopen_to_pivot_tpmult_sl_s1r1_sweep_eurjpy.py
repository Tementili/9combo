from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

ROOT = Path(r"C:\Users\temen\5 joulu\TrailingStopLoss pivot peak trading 6 march 2026")
M15_PATH = ROOT / "forex raw data" / "Dukascopy" / "weekly_monthly_analysis" / "2003_20260306_15m" / "eurjpy" / "derived_15m" / "eurjpy_15m_20030804_20260306.csv"
DAILY_PATH = ROOT / "pair_workspaces" / "eurjpy_2003_20260306_15m_v1" / "eurjpy_daily_ohlcv_ver3.csv"
OUT_DIR = Path(__file__).parent

PIP = 0.01
SPREAD_PIPS = 1.0
TP_MULT_GRID = [1.25, 1.30, 1.35, 1.40, 1.45, 1.50, 1.55, 1.60, 1.65, 1.70, 1.75, 1.80]


def load_data():
    d = pd.read_csv(DAILY_PATH, usecols=["Date", "High", "Low", "Close"], low_memory=False)
    d["Date"] = pd.to_datetime(d["Date"], errors="coerce")
    for c in ["High", "Low", "Close"]:
        d[c] = pd.to_numeric(d[c], errors="coerce")
    d = d.dropna().sort_values("Date").reset_index(drop=True)

    # Previous-day pivot ladder for today's decisions.
    d["yh"] = d["High"].shift(1)
    d["yl"] = d["Low"].shift(1)
    d["yc"] = d["Close"].shift(1)
    d["pp"] = (d["yh"] + d["yl"] + d["yc"]) / 3.0
    d["r1"] = 2.0 * d["pp"] - d["yl"]
    d["s1"] = 2.0 * d["pp"] - d["yh"]
    d = d.dropna(subset=["pp", "r1", "s1"]).copy()
    d["dkey"] = d["Date"].dt.normalize()

    m = pd.read_csv(M15_PATH, usecols=["Datetime", "Open", "High", "Low", "Close"], low_memory=False)
    m["Datetime"] = pd.to_datetime(m["Datetime"], errors="coerce")
    for c in ["Open", "High", "Low", "Close"]:
        m[c] = pd.to_numeric(m[c], errors="coerce")
    m = m.dropna().sort_values("Datetime").reset_index(drop=True)
    m["dkey"] = m["Datetime"].dt.normalize()

    # Full 96-bar days only.
    full_days = m.groupby("dkey").size()
    full_days = set(full_days[full_days == 96].index)
    m = m[m["dkey"].isin(full_days)].copy()

    bars = {}
    for dkey, g in m.groupby("dkey", sort=True):
        bars[dkey] = {
            "open": float(g["Open"].iloc[0]),
            "h": g["High"].to_list(),
            "l": g["Low"].to_list(),
            "last_c": float(g["Close"].iloc[-1]),
        }

    info = {}
    for _, r in d.iterrows():
        k = r["dkey"]
        if k in bars:
            info[k] = {"pp": float(r["pp"]), "r1": float(r["r1"]), "s1": float(r["s1"])}
    days = sorted(info.keys())
    return days, info, bars


def run_one(days, info, bars, tp_mult: float):
    total_pips = 0.0
    wins = 0
    losses = 0
    time_exits = 0
    buys = 0
    sells = 0
    skipped_equal = 0
    skipped_invalid_sl = 0

    for d in days:
        lv = info[d]
        b = bars[d]
        entry = b["open"]
        pp = lv["pp"]

        # Direction by day-open location vs pivot.
        if entry < pp:
            side = "BUY"
            buys += 1
            base_dist = pp - entry
            sl = lv["s1"]  # requested: opposite side beyond S1 for buys (using S1 level)
            if sl >= entry:
                skipped_invalid_sl += 1
                continue
            tp = entry + tp_mult * base_dist
        elif entry > pp:
            side = "SELL"
            sells += 1
            base_dist = entry - pp
            sl = lv["r1"]  # opposite side beyond R1 for sells (using R1 level)
            if sl <= entry:
                skipped_invalid_sl += 1
                continue
            tp = entry - tp_mult * base_dist
        else:
            skipped_equal += 1
            continue

        done = False
        for h, l in zip(b["h"], b["l"]):
            if side == "BUY":
                hit_sl = l <= sl
                hit_tp = h >= tp
            else:
                hit_sl = h >= sl
                hit_tp = l <= tp

            if hit_sl and hit_tp:
                px = sl
                losses += 1
                done = True
            elif hit_sl:
                px = sl
                losses += 1
                done = True
            elif hit_tp:
                px = tp
                wins += 1
                done = True

            if done:
                if side == "BUY":
                    pips = ((px - entry) / PIP) - SPREAD_PIPS
                else:
                    pips = ((entry - px) / PIP) - SPREAD_PIPS
                total_pips += pips
                break

        if not done:
            px = b["last_c"]
            time_exits += 1
            if side == "BUY":
                pips = ((px - entry) / PIP) - SPREAD_PIPS
            else:
                pips = ((entry - px) / PIP) - SPREAD_PIPS
            total_pips += pips

    cal_days = len(days)
    return {
        "tp_mult_open_to_pivot": tp_mult,
        "sl_level_rule": "buy->S1, sell->R1",
        "wins": wins,
        "losses": losses,
        "time_exits": time_exits,
        "buys": buys,
        "sells": sells,
        "skipped_equal_open_pp": skipped_equal,
        "skipped_invalid_sl_side": skipped_invalid_sl,
        "win_rate_excl_time_exit": round(wins / max(wins + losses, 1), 6),
        "total_pips": round(total_pips, 4),
        "cal_ppd": round(total_pips / max(cal_days, 1), 4),
    }


def main():
    days, info, bars = load_data()
    rows = []
    for m in TP_MULT_GRID:
        r = run_one(days, info, bars, m)
        rows.append(r)
        print(f"tp_mult={m:.2f} cal_ppd={r['cal_ppd']:+.4f} wins={r['wins']} losses={r['losses']}")

    df = pd.DataFrame(rows).sort_values("cal_ppd", ascending=False).reset_index(drop=True)
    out = {
        "pair": "EURJPY",
        "period": [str(days[0].date()), str(days[-1].date())],
        "entry": "daily open",
        "direction_rule": "open<pivot buy, open>pivot sell",
        "tp_rule": "tp = tp_mult * |open-pivot| from entry",
        "sl_rule": "buy uses S1 level, sell uses R1 level",
        "spread_pips": SPREAD_PIPS,
        "tp_mult_grid": TP_MULT_GRID,
        "best": df.iloc[0].to_dict(),
        "all_results": df.to_dict(orient="records"),
    }

    out_json = OUT_DIR / "dayopen_to_pivot_tpmult_sl_s1r1_sweep_eurjpy_full_days.json"
    out_csv = OUT_DIR / "dayopen_to_pivot_tpmult_sl_s1r1_sweep_eurjpy_full_days.csv"
    out_json.write_text(json.dumps(out, indent=2), encoding="utf-8")
    df.to_csv(out_csv, index=False)

    print("\nBEST:")
    print(df.iloc[0].to_dict())
    print(f"saved={out_json}")
    print(f"saved={out_csv}")


if __name__ == "__main__":
    main()

