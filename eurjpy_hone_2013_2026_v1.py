from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from bl_runner import PAIR_CFG, _load_daily, _load_hourly


# Locked by user request
PAIR = "eurjpy"
REFERENCE_MODE = "pivot"  # above-only gate against pivot
MAX_HOLD_HOURS = 432
SELL_SIZE = 130.0
BUY_SIZE = 100.0

TRAIN_START = "2013-01-01"
TRAIN_END = "2016-12-31"
ADJ_START = "2017-01-01"
ADJ_END = "2022-12-31"
OOS_START = "2023-01-01"
OOS_END = "2026-03-06"

# "Use 10 CPUs only": this script intentionally runs single-process to avoid PC jamming.
MAX_CPUS_DECLARED = 10

OUTPUT_DIR = Path(__file__).parent


def pip_size(pair: str) -> float:
    return 0.01 if pair.endswith("jpy") else 0.0001


def first_bar_per_day(times) -> dict:
    out = {}
    for i, ts in enumerate(times):
        d = pd.Timestamp(ts).date()
        if d not in out:
            out[d] = i
    return out


def build_day_rows(daily: pd.DataFrame, opens, times) -> list[dict]:
    day_map = first_bar_per_day(times)
    rows: list[dict] = []
    for i in range(1, len(daily)):
        d = pd.Timestamp(daily.loc[i, "Date"])
        key = d.date()
        if key not in day_map:
            continue
        yh = float(daily.loc[i - 1, "High"])
        yl = float(daily.loc[i - 1, "Low"])
        yc = float(daily.loc[i - 1, "Close"])
        if yh <= yl:
            continue
        bar_idx = int(day_map[key])
        dopen = float(opens[bar_idx])
        rows.append(
            {
                "date": d,
                "bar_idx": bar_idx,
                "dopen": dopen,
                "yh": yh,
                "yl": yl,
                "yc": yc,
            }
        )
    return rows


def split_rows(rows: list[dict], start: str, end: str) -> list[dict]:
    s = pd.Timestamp(start)
    e = pd.Timestamp(end)
    return [r for r in rows if s <= r["date"] <= e]


def simulate_config(rows: list[dict], highs, lows, closes, pip: float, cfg: dict) -> dict:
    buy_tp_above_yh = float(cfg["buy_tp_above_yh_pips"])
    sell_tp_after_below_yh = float(cfg["sell_tp_after_buytp_below_yh_pips"])
    sell_sl_above_yh = float(cfg["sell_sl_above_yh_pips"])
    shared_offset = float(cfg["shared_selltp_buysl_offset_from_yl_pips"])

    tot_pips = 0.0
    tot_weighted = 0.0
    traded_days = 0
    case_counts = {
        "BUY_TP_THEN_SELL_TP_MOVED": 0,
        "SELL_TP_INITIAL_BEFORE_BUYTP": 0,
        "BUY_TP_THEN_SELL_SL": 0,
        "MIXED_TP_TIME_EXIT": 0,
        "BOTH_TIME_EXIT": 0,
        "OTHER": 0,
    }

    for r in rows:
        yh = r["yh"]
        yl = r["yl"]
        yc = r["yc"]
        dopen = r["dopen"]
        b0 = r["bar_idx"]

        ref = (yh + yl + yc) / 3.0 if REFERENCE_MODE == "pivot" else (yh + yl) / 2.0
        if dopen <= ref:
            continue

        traded_days += 1

        buy_entry = dopen
        sell_entry = dopen
        buy_sl = yl + shared_offset * pip
        buy_tp = yh + buy_tp_above_yh * pip
        sell_sl = yh + sell_sl_above_yh * pip
        sell_tp = yl + shared_offset * pip
        sell_tp_moved = False

        buy_open = True
        sell_open = True
        buy_exit_px = None
        sell_exit_px = None
        buy_reason = None
        sell_reason = None

        end_bar = min(b0 + MAX_HOLD_HOURS, len(highs))
        for j in range(b0, end_bar):
            h = float(highs[j])
            l = float(lows[j])

            # BUY first; if TP hits, move SELL TP.
            if buy_open:
                buy_hit_sl = l <= buy_sl
                buy_hit_tp = h >= buy_tp
                if buy_hit_sl and buy_hit_tp:
                    buy_open = False
                    buy_exit_px = buy_sl
                    buy_reason = "BOTH_SL"
                elif buy_hit_sl:
                    buy_open = False
                    buy_exit_px = buy_sl
                    buy_reason = "SL"
                elif buy_hit_tp:
                    buy_open = False
                    buy_exit_px = buy_tp
                    buy_reason = "TP"
                    if sell_open:
                        sell_tp = yh - sell_tp_after_below_yh * pip
                        sell_tp_moved = True

            if sell_open:
                sell_hit_sl = h >= sell_sl
                sell_hit_tp = l <= sell_tp
                if sell_hit_sl and sell_hit_tp:
                    sell_open = False
                    sell_exit_px = sell_sl
                    sell_reason = "BOTH_SL"
                elif sell_hit_sl:
                    sell_open = False
                    sell_exit_px = sell_sl
                    sell_reason = "SL"
                elif sell_hit_tp:
                    sell_open = False
                    sell_exit_px = sell_tp
                    sell_reason = "TP"

            if (not buy_open) and (not sell_open):
                break

        last_close = float(closes[end_bar - 1])
        if buy_open:
            buy_open = False
            buy_exit_px = last_close
            buy_reason = "TIME_EXIT"
        if sell_open:
            sell_open = False
            sell_exit_px = last_close
            sell_reason = "TIME_EXIT"

        buy_p = (buy_exit_px - buy_entry) / pip
        sell_p = (sell_entry - sell_exit_px) / pip
        tot_pips += (buy_p + sell_p)
        tot_weighted += (buy_p * BUY_SIZE + sell_p * SELL_SIZE)

        if (buy_reason == "TP") and (sell_reason == "TP") and sell_tp_moved:
            case_counts["BUY_TP_THEN_SELL_TP_MOVED"] += 1
        elif (sell_reason == "TP") and (not sell_tp_moved):
            case_counts["SELL_TP_INITIAL_BEFORE_BUYTP"] += 1
        elif (buy_reason == "TP") and (sell_reason in ("SL", "BOTH_SL")):
            case_counts["BUY_TP_THEN_SELL_SL"] += 1
        elif (buy_reason == "TP") and (sell_reason == "TIME_EXIT"):
            case_counts["MIXED_TP_TIME_EXIT"] += 1
        elif (buy_reason == "TIME_EXIT") and (sell_reason == "TIME_EXIT"):
            case_counts["BOTH_TIME_EXIT"] += 1
        else:
            case_counts["OTHER"] += 1

    return {
        "traded_days": traded_days,
        "total_pips": round(tot_pips, 4),
        "total_weighted_pips": round(tot_weighted, 4),
        "cases": case_counts,
    }


def ppd(total_pips: float, start: str, end: str) -> float:
    days = (pd.Timestamp(end) - pd.Timestamp(start)).days + 1
    return total_pips / max(days, 1)


def grid_round(round_n: int, best_cfg: dict | None) -> list[dict]:
    if round_n == 1:
        buy_tp_grid = [10, 20, 30]
        sell_move_grid = [5, 15, 25]
        sell_sl_grid = [200, 300, 400]
        shared_grid = [-20, 0, 20]
    else:
        # neighborhood around best
        b = int(best_cfg["buy_tp_above_yh_pips"])
        m = int(best_cfg["sell_tp_after_buytp_below_yh_pips"])
        s = int(best_cfg["sell_sl_above_yh_pips"])
        h = int(best_cfg["shared_selltp_buysl_offset_from_yl_pips"])
        buy_tp_grid = sorted(set([max(5, b - 5), b, b + 5]))
        sell_move_grid = sorted(set([max(1, m - 5), m, m + 5]))
        sell_sl_grid = sorted(set([max(100, s - 50), s, s + 50]))
        shared_grid = sorted(set([h - 10, h, h + 10]))

    cfgs = []
    for b in buy_tp_grid:
        for m in sell_move_grid:
            for s in sell_sl_grid:
                for h in shared_grid:
                    cfgs.append(
                        {
                            "buy_tp_above_yh_pips": float(b),
                            "sell_tp_after_buytp_below_yh_pips": float(m),
                            "sell_sl_above_yh_pips": float(s),
                            "shared_selltp_buysl_offset_from_yl_pips": float(h),
                        }
                    )
    # dedupe
    seen = set()
    out = []
    for c in cfgs:
        k = tuple(sorted(c.items()))
        if k not in seen:
            seen.add(k)
            out.append(c)
    return out


def main() -> None:
    cfgp = PAIR_CFG[PAIR]
    daily = _load_daily(Path(cfgp["daily"]))
    times, opens, highs, lows, closes = _load_hourly(Path(cfgp["h1"]))
    pip = pip_size(PAIR)

    all_rows = build_day_rows(daily, opens, times)
    train_rows = split_rows(all_rows, TRAIN_START, TRAIN_END)
    adj_rows = split_rows(all_rows, ADJ_START, ADJ_END)
    oos_rows = split_rows(all_rows, OOS_START, OOS_END)

    print(f"Pair={PAIR.upper()} | mode={REFERENCE_MODE} | CPUs cap={MAX_CPUS_DECLARED} (single-process)")
    print(f"Rows: train={len(train_rows)} adjust={len(adj_rows)} oos={len(oos_rows)}")

    best_cfg = None
    best_adj = None
    rounds = []
    plateau = 0

    for rnd in range(1, 7):
        cfgs = grid_round(rnd, best_cfg)
        print(f"\nR{rnd:02d}: testing {len(cfgs)} configs")
        winner = None
        winner_train_ppd = None
        winner_adj_ppd = None
        winner_train = None
        winner_adj = None

        for c in cfgs:
            tr = simulate_config(train_rows, highs, lows, closes, pip, c)
            ad = simulate_config(adj_rows, highs, lows, closes, pip, c)
            tr_ppd = ppd(tr["total_pips"], TRAIN_START, TRAIN_END)
            ad_ppd = ppd(ad["total_pips"], ADJ_START, ADJ_END)

            # train sanity + adjust objective
            score = ad_ppd
            if tr_ppd <= 0:
                score -= 9999.0

            if (winner is None) or (score > winner_adj_ppd):
                winner = c
                winner_train_ppd = tr_ppd
                winner_adj_ppd = score
                winner_train = tr
                winner_adj = ad

        # recover true adj ppd if penalized
        true_adj_ppd = ppd(winner_adj["total_pips"], ADJ_START, ADJ_END)
        delta = 0.0 if best_adj is None else (true_adj_ppd - best_adj)
        print(
            f"  best cfg: buyTP=YH+{winner['buy_tp_above_yh_pips']:.1f} "
            f"sellMove=YH-{winner['sell_tp_after_buytp_below_yh_pips']:.1f} "
            f"sellSL=YH+{winner['sell_sl_above_yh_pips']:.1f} "
            f"shared=YL{winner['shared_selltp_buysl_offset_from_yl_pips']:+.1f} | "
            f"train_ppd={winner_train_ppd:+.4f} adj_ppd={true_adj_ppd:+.4f} delta={delta:+.4f}"
        )

        rounds.append(
            {
                "round": rnd,
                "n_configs": len(cfgs),
                "winner_cfg": winner,
                "train_ppd": round(winner_train_ppd, 4),
                "adjust_ppd": round(true_adj_ppd, 4),
                "delta_vs_prev_adjust": round(delta, 4),
            }
        )

        if (best_adj is not None) and (delta < 0.2):
            plateau += 1
        else:
            plateau = 0

        best_cfg = winner
        best_adj = true_adj_ppd

        if rnd >= 3 and plateau >= 2:
            print("  stop: adjust plateau (<0.2 ppd for 2 rounds)")
            break

    # One-shot OOS
    oos = simulate_config(oos_rows, highs, lows, closes, pip, best_cfg)
    oos_ppd = ppd(oos["total_pips"], OOS_START, OOS_END)
    print("\nOOS (one-shot):")
    print(f"  total_pips={oos['total_pips']:+.4f}  ppd={oos_ppd:+.4f}  traded_days={oos['traded_days']}")
    print("  case counts:", oos["cases"])

    out = {
        "pair": PAIR.upper(),
        "reference_mode": REFERENCE_MODE,
        "split": {
            "train": [TRAIN_START, TRAIN_END],
            "adjust": [ADJ_START, ADJ_END],
            "oos": [OOS_START, OOS_END],
        },
        "max_cpus_declared": MAX_CPUS_DECLARED,
        "best_cfg": best_cfg,
        "rounds": rounds,
        "oos": {
            "total_pips": oos["total_pips"],
            "ppd": round(oos_ppd, 4),
            "traded_days": oos["traded_days"],
            "cases": oos["cases"],
        },
    }
    out_path = OUTPUT_DIR / f"eurjpy_hone_2013_2026_v1_result.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\nResult written: {out_path}")


if __name__ == "__main__":
    main()

