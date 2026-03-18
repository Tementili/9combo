"""
bl_layer4_runner.py  -  Layer 4: Main+Addon, BE_OFF vs BE_ON comparison
========================================================================
Grid: same base params as Layer 1 + ao_frac {0.5, 0.7, 1.0} = 2592 configs.
Each config is run with BE_OFF and BE_ON in the same pool task.

Honing:
  Round 1: full 2592-config grid on Train.
  Selection metric: BE_ON Train cal_ppd (since BE_ON is the corrected law).
  Rounds 2+: local neighborhood around Train winner.
  Stop: 2 consecutive rounds with BE_ON Adjust improvement < 0.20.
  Freeze: same params used for BOTH variants (BE impact isolated).
  OOS: one-shot, both variants with frozen params.

Side-by-side output per pair:
  main_pips | addon_pips | total_pips
  be_tp_count | sl_count | tp_count | time_exit_count
  cal_ppd on Train / Adjust / OOS

Write scope: baseline_discovery_v1/ ONLY.
"""
from __future__ import annotations

import sys
import json
import hashlib
import itertools
import concurrent.futures
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

OUTPUT_DIR = Path(__file__).parent
sys.path.insert(0, str(OUTPUT_DIR))

from bl_defs import (
    SPLIT, WORKERS, WINDOWS,
    SL_MULTS, SL_MAIN_FRACS, TP_MODES, SKIP_FRACS,
    HONING_STOP_DELTA, MAX_ROUNDS, MIN_ROUNDS,
    _build_rows_all_windows, _n_cal_days,
)
from bl_layer4_defs import (
    AO_SIZE, AO_FRACS,
    _run_layer4, _summarize_layer4,
    _pool_init_l4, _task_l4,
)
from bl_runner import PAIR_CFG, _load_daily, _load_hourly

RUNNER_VERSION = "v1.0.0-layer4"


# ── Grid ───────────────────────────────────────────────────────────────────────
def _full_grid_l4() -> list[dict]:
    """2592 configs: 6x3x4x4x3 base x 3 ao_frac."""
    cfgs = []
    for wh in WINDOWS:
        for slm in SL_MULTS:
            for smf in SL_MAIN_FRACS:
                for tpm in TP_MODES:
                    for sf in SKIP_FRACS:
                        for aof in AO_FRACS:
                            cfgs.append({
                                "window_h":     wh,
                                "sl_mult":      slm,
                                "sl_main_frac": smf,
                                "tp_mode":      tpm,
                                "skip_frac":    sf,
                                "ao_frac":      aof,
                            })
    return cfgs


def _neighbor_grid_l4(winner: dict, rnd: int) -> list[dict]:
    def _nbrs(lst, val, steps):
        try:    idx = lst.index(val)
        except ValueError:
            diffs = [abs(v - val) for v in lst]; idx = diffs.index(min(diffs))
        return lst[max(0, idx - steps): min(len(lst) - 1, idx + steps) + 1]

    s   = max(1, 3 - rnd // 3)
    wh  = winner["window_h"]
    cfgs = []
    for slm in _nbrs(SL_MULTS,      winner["sl_mult"],      s):
        for smf in _nbrs(SL_MAIN_FRACS, winner["sl_main_frac"], s):
            for tpm in TP_MODES:
                for sf in _nbrs(SKIP_FRACS, winner["skip_frac"], s):
                    for aof in _nbrs(AO_FRACS, winner["ao_frac"], s):
                        cfgs.append({"window_h": wh, "sl_mult": slm,
                                     "sl_main_frac": smf, "tp_mode": tpm,
                                     "skip_frac": sf, "ao_frac": aof})
    seen = set(); unique = []
    for c in cfgs:
        k = (c["window_h"], c["sl_mult"], c["sl_main_frac"],
             c["tp_mode"], c["skip_frac"], c["ao_frac"])
        if k not in seen: seen.add(k); unique.append(c)
    return unique


# ── Parallel sweep ─────────────────────────────────────────────────────────────
def _run_sweep_l4(rows_records, n_cal, pip, cost, highs, lows, closes, cfgs):
    tasks = [(rows_records, n_cal, pip, cost, cfg) for cfg in cfgs]
    results = []
    with concurrent.futures.ProcessPoolExecutor(
            max_workers=WORKERS,
            initializer=_pool_init_l4,
            initargs=(highs, lows, closes)) as pool:
        for cfg, summ_off, summ_on in pool.map(_task_l4, tasks):
            results.append((cfg, summ_off, summ_on))
    # Sort by BE_ON Train cal_ppd (honing selection metric)
    results.sort(key=lambda x: x[2]["cal_ppd"], reverse=True)
    return results


# ── Config key ─────────────────────────────────────────────────────────────────
def _ckey(cfg):
    return (f"w{cfg['window_h']}|slm{cfg['sl_mult']:.2f}"
            f"|smf{cfg['sl_main_frac']:.2f}|{cfg['tp_mode']}"
            f"|sf{cfg['skip_frac']:.2f}|ao{cfg['ao_frac']:.2f}")


# ── Honing ─────────────────────────────────────────────────────────────────────
def run_honing_l4(tr_rows, ad_rows, pip, cost, highs, lows, closes):
    n_tr  = _n_cal_days(SPLIT["train_start"],  SPLIT["train_end"])
    n_adj = _n_cal_days(SPLIT["adjust_start"], SPLIT["adjust_end"])
    tr_rec = tr_rows.to_dict("records")

    best_adj_on = -9999.0
    plateau     = 0
    winner      = None
    rnd_hist    = []

    for rnd in range(1, MAX_ROUNDS + 1):
        if rnd == 1:
            cfgs = _full_grid_l4()
            print(f"  R{rnd:02d}: Full grid {len(cfgs)} configs ...", flush=True)
        else:
            cfgs = _neighbor_grid_l4(winner, rnd)
            print(f"  R{rnd:02d}: Neighborhood {len(cfgs)} configs ...", flush=True)

        tr_res = _run_sweep_l4(tr_rec, n_tr, pip, cost, highs, lows, closes, cfgs)
        winner_cfg, winner_off_tr, winner_on_tr = tr_res[0]

        # Evaluate winner on Adjust (both variants)
        df = ad_rows
        ad_off = _summarize_layer4(
            _run_layer4(df, highs, lows, closes, pip, cost, winner_cfg, be_on=False), n_adj)
        ad_on  = _summarize_layer4(
            _run_layer4(df, highs, lows, closes, pip, cost, winner_cfg, be_on=True),  n_adj)

        adj_on_ppd = ad_on["cal_ppd"]
        delta      = adj_on_ppd - best_adj_on
        winner     = winner_cfg

        print(f"      BE_OFF Tr={winner_off_tr['cal_ppd']:+.4f}  "
              f"BE_ON Tr={winner_on_tr['cal_ppd']:+.4f}  "
              f"BE_ON Adj={adj_on_ppd:+.4f}  d={delta:+.4f}  "
              f"cfg={_ckey(winner)}", flush=True)

        rnd_hist.append({
            "round":          rnd,
            "n_configs":      len(cfgs),
            "winner_cfg":     winner,
            "be_off_tr_ppd":  winner_off_tr["cal_ppd"],
            "be_on_tr_ppd":   winner_on_tr["cal_ppd"],
            "be_on_adj_ppd":  adj_on_ppd,
            "adj_delta":      round(delta, 4),
        })

        if adj_on_ppd > best_adj_on:
            best_adj_on = adj_on_ppd; plateau = 0
        else:
            plateau += 1

        if rnd >= MIN_ROUNDS and plateau >= 2:
            print(f"  Stop: BE_ON Adj plateau ({plateau}x) at R{rnd}.", flush=True)
            break
        if rnd >= MIN_ROUNDS and delta < HONING_STOP_DELTA:
            plateau += 1

    frozen = winner
    # Final Adjust with frozen params (both variants)
    fr_off_tr = winner_off_tr
    fr_on_tr  = winner_on_tr
    fr_off_adj = ad_off
    fr_on_adj  = ad_on

    print(f"  Frozen: {_ckey(frozen)}", flush=True)
    print(f"  BE_OFF  Tr={fr_off_tr['cal_ppd']:+.4f}  Adj={fr_off_adj['cal_ppd']:+.4f}", flush=True)
    print(f"  BE_ON   Tr={fr_on_tr['cal_ppd']:+.4f}  Adj={fr_on_adj['cal_ppd']:+.4f}", flush=True)

    return frozen, fr_off_tr, fr_on_tr, fr_off_adj, fr_on_adj, rnd_hist


# ── OOS (one-shot, locked) ─────────────────────────────────────────────────────
def run_oos_l4(oos_rows, frozen, pip, cost, highs, lows, closes, pair, chash):
    lock_path = OUTPUT_DIR / f"bl_l4_{pair}_oos_lock.json"
    res_path  = OUTPUT_DIR / f"bl_l4_{pair}_oos_results.json"

    if lock_path.exists():
        lock = json.loads(lock_path.read_text())
        print(f"  OOS LOCKED (already run): ts={lock.get('timestamp', '')[:16]}")
        return None, None

    n_oos = _n_cal_days(SPLIT["oos_start"], SPLIT["oos_end"])
    res_off = _run_layer4(oos_rows, highs, lows, closes, pip, cost, frozen, be_on=False)
    res_on  = _run_layer4(oos_rows, highs, lows, closes, pip, cost, frozen, be_on=True)
    s_off   = _summarize_layer4(res_off, n_oos)
    s_on    = _summarize_layer4(res_on,  n_oos)

    ts = datetime.now(timezone.utc).isoformat()
    out = {
        "pair": pair, "timestamp": ts, "code_hash": chash,
        "frozen_cfg": frozen, "n_oos": n_oos,
        "be_off": s_off, "be_on": s_on,
    }
    res_path.write_text(json.dumps(out, indent=2))
    lock_path.write_text(json.dumps(out, indent=2))
    return s_off, s_on


# ── Side-by-side printer ───────────────────────────────────────────────────────
def _print_comparison(pair: str, frozen: dict,
                       off_tr, on_tr, off_adj, on_adj,
                       off_oos, on_oos):
    lbl  = _ckey(frozen)
    head = (f"\n  {'Metric':<24} {'BE_OFF':>10} {'BE_ON':>10} {'Delta(ON-OFF)':>14}")
    sep  = "  " + "-" * 60

    def _row(name, a, b, fmt="{:+.4f}"):
        if a is None or b is None:
            return f"  {name:<24} {'N/A':>10} {'N/A':>10} {'N/A':>14}"
        d = b - a if isinstance(b, (int, float)) and isinstance(a, (int, float)) else "?"
        fa = fmt.format(a)
        fb = fmt.format(b)
        fd = fmt.format(d) if isinstance(d, float) else str(d)
        return f"  {name:<24} {fa:>10} {fb:>10} {fd:>14}"

    def _irow(name, a, b):
        d  = b - a
        return f"  {name:<24} {a:>10d} {b:>10d} {d:>+14d}"

    print(f"\n  {pair.upper()} — Layer 4 (Main+Addon)  frozen: {lbl}")
    print(sep)
    print(head)
    print(sep)

    for split_lbl, off, on in [("TRAIN", off_tr, on_tr),
                                 ("ADJUST", off_adj, on_adj),
                                 ("OOS", off_oos, on_oos)]:
        if off is None:
            continue
        print(f"\n  [{split_lbl}]")
        print(_row("cal_ppd",        off["cal_ppd"],       on["cal_ppd"]))
        print(_row("  main_cal_ppd", off["main_cal_ppd"],  on["main_cal_ppd"]))
        print(_row("  addon_cal_ppd",off["addon_cal_ppd"], on["addon_cal_ppd"]))
        print(_row("win_rate",        off["win_rate"],       on["win_rate"],    "{:.4f}"))
        print(_irow("  sl_count",     off["sl_count"],       on["sl_count"]))
        print(_irow("  tp_count",     off["tp_count"],       on["tp_count"]))
        print(_irow("  be_tp_count",  off["be_tp_count"],    on["be_tp_count"]))
        print(_irow("  time_exit",    off["time_exit_count"],on["time_exit_count"]))
        print(_row("addon_fire_rate", off["addon_fire_rate"],on["addon_fire_rate"], "{:.4f}"))
        print(_row("worst_day",       off["worst_day"],      on["worst_day"]))
        print(_row("best_day",        off["best_day"],       on["best_day"]))

    print(sep)


# ── Code hash ──────────────────────────────────────────────────────────────────
def _chash():
    src = (Path(__file__).read_bytes()
           + (OUTPUT_DIR / "bl_layer4_defs.py").read_bytes()
           + (OUTPUT_DIR / "bl_defs.py").read_bytes())
    return hashlib.sha256(src).hexdigest()[:20]


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    print(f"\n{'='*68}")
    print(f"  Layer 4: Main + Addon  (BE_OFF vs BE_ON)  {RUNNER_VERSION}")
    print(f"  Grid: 2592 configs (864 base x 3 ao_frac)  ao_frac scales sl_dist")
    print(f"  Each config runs both BE variants in one pool task")
    print(f"  Selection: BE_ON Train cal_ppd  |  Freeze: same params both variants")
    print(f"  Write scope: baseline_discovery_v1/ ONLY")
    print(f"{'='*68}")

    chash  = _chash()
    all_r  = {}

    for pair in ["eurjpy", "eurusd", "usdjpy"]:
        print(f"\n{'-'*60}\n  {pair.upper()}\n{'-'*60}")
        cfg_p = PAIR_CFG[pair]
        pip   = cfg_p["pip"]; cost = cfg_p["cost_pips"]

        print("  Loading data ...", flush=True)
        daily = _load_daily(cfg_p["daily"])
        times, opens, highs, lows, closes = _load_hourly(cfg_p["h1"])
        print(f"  Daily={len(daily)} rows  Hourly={len(highs)} bars", flush=True)

        print("  Building rows (all windows, causal) ...", flush=True)
        tr_rows  = _build_rows_all_windows(daily, times, opens,
                                           SPLIT["train_start"],  SPLIT["train_end"])
        ad_rows  = _build_rows_all_windows(daily, times, opens,
                                           SPLIT["adjust_start"], SPLIT["adjust_end"])
        oos_rows = _build_rows_all_windows(daily, times, opens,
                                           SPLIT["oos_start"],    SPLIT["oos_end"])

        print(f"  Train={len(tr_rows)}  Adj={len(ad_rows)}  OOS={len(oos_rows)}", flush=True)

        print("\n  Honing (BE_ON Train drives selection) ...", flush=True)
        frozen, off_tr, on_tr, off_adj, on_adj, rnd_hist = run_honing_l4(
            tr_rows, ad_rows, pip, cost, highs, lows, closes)

        # Save frozen artifact
        frozen_out = {
            "pair": pair, "runner_version": RUNNER_VERSION,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "code_hash": chash, "frozen_cfg": frozen,
            "round_history": rnd_hist,
            "be_off": {"train": off_tr, "adjust": off_adj},
            "be_on":  {"train": on_tr,  "adjust": on_adj},
        }
        (OUTPUT_DIR / f"bl_l4_{pair}_frozen.json").write_text(
            json.dumps(frozen_out, indent=2))
        print(f"  Frozen artifact written: bl_l4_{pair}_frozen.json", flush=True)

        # OOS one-shot
        print("\n  Running OOS (one-time, both variants) ...", flush=True)
        oos_off, oos_on = run_oos_l4(oos_rows, frozen, pip, cost,
                                      highs, lows, closes, pair, chash)

        # Print side-by-side comparison
        _print_comparison(pair, frozen,
                          off_tr, on_tr, off_adj, on_adj, oos_off, oos_on)

        all_r[pair] = {
            "frozen": frozen,
            "be_off": {"train": off_tr, "adjust": off_adj, "oos": oos_off},
            "be_on":  {"train": on_tr,  "adjust": on_adj,  "oos": oos_on},
        }

    # Cross-pair summary table
    print(f"\n{'='*68}")
    print("  CROSS-PAIR SUMMARY — Layer 4")
    print(f"{'='*68}")
    hdr = (f"  {'Pair':<8} {'Window':>5}h {'ao_frac':>8} {'tp_mode':>12} "
           f"{'skip':>5} | {'OFF Tr':>8} {'ON Tr':>8} {'OFF Adj':>8} {'ON Adj':>8} "
           f"| {'OFF OOS':>8} {'ON OOS':>8} {'Delta OOS':>10}")
    print(hdr)
    print("  " + "-" * 105)
    for p, res in all_r.items():
        fc  = res["frozen"]
        row = (f"  {p.upper():<8} {fc['window_h']:>5} {fc['ao_frac']:>8.2f} "
               f"{fc['tp_mode']:>12} {fc['skip_frac']:>5.2f} | "
               f"{res['be_off']['train']['cal_ppd']:>8.4f} "
               f"{res['be_on']['train']['cal_ppd']:>8.4f} "
               f"{res['be_off']['adjust']['cal_ppd']:>8.4f} "
               f"{res['be_on']['adjust']['cal_ppd']:>8.4f} | ")
        oos_off = res["be_off"]["oos"]
        oos_on  = res["be_on"]["oos"]
        if oos_off and oos_on:
            d = oos_on["cal_ppd"] - oos_off["cal_ppd"]
            row += (f"{oos_off['cal_ppd']:>8.4f} "
                    f"{oos_on['cal_ppd']:>8.4f} {d:>+10.4f}")
        else:
            row += f"{'LOCKED':>8} {'LOCKED':>8} {'---':>10}"
        print(row)

    (OUTPUT_DIR / "bl_l4_summary.json").write_text(
        json.dumps(all_r, indent=2, default=str))
    print("\n  Summary written: bl_l4_summary.json")
    print("  STOP GATE: Layer 4 complete. Awaiting next supervisor instruction.")
    print(f"{'='*68}\n")


if __name__ == "__main__":
    main()
