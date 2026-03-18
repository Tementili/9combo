"""
run_layers_v2.py  -  Orchestrate L1-L4 for all pairs
=====================================================
Strict isolation: each layer is a separate experiment.
Honing: Train-only selection, Adjust monitored, stop on plateau.
OOS: one-shot per layer after freeze, locked.
Artifacts written to baseline_discovery_v1/ ONLY.
"""
from __future__ import annotations

import sys
import json
import hashlib
import concurrent.futures
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))

from bl_defs import (
    SPLIT, WORKERS, _build_rows_all_windows, _n_cal_days, HONING_STOP_DELTA,
    MAX_ROUNDS, MIN_ROUNDS,
)
from bl_engine_v2 import (
    build_grid, build_neighbor_grid, cfg_id,
    pool_init, pool_task, replay_config,
    MAIN_SIZE, AO_SIZE, SAR_SIZE,
)
from bl_report_v2 import (
    build_ledger, build_leg_summary, check_integrity,
    write_ledger, write_leg_summary, write_crosspair_summary, write_validation,
)
from bl_runner import PAIR_CFG, _load_daily, _load_hourly   # reuse data paths

OUTPUT_DIR = HERE

# ── Data cache (one load per pair per run) ────────────────────────────────────
def _load_pair(pair: str):
    cfg_p = PAIR_CFG[pair]
    daily = _load_daily(cfg_p["daily"])
    times, opens, highs, lows, closes = _load_hourly(cfg_p["h1"])
    print(f"  [{pair}] Daily={len(daily)}  Hourly={len(highs)}", flush=True)
    tr_rows  = _build_rows_all_windows(daily, times, opens,
                                       SPLIT["train_start"],  SPLIT["train_end"])
    ad_rows  = _build_rows_all_windows(daily, times, opens,
                                       SPLIT["adjust_start"], SPLIT["adjust_end"])
    oos_rows = _build_rows_all_windows(daily, times, opens,
                                       SPLIT["oos_start"],    SPLIT["oos_end"])
    print(f"  [{pair}] Train={len(tr_rows)}  Adj={len(ad_rows)}  OOS={len(oos_rows)}", flush=True)
    return (PAIR_CFG[pair]["pip"], PAIR_CFG[pair]["cost_pips"],
            times, opens, highs, lows, closes,
            tr_rows, ad_rows, oos_rows)


# ── Compute cal_ppd from records ───────────────────────────────────────────────
# ── Lightweight parallel sweep (no records returned, summary only) ─────────────
def _sweep(rows_records, pip, cost, highs, lows, closes, cfgs, layer, n_cal) -> list:
    """
    Returns list of (cfg, ppd_select, summ_off, summ_on) sorted by ppd_select desc.
    Does NOT return per-trade records — memory-safe for large grids.
    """
    tasks = [(rows_records, pip, cost, cfg, layer, n_cal) for cfg in cfgs]
    out = []
    with concurrent.futures.ProcessPoolExecutor(
            max_workers=WORKERS,
            initializer=pool_init,
            initargs=(highs, lows, closes)) as pool:
        for cfg, lyr, ppd_sel, s_off, s_on in pool.map(pool_task, tasks):
            out.append((cfg, ppd_sel, s_off, s_on))
    out.sort(key=lambda x: x[1], reverse=True)
    return out


def _replay(cfg, rows_df, pip, cost, highs, lows, closes, layer):
    """Replay one frozen config to get full per-trade records for ledger."""
    pool_init(highs, lows, closes)   # init globals in main process
    return replay_config(cfg, rows_df, pip, cost, layer)


# ── Honing ─────────────────────────────────────────────────────────────────────
def run_honing(layer: str, tr_rows, ad_rows, pip, cost, highs, lows, closes) -> tuple:
    """
    Phase 1 (sweep): lightweight summaries only — memory-safe for large grids.
    Phase 2 (replay): re-run frozen config to get full records for ledger.
    Returns: (frozen_cfg, tr_off, tr_on, ad_off, ad_on, history)
    """
    n_tr  = _n_cal_days(SPLIT["train_start"],  SPLIT["train_end"])
    n_adj = _n_cal_days(SPLIT["adjust_start"], SPLIT["adjust_end"])
    tr_rec = tr_rows.to_dict("records")
    ad_rec = ad_rows.to_dict("records")

    best_adj = -9999.0; plateau = 0; winner = None
    history = []

    # ── Phase 1: honing sweep (summary only, no records) ──────────────────────
    for rnd in range(1, MAX_ROUNDS + 1):
        cfgs = build_grid(layer) if rnd == 1 else build_neighbor_grid(winner, layer, rnd)
        label = "Full grid" if rnd == 1 else "Neighborhood"
        print(f"  R{rnd:02d}: {label} {len(cfgs)} configs ...", flush=True)

        tr_out = _sweep(tr_rec, pip, cost, highs, lows, closes, cfgs, layer, n_tr)
        w_cfg, w_ppd, w_s_off, w_s_on = tr_out[0]

        # Evaluate winner on Adjust (monitoring only)
        ad_out = _sweep(ad_rec, pip, cost, highs, lows, closes, [w_cfg], layer, n_adj)
        _, adj_ppd, w_s_off_adj, w_s_on_adj = ad_out[0]

        delta  = adj_ppd - best_adj
        winner = w_cfg
        print(f"      Tr={w_ppd:+.4f}  Adj={adj_ppd:+.4f}  d={delta:+.4f}  "
              f"cfg={cfg_id(winner)}", flush=True)

        history.append({"round": rnd, "n_configs": len(cfgs),
                         "winner_cfg": winner,
                         "tr_ppd": round(w_ppd, 4),
                         "adj_ppd": round(adj_ppd, 4),
                         "adj_delta": round(delta, 4)})

        if adj_ppd > best_adj: best_adj = adj_ppd; plateau = 0
        else: plateau += 1

        if rnd >= MIN_ROUNDS and plateau >= 2:
            print(f"  Stop: plateau ({plateau}x) at R{rnd}.", flush=True)
            break
        if rnd >= MIN_ROUNDS and delta < HONING_STOP_DELTA:
            plateau += 1

    # ── Phase 2: replay frozen winner to get full records for ledger ───────────
    print(f"  Replaying frozen config for ledger records ...", flush=True)
    tr_off,  tr_on  = _replay(winner, tr_rows, pip, cost, highs, lows, closes, layer)
    ad_off,  ad_on  = _replay(winner, ad_rows, pip, cost, highs, lows, closes, layer)

    return winner, tr_off, tr_on, ad_off, ad_on, history


# ── OOS (one-shot, locked per layer+pair) ─────────────────────────────────────
def run_oos(layer: str, pair: str, frozen: dict,
            oos_rows, pip, cost, highs, lows, closes, chash: str) -> tuple:
    lock = OUTPUT_DIR / f"layer{layer[1:]}_oos_lock_{pair}.json"
    if lock.exists():
        d = json.loads(lock.read_text())
        print(f"  OOS LOCKED: ts={d['timestamp'][:16]}")
        return None, None

    oos_off, oos_on = _replay(frozen, oos_rows, pip, cost, highs, lows, closes, layer)

    ts = datetime.now(timezone.utc).isoformat()
    info = {"pair": pair, "layer": layer, "timestamp": ts,
            "code_hash": chash, "frozen_cfg": frozen}
    lock.write_text(json.dumps(info, indent=2))
    return oos_off, oos_on


# ── Per-split artifact writer ─────────────────────────────────────────────────
def write_split_artifacts(layer: str, pair: str,
                           tr_off, tr_on, ad_off, ad_on, oos_off, oos_on,
                           n_tr: int, n_adj: int, n_oos: int,
                           all_checks: dict):
    ledgers_written = []
    summaries = []
    splits_data = [
        ("train",  tr_off,  tr_on,  n_tr),
        ("adjust", ad_off,  ad_on,  n_adj),
        ("oos",    oos_off, oos_on, n_oos),
    ]

    for split, recs_off, recs_on, n_cal in splits_data:
        if recs_off is None:
            continue
        if layer == "L4":
            for recs, bm in [(recs_off, "BE_OFF"), (recs_on, "BE_ON")]:
                led = build_ledger(recs, pair, split)
                lpath = write_ledger(led, OUTPUT_DIR, layer, pair, f"{split}_{bm.lower()}")
                ledgers_written.append(str(lpath))
                s = build_leg_summary(led, n_cal, be_mode=bm)
                s.update({"split": split, "pair": pair, "layer": layer})
                summaries.append(s)
        else:
            led = build_ledger(recs_off, pair, split)
            lpath = write_ledger(led, OUTPUT_DIR, layer, pair, split)
            ledgers_written.append(str(lpath))
            s = build_leg_summary(led, n_cal)
            s.update({"split": split, "pair": pair, "layer": layer})
            summaries.append(s)

    sfile = write_leg_summary(summaries, OUTPUT_DIR, layer, pair)
    return ledgers_written, str(sfile), summaries


# ── Topline printer ────────────────────────────────────────────────────────────
def _print_topline(layer: str, pair: str, summaries: list):
    tr_rows = [s for s in summaries if s["split"] == "train"]
    ad_rows = [s for s in summaries if s["split"] == "adjust"]
    oos_rows = [s for s in summaries if s["split"] == "oos"]

    def _ppd(rows, bm=None):
        for s in rows:
            if bm is None or s.get("be_mode") == bm:
                return s["total_cal_ppd"]
        return float("nan")

    if layer == "L4":
        print(f"\n  {pair.upper()} [{layer}]  (BE_OFF | BE_ON)")
        for bm in ("BE_OFF", "BE_ON"):
            print(f"    {bm}: Tr={_ppd(tr_rows,bm):+.4f}  "
                  f"Adj={_ppd(ad_rows,bm):+.4f}  "
                  f"OOS={_ppd(oos_rows,bm):+.4f}")
        # Breakdown for BE_ON OOS
        for s in oos_rows:
            if s.get("be_mode") == "BE_ON":
                print(f"    BE_ON OOS: main={s['main_cal_ppd']:+.4f}  "
                      f"addon={s['addon_cal_ppd']:+.4f}  "
                      f"be_tp={s['be_tp_count']}  sl={s['sl_count']}  "
                      f"tp={s['tp_count']}")
    else:
        print(f"\n  {pair.upper()} [{layer}]")
        for sp, rows in [("Tr", tr_rows), ("Adj", ad_rows), ("OOS", oos_rows)]:
            if rows:
                s = rows[0]
                print(f"    {sp}: total={s['total_cal_ppd']:+.4f}  "
                      f"main={s['main_cal_ppd']:+.4f}  "
                      f"addon={s['addon_cal_ppd']:+.4f}  "
                      f"sar={s['sar_cal_ppd']:+.4f}  "
                      f"n_main={s['main_n_trades']}")


# ── Main ───────────────────────────────────────────────────────────────────────
LAYERS = ["L1", "L2", "L3", "L4"]

def main():
    print(f"\n{'='*68}")
    print(f"  Baseline Discovery v2 — Strict Layer Isolation")
    print(f"  Layers: L1(main) L2(addon) L3(main+SAR) L4(main+addon BE)")
    print(f"  Pairs: EURJPY EURUSD USDJPY")
    print(f"  Write scope: baseline_discovery_v1/ ONLY")
    print(f"{'='*68}")

    chash = hashlib.sha256(
        Path(__file__).read_bytes() +
        (HERE / "bl_engine_v2.py").read_bytes() +
        (HERE / "bl_report_v2.py").read_bytes()
    ).hexdigest()[:20]

    n_tr  = _n_cal_days(SPLIT["train_start"],  SPLIT["train_end"])
    n_adj = _n_cal_days(SPLIT["adjust_start"], SPLIT["adjust_end"])
    n_oos = _n_cal_days(SPLIT["oos_start"],    SPLIT["oos_end"])

    all_pair_data = {}
    for pair in ["eurjpy", "eurusd", "usdjpy"]:
        print(f"\n  Loading {pair.upper()} ...", flush=True)
        all_pair_data[pair] = _load_pair(pair)

    all_validation = {}

    for layer in LAYERS:
        print(f"\n{'='*68}\n  LAYER {layer}\n{'='*68}")
        layer_summaries = []
        all_checks_layer = {}

        for pair in ["eurjpy", "eurusd", "usdjpy"]:
            print(f"\n  {'-'*50}\n  {layer} — {pair.upper()}\n  {'-'*50}")
            (pip, cost, times, opens, highs, lows, closes,
             tr_rows, ad_rows, oos_rows) = all_pair_data[pair]

            # Honing
            frozen, tr_off, tr_on, ad_off, ad_on, hist = run_honing(
                layer, tr_rows, ad_rows, pip, cost, highs, lows, closes)

            print(f"  Frozen: {cfg_id(frozen)}", flush=True)

            # Save frozen
            frozen_path = OUTPUT_DIR / f"layer{layer[1:]}_frozen_{pair}.json"
            frozen_path.write_text(json.dumps({
                "pair": pair, "layer": layer, "timestamp": datetime.now(timezone.utc).isoformat(),
                "code_hash": chash, "frozen_cfg": frozen, "round_history": hist,
            }, indent=2))

            # OOS one-shot
            print(f"  Running OOS (one-time) ...", flush=True)
            oos_off, oos_on = run_oos(layer, pair, frozen, oos_rows,
                                       pip, cost, highs, lows, closes, chash)

            # Write artifacts
            ledgers, sfile, sums = write_split_artifacts(
                layer, pair,
                tr_off, tr_on, ad_off, ad_on, oos_off, oos_on,
                n_tr, n_adj, n_oos, {}
            )

            # Integrity checks
            tr_summ_off = next((s for s in sums if s["split"]=="train" and s.get("be_mode") in ("BE_OFF","N/A")), {})
            tr_summ_on  = next((s for s in sums if s["split"]=="train" and s.get("be_mode") == "BE_ON"), None)
            checks = check_integrity(tr_summ_off if tr_summ_off else {}, tr_summ_on)
            checks["pair"] = pair
            all_checks_layer[pair] = checks

            # Print topline
            _print_topline(layer, pair, sums)

            # Collect for cross-pair summary
            for s in sums:
                s["pair"] = pair
                layer_summaries.append(s)

            print(f"  Ledgers: {len(ledgers)} files")
            print(f"  Summary: {sfile}")

        # Cross-pair summary
        cp_path = write_crosspair_summary(layer_summaries, OUTPUT_DIR, layer)
        print(f"\n  Cross-pair summary: {cp_path}")

        # Validation JSON
        val_path = write_validation(all_checks_layer, OUTPUT_DIR, layer)
        print(f"  Validation checks: {val_path}")

        all_validation[layer] = all_checks_layer

    # Final validation summary
    print(f"\n{'='*68}")
    print("  VALIDATION SUMMARY")
    print(f"{'='*68}")
    for layer, pairs_checks in all_validation.items():
        for pair, chks in pairs_checks.items():
            passed = chks.get("_all_pass", False)
            status = "PASS" if passed else "FAIL"
            print(f"  [{status}] {layer} {pair.upper()}")

    print(f"\n  All artifacts in: {OUTPUT_DIR}")
    print("  STOP GATE: L1-L4 complete. Awaiting supervisor review before L5.")
    print(f"{'='*68}\n")


if __name__ == "__main__":
    main()
