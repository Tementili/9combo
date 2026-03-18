"""
bl_runner.py  -  Baseline Discovery Step 1 Runner
==================================================
Main-only ATR14 backtest. No addon, no SAR, no BE.
Pairs: EURJPY, EURUSD, USDJPY.
Windows: 24h (00:00 UTC), 12h (12:00 UTC), 8h (08:00 UTC).

Honing protocol:
  Round 1: full 864-config grid on Train.
  Rounds 2+: local neighborhood around Train winner; Adjust monitored.
  Stop: Adjust improvement < 0.20 ppd for 2 consecutive rounds.
  Freeze winner per pair. OOS exactly once (lock enforced).

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
    SPLIT, MAIN_SIZE, WORKERS, ATR_PERIOD, WINDOWS,
    SL_MULTS, SL_MAIN_FRACS, TP_MODES, SKIP_FRACS,
    HONING_STOP_DELTA, MAX_ROUNDS, MIN_ROUNDS,
    WINDOW_UTC_START, RUNNER_VERSION, MODULE_TAG,
    _atr_series, _build_rows_all_windows,
    _run_backtest_bl, _summarize_bl,
    _pool_init_bl, _task_bl, _n_cal_days,
)

# ── Pair data paths ────────────────────────────────────────────────────────────
_DATA_ROOT = Path(r"C:/Users/temen/5 joulu/TrailingStopLoss pivot peak trading 6 march 2026")

PAIR_CFG = {
    "eurjpy": {
        "pip": 0.01, "cost_pips": 2.0,
        "daily": _DATA_ROOT / "pair_workspaces" / "eurjpy_2003_20260306_15m_bucket10_v1" / "eurjpy_daily_ohlcv_ver3.csv",
        "h1":    _DATA_ROOT / "pair_workspaces" / "eurjpy_2003_20260306_15m_bucket10_hardstop_v1" / "hourly chart" / "processed_eurjpy_data_60m_20030804_20260306.csv",
    },
    "eurusd": {
        "pip": 0.0001, "cost_pips": 1.0,
        "daily": _DATA_ROOT / "NEW_AGENT_HOURLY_33PIPS_PACK" / "pair_datasets" / "eurusd" / "processed_eurusd_data_daily.csv",
        "h1":    _DATA_ROOT / "NEW_AGENT_HOURLY_33PIPS_PACK" / "pair_datasets" / "eurusd" / "processed_eurusd_data_60m.csv",
    },
    "usdjpy": {
        "pip": 0.01, "cost_pips": 2.0,
        "daily": _DATA_ROOT / "NEW_AGENT_HOURLY_33PIPS_PACK" / "pair_datasets" / "usdjpy" / "processed_usdjpy_data_daily.csv",
        "h1":    _DATA_ROOT / "NEW_AGENT_HOURLY_33PIPS_PACK" / "pair_datasets" / "usdjpy" / "processed_usdjpy_data_60m.csv",
    },
}


# ── Data loaders ──────────────────────────────────────────────────────────────
def _load_daily(path: Path) -> pd.DataFrame:
    d = pd.read_csv(path, low_memory=False)
    d["Date"] = pd.to_datetime(d["Date"], errors="coerce")
    for col in ["Open", "High", "Low", "Close"]:
        d[col] = pd.to_numeric(d[col], errors="coerce")
    return d.dropna(subset=["Date", "Open", "High", "Low", "Close"]) \
             .sort_values("Date").reset_index(drop=True)


def _load_hourly(path: Path):
    m = pd.read_csv(path, usecols=["Datetime", "Open", "High", "Low", "Close"],
                    low_memory=False)
    m["Datetime"] = pd.to_datetime(m["Datetime"], errors="coerce")
    for col in ["Open", "High", "Low", "Close"]:
        m[col] = pd.to_numeric(m[col], errors="coerce")
    m = m.dropna().sort_values("Datetime").reset_index(drop=True)
    return (m["Datetime"].to_numpy(),
            m["Open"].to_numpy(float),
            m["High"].to_numpy(float),
            m["Low"].to_numpy(float),
            m["Close"].to_numpy(float))


# ── Grid generators ───────────────────────────────────────────────────────────
def _full_grid() -> list[dict]:
    """All 6x3x4x4x3 = 864 configs."""
    cfgs = []
    for wh in WINDOWS:
        for slm in SL_MULTS:
            for smf in SL_MAIN_FRACS:
                for tpm in TP_MODES:
                    for sf in SKIP_FRACS:
                        cfgs.append({
                            "window_h":     wh,
                            "sl_mult":      slm,
                            "sl_main_frac": smf,
                            "tp_mode":      tpm,
                            "skip_frac":    sf,
                        })
    return cfgs


def _neighbor_grid(winner: dict, round_n: int) -> list[dict]:
    """
    Local neighborhood around winner for rounds 2+.
    Numeric params: ±2 index positions (or ±1 for fine rounds).
    Categorical tp_mode: all 4.
    window_h: fixed at winner's value (window selected in R1).
    """
    def _nbrs(lst, val, n_steps):
        try:
            idx = lst.index(val)
        except ValueError:
            # winner value not in grid; find closest
            diffs = [abs(v - val) for v in lst]
            idx = diffs.index(min(diffs))
        lo = max(0, idx - n_steps)
        hi = min(len(lst) - 1, idx + n_steps)
        return lst[lo: hi + 1]

    steps = max(1, 3 - round_n // 3)   # narrow progressively
    wh  = winner["window_h"]
    slm_vals = _nbrs(SL_MULTS,      winner["sl_mult"],      steps)
    smf_vals = _nbrs(SL_MAIN_FRACS, winner["sl_main_frac"], steps)
    sf_vals  = _nbrs(SKIP_FRACS,    winner["skip_frac"],    steps)

    cfgs = []
    for slm in slm_vals:
        for smf in smf_vals:
            for tpm in TP_MODES:   # always sweep all tp modes
                for sf in sf_vals:
                    cfgs.append({
                        "window_h":     wh,
                        "sl_mult":      slm,
                        "sl_main_frac": smf,
                        "tp_mode":      tpm,
                        "skip_frac":    sf,
                    })
    # Deduplicate
    seen = set()
    unique = []
    for c in cfgs:
        k = (c["window_h"], c["sl_mult"], c["sl_main_frac"],
             c["tp_mode"], c["skip_frac"])
        if k not in seen:
            seen.add(k); unique.append(c)
    return unique


# ── Parallel sweep ─────────────────────────────────────────────────────────────
def _run_sweep(
    rows_records: list,
    n_cal: int,
    pip: float,
    cost_pips: float,
    highs: np.ndarray,
    lows: np.ndarray,
    closes: np.ndarray,
    cfgs: list[dict],
) -> list[tuple[dict, dict]]:
    """Run cfgs in parallel; return sorted (cfg, summary) by cal_ppd desc."""
    tasks = [(rows_records, n_cal, pip, cost_pips, cfg) for cfg in cfgs]
    results = []
    with concurrent.futures.ProcessPoolExecutor(
            max_workers=WORKERS,
            initializer=_pool_init_bl,
            initargs=(highs, lows, closes)) as pool:
        for cfg, summ in pool.map(_task_bl, tasks):
            results.append((cfg, summ))
    results.sort(key=lambda x: x[1]["cal_ppd"], reverse=True)
    return results


# ── Honing ─────────────────────────────────────────────────────────────────────
def _cfg_key(cfg: dict) -> str:
    return (f"w{cfg['window_h']}|slm{cfg['sl_mult']:.4f}"
            f"|smf{cfg['sl_main_frac']:.2f}|tpm={cfg['tp_mode']}"
            f"|sf{cfg['skip_frac']:.3f}")


def run_honing(
    tr_rows: pd.DataFrame, ad_rows: pd.DataFrame,
    pip: float, cost_pips: float,
    highs: np.ndarray, lows: np.ndarray, closes: np.ndarray,
    pair: str,
) -> tuple[dict, dict, dict, list]:
    """
    Honing protocol (LOCKED):
      Train drives selection. Adjust monitored. OOS blocked until freeze.
      Stop when best Adjust improvement < HONING_STOP_DELTA for 2 rounds.
      Returns: (frozen_cfg, train_summ, adj_summ, round_history)
    """
    n_tr  = _n_cal_days(SPLIT["train_start"],  SPLIT["train_end"])
    n_adj = _n_cal_days(SPLIT["adjust_start"], SPLIT["adjust_end"])

    tr_rec  = tr_rows.to_dict("records")
    ad_rec  = ad_rows.to_dict("records")

    round_hist = []
    best_adj_ppd  = -999.0
    plateau_count = 0
    winner_cfg    = None

    for rnd in range(1, MAX_ROUNDS + 1):
        if rnd == 1:
            cfgs = _full_grid()
            print(f"  R{rnd:02d}: Full grid {len(cfgs)} configs ...", flush=True)
        else:
            cfgs = _neighbor_grid(winner_cfg, rnd)
            print(f"  R{rnd:02d}: Neighborhood {len(cfgs)} configs "
                  f"around {_cfg_key(winner_cfg)} ...", flush=True)

        # ── Sweep on Train only ────────────────────────────────────────────
        tr_results = _run_sweep(tr_rec, n_tr, pip, cost_pips, highs, lows, closes, cfgs)
        rnd_winner_cfg, rnd_winner_tr = tr_results[0]

        # ── Evaluate winner on Adjust (monitoring only) ────────────────────
        ad_res = _run_backtest_bl(ad_rows, highs, lows, closes, pip, cost_pips,
                                  rnd_winner_cfg)
        rnd_winner_adj = _summarize_bl(ad_res, n_adj)

        adj_ppd = rnd_winner_adj["cal_ppd"]
        delta   = adj_ppd - best_adj_ppd
        winner_cfg = rnd_winner_cfg

        print(f"      Train={rnd_winner_tr['cal_ppd']:+.4f}  "
              f"Adj={adj_ppd:+.4f}  Adj_delta={delta:+.4f}  "
              f"cfg={_cfg_key(winner_cfg)}", flush=True)

        round_hist.append({
            "round":       rnd,
            "n_configs":   len(cfgs),
            "winner_cfg":  winner_cfg,
            "train_ppd":   rnd_winner_tr["cal_ppd"],
            "adj_ppd":     adj_ppd,
            "adj_delta":   round(delta, 4),
        })

        if adj_ppd > best_adj_ppd:
            best_adj_ppd  = adj_ppd
            plateau_count = 0
        else:
            plateau_count += 1

        if rnd >= MIN_ROUNDS and plateau_count >= 2:
            print(f"  Stop: Adj plateau ({plateau_count}x < {HONING_STOP_DELTA}) "
                  f"at round {rnd}.", flush=True)
            break
        if rnd >= MIN_ROUNDS and delta < HONING_STOP_DELTA:
            plateau_count += 1   # partial credit — need 2 consecutive

    # ── Freeze winner ──────────────────────────────────────────────────────
    frozen_cfg = winner_cfg
    frozen_tr  = rnd_winner_tr
    frozen_adj = rnd_winner_adj

    print(f"  Frozen: {_cfg_key(frozen_cfg)}", flush=True)
    print(f"  Train={frozen_tr['cal_ppd']:+.4f}  "
          f"Adj={frozen_adj['cal_ppd']:+.4f}", flush=True)

    return frozen_cfg, frozen_tr, frozen_adj, round_hist


# ── OOS (one-shot, locked) ─────────────────────────────────────────────────────
def run_oos(
    oos_rows: pd.DataFrame,
    frozen_cfg: dict,
    pip: float,
    cost_pips: float,
    highs: np.ndarray,
    lows: np.ndarray,
    closes: np.ndarray,
    pair: str,
    code_hash: str,
) -> dict | None:
    lock_path = OUTPUT_DIR / f"bl_{pair}_oos_lock.json"
    res_path  = OUTPUT_DIR / f"bl_{pair}_oos_results.json"

    if lock_path.exists():
        lock = json.loads(lock_path.read_text())
        print(f"  OOS LOCKED (already run): "
              f"oos_cal_ppd={lock.get('oos_cal_ppd')}  ts={lock.get('timestamp')[:16]}")
        return None

    n_oos = _n_cal_days(SPLIT["oos_start"], SPLIT["oos_end"])
    res   = _run_backtest_bl(oos_rows, highs, lows, closes, pip, cost_pips, frozen_cfg)
    summ  = _summarize_bl(res, n_oos)

    ts = datetime.now(timezone.utc).isoformat()
    out = {
        "pair":          pair,
        "timestamp":     ts,
        "code_hash":     code_hash,
        "frozen_cfg":    frozen_cfg,
        "oos_cal_ppd":   summ["cal_ppd"],
        "oos_win_rate":  summ["win_rate"],
        "oos_avg_pips":  summ["avg_pips_per_trade"],
        "oos_ter":       summ["time_exit_rate"],
        "oos_trades":    summ["total_trades"],
        "oos_best_day":  summ["best_day_pips"],
        "oos_worst_day": summ["worst_day_pips"],
        "oos_n_cal":     n_oos,
    }
    res_path.write_text(json.dumps(out, indent=2))
    lock_path.write_text(json.dumps(out, indent=2))
    return out


# ── Code fingerprint ───────────────────────────────────────────────────────────
def _code_hash() -> str:
    src = Path(__file__).read_bytes() + (OUTPUT_DIR / "bl_defs.py").read_bytes()
    return hashlib.sha256(src).hexdigest()[:20]


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print(f"\n{'='*68}")
    print(f"  Baseline Discovery Step 1  {RUNNER_VERSION}")
    print(f"  Main leg only, ATR14, no addon/SAR/BE")
    print(f"  Grid: 6 sl_mult x 3 sl_frac x 4 tp x 4 skip x 3 windows = 864")
    print(f"  Write scope: baseline_discovery_v1/ ONLY")
    print(f"{'='*68}")

    # Split integrity guard (enforced before any data loads)
    te  = pd.Timestamp(SPLIT["train_end"])
    as_ = pd.Timestamp(SPLIT["adjust_start"])
    ae  = pd.Timestamp(SPLIT["adjust_end"])
    os_ = pd.Timestamp(SPLIT["oos_start"])
    assert te < as_ <= ae < os_, "LEAKAGE: split integrity violated"

    chash = _code_hash()
    all_results = {}

    for pair in ["eurjpy", "eurusd", "usdjpy"]:
        print(f"\n{'-'*60}\n  {pair.upper()}\n{'-'*60}")
        cfg_p     = PAIR_CFG[pair]
        pip       = cfg_p["pip"]
        cost_pips = cfg_p["cost_pips"]

        print("  Loading daily data ...", flush=True)
        daily = _load_daily(cfg_p["daily"])
        print("  Loading hourly data ...", flush=True)
        times, opens, highs, lows, closes = _load_hourly(cfg_p["h1"])
        print(f"  Daily rows: {len(daily)}  "
              f"Hourly bars: {len(highs)}", flush=True)

        # Build rows for all splits (separate row sets for Train/Adjust/OOS)
        print("  Building rows (all windows, causal) ...", flush=True)
        tr_rows  = _build_rows_all_windows(daily, times, opens,
                                           SPLIT["train_start"], SPLIT["train_end"])
        ad_rows  = _build_rows_all_windows(daily, times, opens,
                                           SPLIT["adjust_start"], SPLIT["adjust_end"])
        oos_rows = _build_rows_all_windows(daily, times, opens,
                                           SPLIT["oos_start"], SPLIT["oos_end"])

        # OOS leakage guard
        if len(tr_rows) > 0:
            assert tr_rows["date"].max() < pd.Timestamp(SPLIT["oos_start"]), \
                "LEAKAGE: OOS rows in Train"
        if len(ad_rows) > 0:
            assert ad_rows["date"].max() < pd.Timestamp(SPLIT["oos_start"]), \
                "LEAKAGE: OOS rows in Adjust"

        print(f"  Train rows: {len(tr_rows)}  "
              f"Adjust rows: {len(ad_rows)}  "
              f"OOS rows: {len(oos_rows)}", flush=True)

        # Warmup guard: ATR14 valid in first Train row
        valid_tr = tr_rows.dropna(subset=["ATR14"])
        assert len(valid_tr) > 0, f"{pair}: no valid ATR14 rows in Train"
        first_valid = valid_tr.iloc[0]
        print(f"  First valid Train row: {first_valid['trade_date_str']}  "
              f"ATR14={first_valid['ATR14']:.5f}", flush=True)

        # Run honing
        print("\n  Starting honing ...", flush=True)
        frozen_cfg, tr_summ, adj_summ, rnd_hist = run_honing(
            tr_rows, ad_rows, pip, cost_pips, highs, lows, closes, pair
        )

        # Save artifacts
        out = {
            "pair":         pair,
            "runner_version": RUNNER_VERSION,
            "timestamp":    datetime.now(timezone.utc).isoformat(),
            "code_hash":    chash,
            "frozen_cfg":   frozen_cfg,
            "train": tr_summ,
            "adjust": adj_summ,
            "round_history": rnd_hist,
            "split": SPLIT,
        }
        (OUTPUT_DIR / f"bl_{pair}_frozen.json").write_text(json.dumps(out, indent=2))
        print(f"  Frozen params written: bl_{pair}_frozen.json", flush=True)

        # OOS (one-shot, locked)
        print("\n  Running OOS (one-time) ...", flush=True)
        oos_out = run_oos(oos_rows, frozen_cfg, pip, cost_pips,
                          highs, lows, closes, pair, chash)
        if oos_out:
            print(f"  OOS: cal_ppd={oos_out['oos_cal_ppd']:+.4f}  "
                  f"WR={oos_out['oos_win_rate']:.3f}  "
                  f"trades={oos_out['oos_trades']}  "
                  f"worst={oos_out['oos_worst_day']:.1f}", flush=True)

        all_results[pair] = {"frozen": frozen_cfg, "train": tr_summ,
                              "adjust": adj_summ, "oos": oos_out}

    # Cross-pair summary
    print(f"\n{'='*68}\n  CROSS-PAIR SUMMARY\n{'='*68}")
    hdr = f"  {'Pair':<8} {'Window':>6} {'sl_mult':>8} {'sl_frac':>7} "
    hdr += f"{'tp_mode':>12} {'skip':>5} {'Tr PPD':>8} {'Adj PPD':>8}"
    print(hdr)
    for pair, res in all_results.items():
        fc   = res["frozen"]
        tr   = res["train"]
        adj  = res["adjust"]
        row  = (f"  {pair.upper():<8} {fc['window_h']:>6}h"
                f" {fc['sl_mult']:>8.4f} {fc['sl_main_frac']:>7.2f}"
                f" {fc['tp_mode']:>12} {fc['skip_frac']:>5.2f}"
                f" {tr['cal_ppd']:>8.4f} {adj['cal_ppd']:>8.4f}")
        print(row)

    summ_path = OUTPUT_DIR / "bl_step1_summary.json"
    summ_path.write_text(json.dumps(all_results, indent=2, default=str))
    print(f"\n  Summary written: {summ_path.name}")
    print("  Step 1 complete. Stop gate: awaiting supervisor GO for Layer 2.")
    print(f"{'='*68}\n")


if __name__ == "__main__":
    main()
