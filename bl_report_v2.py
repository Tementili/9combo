"""
bl_report_v2.py  -  Ledger, per-leg summary, integrity checks
=============================================================
Consumes leg record lists produced by bl_engine_v2.py and writes:
  layer{n}_trade_ledger_{pair}_{split}.csv
  layer{n}_leg_summary_{pair}.csv
  layer{n}_crosspair_summary.csv
  validation_checks_layer{n}.json
"""
from __future__ import annotations
import json
import math
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timezone
from typing import Sequence


# ── Ledger columns (strict order) ────────────────────────────────────────────
LEDGER_COLS = [
    "pair", "split", "layer", "be_mode", "config_id",
    "date", "window_h", "direction",
    "leg_tag", "side",
    "entry_bar_idx", "exit_bar_idx",
    "entry_px", "exit_px",
    "exit_reason", "filled",
    "pips_net", "lots",
    "skip_reason",
]


def build_ledger(records: list[dict], pair: str, split: str) -> pd.DataFrame:
    """
    Convert raw leg records into a ledger DataFrame with canonical columns.
    Adds pair and split columns.
    """
    if not records:
        return pd.DataFrame(columns=LEDGER_COLS)
    df = pd.DataFrame(records)
    df["pair"]  = pair
    df["split"] = split
    # Ensure all columns exist
    for col in LEDGER_COLS:
        if col not in df.columns:
            df[col] = None
    return df[LEDGER_COLS].reset_index(drop=True)


# ── Per-leg summary ────────────────────────────────────────────────────────────
def _leg_stats(df_leg: pd.DataFrame, size_col="lots") -> dict:
    """Compute stats for one leg tag from a filtered ledger."""
    filled = df_leg[df_leg["filled"] == True]
    n = len(filled)
    if n == 0:
        return {"n_trades": 0, "total_pips": 0.0, "cal_ppd": 0.0,
                "win_rate": 0.0, "time_exit_rate": 0.0,
                "best_day": 0.0, "worst_day": 0.0}
    tots   = filled["pips_net"].values
    wins   = int((tots > 0).sum())
    ter    = int((filled["exit_reason"] == "TIME_EXIT").sum())
    return {
        "n_trades":       n,
        "total_pips":     float(tots.sum()),
        "win_rate":       round(wins / n, 4),
        "time_exit_rate": round(ter / n, 4),
        "best_day":       round(float(tots.max()), 2),
        "worst_day":      round(float(tots.min()), 2),
    }


def build_leg_summary(ledger: pd.DataFrame, n_cal_days: int,
                       be_mode: str | None = None) -> dict:
    """
    Build per-leg summary from a ledger DataFrame.
    be_mode: filter to one BE variant ("BE_OFF" / "BE_ON" / None for non-L4).
    Returns dict with main_, addon_, sar_ and total_ keys.
    """
    df = ledger.copy()
    if be_mode and "be_mode" in df.columns:
        df = df[df["be_mode"] == be_mode]

    def leg(tag):
        return _leg_stats(df[df["leg_tag"] == tag])

    m = leg("main"); a = leg("addon"); s = leg("sar")
    total_pips = m["total_pips"] + a["total_pips"] + s["total_pips"]
    n_cal      = max(n_cal_days, 1)

    # exit reason breakdown for main
    main_filled = df[(df["leg_tag"] == "main") & (df["filled"] == True)]
    be_tp_count = int((main_filled["exit_reason"] == "BE_TP").sum())
    sl_count    = int(main_filled["exit_reason"].isin(["SL","BOTH_SL"]).sum())
    tp_count    = int((main_filled["exit_reason"] == "TP").sum())
    te_count    = int((main_filled["exit_reason"] == "TIME_EXIT").sum())

    return {
        "be_mode":            be_mode or "N/A",
        "n_cal_days":         n_cal_days,

        "main_n_trades":      m["n_trades"],
        "main_total_pips":    round(m["total_pips"], 2),
        "main_cal_ppd":       round(m["total_pips"] / n_cal, 4),
        "main_win_rate":      m["win_rate"],
        "main_ter":           m["time_exit_rate"],
        "main_best_day":      m["best_day"],
        "main_worst_day":     m["worst_day"],

        "addon_n_trades":     a["n_trades"],
        "addon_total_pips":   round(a["total_pips"], 2),
        "addon_cal_ppd":      round(a["total_pips"] / n_cal, 4),
        "addon_win_rate":     a["win_rate"],
        "addon_ter":          a["time_exit_rate"],

        "sar_n_trades":       s["n_trades"],
        "sar_total_pips":     round(s["total_pips"], 2),
        "sar_cal_ppd":        round(s["total_pips"] / n_cal, 4),
        "sar_win_rate":       s["win_rate"],
        "sar_ter":            s["time_exit_rate"],

        "total_pips":         round(total_pips, 2),
        "total_cal_ppd":      round(total_pips / n_cal, 4),

        # Exit reason breakdown (main leg only)
        "be_tp_count":        be_tp_count,
        "sl_count":           sl_count,
        "tp_count":           tp_count,
        "time_exit_count":    te_count,
    }


# ── Integrity checks ───────────────────────────────────────────────────────────
def check_integrity(summ: dict, be_on_summ: dict | None = None) -> dict:
    """
    Run integrity assertions.
    Returns dict: {check_name: {pass: bool, detail: str}}
    """
    checks = {}

    # Check 1: total_pips == main + addon + sar
    tol = 0.01
    computed = summ["main_total_pips"] + summ["addon_total_pips"] + summ["sar_total_pips"]
    diff = abs(computed - summ["total_pips"])
    checks["total_pips_equals_sum_of_legs"] = {
        "pass": diff < tol,
        "detail": f"total={summ['total_pips']:.2f} sum_legs={computed:.2f} diff={diff:.4f}",
    }

    # Check 2: total_cal_ppd == sum(leg_cal_ppd)
    computed_ppd = summ["main_cal_ppd"] + summ["addon_cal_ppd"] + summ["sar_cal_ppd"]
    diff_ppd = abs(computed_ppd - summ["total_cal_ppd"])
    checks["total_cal_ppd_equals_sum_of_legs"] = {
        "pass": diff_ppd < 0.001,
        "detail": f"total_ppd={summ['total_cal_ppd']:.4f} sum_legs={computed_ppd:.4f} diff={diff_ppd:.6f}",
    }

    # Check 3: BE_OFF has be_tp_count == 0
    checks["be_off_has_zero_be_tp"] = {
        "pass": summ["be_tp_count"] == 0 if summ["be_mode"] == "BE_OFF" else True,
        "detail": f"be_tp_count={summ['be_tp_count']} be_mode={summ['be_mode']}",
    }

    # Check 4: BE_ON has nonzero be_tp_count when addon fired
    if be_on_summ is not None:
        addon_fired = be_on_summ["addon_n_trades"] > 0
        be_tp_ok = (not addon_fired) or (be_on_summ["be_tp_count"] > 0)
        checks["be_on_has_be_tp_when_addon_fires"] = {
            "pass": be_tp_ok,
            "detail": (f"addon_n_trades={be_on_summ['addon_n_trades']} "
                       f"be_tp_count={be_on_summ['be_tp_count']}"),
        }

    all_pass = all(v["pass"] for v in checks.values())
    checks["_all_pass"] = all_pass
    return checks


# ── Artifact writers ───────────────────────────────────────────────────────────
def write_ledger(ledger: pd.DataFrame, out_dir: Path,
                 layer: str, pair: str, split: str) -> Path:
    fname = out_dir / f"layer{layer[1:]}_trade_ledger_{pair}_{split}.csv"
    ledger.to_csv(fname, index=False)
    return fname


def write_leg_summary(rows: list[dict], out_dir: Path,
                      layer: str, pair: str) -> Path:
    """rows: list of per-split summary dicts (train, adjust, oos)."""
    fname = out_dir / f"layer{layer[1:]}_leg_summary_{pair}.csv"
    pd.DataFrame(rows).to_csv(fname, index=False)
    return fname


def write_crosspair_summary(rows: list[dict], out_dir: Path,
                             layer: str) -> Path:
    fname = out_dir / f"layer{layer[1:]}_crosspair_summary.csv"
    pd.DataFrame(rows).to_csv(fname, index=False)
    return fname


def write_validation(checks: dict, out_dir: Path, layer: str) -> Path:
    fname = out_dir / f"validation_checks_layer{layer[1:]}.json"
    out = {"timestamp": datetime.now(timezone.utc).isoformat(),
           "layer": layer, "checks": checks}
    fname.write_text(json.dumps(out, indent=2))
    return fname
