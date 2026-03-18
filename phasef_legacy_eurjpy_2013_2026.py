#!/usr/bin/env python3
from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


# Import legacy phase-f logic from existing corrected engine.
PHASEF_DIR = Path(
    r"C:\Users\temen\5 joulu\TrailingStopLoss pivot peak trading 6 march 2026\ATR 14 only mode\part 4\part 4 - For ATR 14 mode use\phase_f_honing_master"
)
sys.path.insert(0, str(PHASEF_DIR))
import phase_f_defs as _pf  # type: ignore
from phase_f_defs import (  # type: ignore
    BASE_PARAMS,
    C_BL_8H,
    F_PARENTS,
    PAIR_CFG,
    EPSILON,
    MAX_ROUNDS,
    MIN_ROUNDS,
    _adx_series,
    _atr_series,
    _make_f_cfg,
    _pool_init_phasef,
    _summarize_phasef,
    _task_phasef,
    _trending_mask,
    check_freeze_gate,
    check_hourly_gaps,
    check_oos_gate,
    excess_drift,
    gen_round1,
    gen_round2,
    gen_round3plus,
    select_winner,
)


RUNNER_VERSION = "legacy-phasef-eurjpy-2013-2026-v1"
OUTPUT_DIR = Path(__file__).parent
WORKERS = 10  # user requested max 10 CPUs
SPLIT = {
    "train_start": "2013-01-01",
    "train_end": "2016-12-31",
    "adjust_start": "2017-01-01",
    "adjust_end": "2022-12-31",
    "oos_start": "2023-01-01",
    "oos_end": "2026-03-06",
}


def _load_daily(path: Path) -> pd.DataFrame:
    d = pd.read_csv(path, low_memory=False)
    d["Date"] = pd.to_datetime(d["Date"], errors="coerce")
    for c in ["Open", "High", "Low", "Close"]:
        d[c] = pd.to_numeric(d[c], errors="coerce")
    return d.dropna(subset=["Date", "Open", "High", "Low", "Close"]).sort_values("Date").reset_index(drop=True)


def _load_hourly(path: Path):
    m = pd.read_csv(path, usecols=["Datetime", "Open", "High", "Low", "Close"], low_memory=False)
    m["Datetime"] = pd.to_datetime(m["Datetime"], errors="coerce")
    for c in ["Open", "High", "Low", "Close"]:
        m[c] = pd.to_numeric(m[c], errors="coerce")
    m = m.dropna(subset=["Datetime", "Open", "High", "Low", "Close"]).sort_values("Datetime").reset_index(drop=True)
    return m["Datetime"].to_numpy(), m["Open"].to_numpy(float), m["High"].to_numpy(float), m["Low"].to_numpy(float), m["Close"].to_numpy(float)


def _build_rows(daily: pd.DataFrame, start: str, end: str, atr14, adxv, pdi, ndi) -> pd.DataFrame:
    nn = len(daily)
    yh = np.roll(daily["High"].values.astype(float), 1)
    yl = np.roll(daily["Low"].values.astype(float), 1)
    yc = np.roll(daily["Close"].values.astype(float), 1)
    yh[0] = np.nan
    yl[0] = np.nan
    yc[0] = np.nan
    rows = pd.DataFrame(
        {
            "Trade_cmd": daily["Date"].iloc[1:].values,
            "Yesterday_High": yh[1:],
            "Yesterday_Low": yl[1:],
            "Yesterday_Close": yc[1:],
            "ATR14": atr14[: nn - 1],
            "ADX14": adxv[: nn - 1],
            "PDI14": pdi[: nn - 1],
            "NDI14": ndi[: nn - 1],
        }
    )
    rows = rows[(rows["Trade_cmd"] >= pd.Timestamp(start)) & (rows["Trade_cmd"] <= pd.Timestamp(end))].copy().reset_index(drop=True)
    warm = rows["ATR14"].notna() & rows["ADX14"].notna() & rows["PDI14"].notna() & rows["NDI14"].notna()
    rows = rows[warm].copy().reset_index(drop=True)
    if len(rows) == 0:
        raise RuntimeError("No usable rows after warmup filter.")
    return rows


def _n_cal(rows: pd.DataFrame) -> int:
    return len(rows["Trade_cmd"].apply(lambda x: pd.Timestamp(x).date()).unique())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pair", choices=["eurjpy", "eurusd", "usdjpy"], default="eurjpy")
    ap.add_argument("--no-hard-tp", action="store_true", help="Disable hard TP for main/addon and remove SAR TP.")
    args = ap.parse_args()
    pair = args.pair
    no_hard_tp = bool(args.no_hard_tp)

    print(f"\n{RUNNER_VERSION}")
    print(f"Pair={pair.upper()}  workers={WORKERS}  no_hard_tp={no_hard_tp}")
    print(f"Split: train {SPLIT['train_start']}..{SPLIT['train_end']} | adjust {SPLIT['adjust_start']}..{SPLIT['adjust_end']} | oos {SPLIT['oos_start']}..{SPLIT['oos_end']}")

    if no_hard_tp:
        # True free-run mode: no TP exits for main/addon; SAR TP removed.
        _pf._main_tp_price = lambda *a, **k: None
        _pf._addon_tp_price = lambda *a, **k: None

    cfg_p = PAIR_CFG[pair]
    pip = cfg_p["pip"]
    cost = cfg_p["cost_pips"]
    daily = _load_daily(Path(cfg_p["daily"]))
    h_ = daily["High"].values.astype(float)
    l_ = daily["Low"].values.astype(float)
    c_ = daily["Close"].values.astype(float)
    atr14 = _atr_series(h_, l_, c_, 14)
    adxv, pdi, ndi = _adx_series(h_, l_, c_, 14)
    times, opens, highs, lows, closes = _load_hourly(Path(cfg_p["h1"]))
    gap_info = check_hourly_gaps(times, pair)
    print(f"Gaps>72h total={gap_info['n_hourly_gaps_gt72h']} allowlisted={gap_info['n_gap_allowlisted']} hardfail={gap_info['n_gap_hardfail']}")

    tr_rows = _build_rows(daily, SPLIT["train_start"], SPLIT["train_end"], atr14, adxv, pdi, ndi)
    ad_rows = _build_rows(daily, SPLIT["adjust_start"], SPLIT["adjust_end"], atr14, adxv, pdi, ndi)
    n_tr_cal = _n_cal(tr_rows)
    n_ad_cal = _n_cal(ad_rows)
    tr_slm = np.where(_trending_mask(tr_rows["ADX14"].values, tr_rows["PDI14"].values, tr_rows["NDI14"].values), 2.5, 2.0)
    ad_slm = np.where(_trending_mask(ad_rows["ADX14"].values, ad_rows["PDI14"].values, ad_rows["NDI14"].values), 2.5, 2.0)
    tr_dict = tr_rows.to_dict("list")
    ad_dict = ad_rows.to_dict("list")
    print(f"Rows: train={len(tr_rows)} (days={n_tr_cal}) adjust={len(ad_rows)} (days={n_ad_cal})")

    parent_cfg = {**F_PARENTS[pair], "config_id": "PARENT", "threshold": "PARENT", "round": 0}
    if no_hard_tp:
        parent_cfg["sar_tp_removed"] = True
    with concurrent.futures.ProcessPoolExecutor(max_workers=WORKERS, initializer=_pool_init_phasef, initargs=(times, opens, highs, lows, closes)) as pool:
        pt = pool.submit(_task_phasef, (tr_dict, tr_slm.tolist(), pip, cost, parent_cfg))
        pa = pool.submit(_task_phasef, (ad_dict, ad_slm.tolist(), pip, cost, parent_cfg))
        parent_tr = _summarize_phasef(pt.result(), n_tr_cal, cost)["cal_ppd"]
        parent_ad = _summarize_phasef(pa.result(), n_ad_cal, cost)["cal_ppd"]
    print(f"Parent (legacy config): train={parent_tr:.4f} adjust={parent_ad:.4f}")

    evaluated_set = set()
    train_hist = []
    winner = None
    winner_hist = []
    all_round_rows = {}
    stop_reason = "rounds_exhausted"

    with concurrent.futures.ProcessPoolExecutor(max_workers=WORKERS, initializer=_pool_init_phasef, initargs=(times, opens, highs, lows, closes)) as pool:
        for rnd in range(1, MAX_ROUNDS + 1):
            if rnd == 1:
                cfgs = gen_round1(pair)
            elif rnd == 2:
                cfgs = gen_round2(pair, winner)
            else:
                cfgs, n_new = gen_round3plus(pair, winner, rnd, evaluated_set)
                if n_new < 4:
                    stop_reason = f"dedup_lt4_new_candidates_at_round_{rnd}"
                    break
            cfgs = [c for c in cfgs if c["threshold"] not in evaluated_set or c.get("_priority") == 0]
            if no_hard_tp:
                for c in cfgs:
                    c["sar_tp_removed"] = True
            evaluated_set |= {c["threshold"] for c in cfgs}

            tasks = []
            labels = []
            for cfg in cfgs:
                tasks.append((tr_dict, tr_slm.tolist(), pip, cost, cfg))
                labels.append(("tr", cfg["threshold"]))
                tasks.append((ad_dict, ad_slm.tolist(), pip, cost, cfg))
                labels.append(("ad", cfg["threshold"]))
            futs = [pool.submit(_task_phasef, t) for t in tasks]
            raw = [f.result() for f in futs]
            res = {(w, k): r for (w, k), r in zip(labels, raw)}

            round_rows = []
            for cfg in cfgs:
                thr = cfg["threshold"]
                tr_s = _summarize_phasef(res[("tr", thr)], n_tr_cal, cost)
                ad_s = _summarize_phasef(res[("ad", thr)], n_ad_cal, cost)
                ed = excess_drift(parent_ad, parent_tr, ad_s["cal_ppd"], tr_s["cal_ppd"])
                round_rows.append({"_cfg": cfg, "_train": tr_s["cal_ppd"], "_adj": ad_s["cal_ppd"], "_excess": ed, "_tr_s": tr_s, "_ad_s": ad_s})

            rw = select_winner(round_rows, F_PARENTS[pair])
            winner = rw["_cfg"]
            all_round_rows[rnd] = round_rows
            tr_ppd = rw["_train"]
            ad_ppd = rw["_adj"]
            train_hist.append(tr_ppd)
            delta = train_hist[-1] - train_hist[-2] if len(train_hist) >= 2 else None
            winner_hist.append(
                {
                    "round": rnd,
                    "n_cfgs": len(cfgs),
                    "winner_threshold": winner["threshold"],
                    "winner_train": round(tr_ppd, 4),
                    "winner_adjust": round(ad_ppd, 4),
                    "delta_train": round(delta, 4) if delta is not None else None,
                }
            )
            print(f"R{rnd:02d} winner tr={tr_ppd:.4f} adj={ad_ppd:.4f}" + (f" d={delta:+.4f}" if delta is not None else ""))

            if rnd >= MIN_ROUNDS and len(train_hist) >= 3:
                d1 = train_hist[-2] - train_hist[-3]
                d2 = train_hist[-1] - train_hist[-2]
                if d1 < EPSILON and d2 < EPSILON:
                    stop_reason = f"delta_lt_{EPSILON}_x2_consec_after_r{MIN_ROUNDS}"
                    break

    if winner is None:
        raise RuntimeError("No winner produced.")

    # Freeze gate
    w_thr = winner["threshold"]
    winner_ad_row = next((r for r in all_round_rows[max(all_round_rows.keys())] if r["_cfg"]["threshold"] == w_thr), None)
    fw_ad_s = winner_ad_row["_ad_s"]
    fw_tr_s = winner_ad_row["_tr_s"]
    fw_adj = fw_ad_s["cal_ppd"]
    fw_train = fw_tr_s["cal_ppd"]
    fw_excess = excess_drift(parent_ad, parent_tr, fw_adj, fw_train)
    gate_pass, gate_checks = check_freeze_gate(pair, fw_adj, parent_ad, fw_excess, fw_ad_s["time_exit_rate"], fw_ad_s["addon_fire_rate"])
    print(f"Freeze gate: {'PASS' if gate_pass else 'FAIL'} checks={gate_checks}")
    if not gate_pass:
        raise RuntimeError("Freeze gate failed.")

    # One-time OOS (in this isolated runner)
    oos_rows = _build_rows(daily, SPLIT["oos_start"], SPLIT["oos_end"], atr14, adxv, pdi, ndi)
    n_oos_cal = _n_cal(oos_rows)
    oos_dict = oos_rows.to_dict("list")
    oos_slm = np.where(_trending_mask(oos_rows["ADX14"].values, oos_rows["PDI14"].values, oos_rows["NDI14"].values), 2.5, 2.0).tolist()
    frozen = winner
    c0_cfg = {**F_PARENTS[pair], "config_id": "C0_OOS", "threshold": "C0_PARENT", "round": 0}
    cbl_cfg = {**C_BL_8H, "config_id": "CBL_OOS", "threshold": "CBL_8H", "round": 0}
    if no_hard_tp:
        frozen["sar_tp_removed"] = True
        c0_cfg["sar_tp_removed"] = True
        cbl_cfg["sar_tp_removed"] = True
    with concurrent.futures.ProcessPoolExecutor(max_workers=WORKERS, initializer=_pool_init_phasef, initargs=(times, opens, highs, lows, closes)) as pool:
        ff = pool.submit(_task_phasef, (oos_dict, oos_slm, pip, cost, frozen))
        fc0 = pool.submit(_task_phasef, (oos_dict, oos_slm, pip, cost, c0_cfg))
        fcbl = pool.submit(_task_phasef, (oos_dict, oos_slm, pip, cost, cbl_cfg))
        oos_s = _summarize_phasef(ff.result(), n_oos_cal, cost)
        c0_s = _summarize_phasef(fc0.result(), n_oos_cal, cost)
        cbl_s = _summarize_phasef(fcbl.result(), n_oos_cal, cost)

    d_c0 = oos_s["cal_ppd"] - c0_s["cal_ppd"]
    d_cbl = oos_s["cal_ppd"] - cbl_s["cal_ppd"]
    oos_gate = check_oos_gate(oos_s["cal_ppd"], c0_s["cal_ppd"], cbl_s["cal_ppd"])
    print(f"OOS: frozen={oos_s['cal_ppd']:.4f} C0={c0_s['cal_ppd']:.4f} CBL={cbl_s['cal_ppd']:.4f} dC0={d_c0:+.4f} dCBL={d_cbl:+.4f} gate={oos_gate}")

    out = {
        "runner_version": RUNNER_VERSION,
        "pair": pair,
        "no_hard_tp": no_hard_tp,
        "split": SPLIT,
        "workers": WORKERS,
        "stop_reason": stop_reason,
        "parent": {"train_cal_ppd": round(parent_tr, 4), "adjust_cal_ppd": round(parent_ad, 4), "cfg": F_PARENTS[pair]},
        "winner": winner,
        "winner_adjust": round(fw_adj, 4),
        "winner_train": round(fw_train, 4),
        "freeze_gate": {"pass": gate_pass, "checks": gate_checks, "excess_drift": round(fw_excess, 4)},
        "oos": {
            "frozen_cal_ppd": round(oos_s["cal_ppd"], 4),
            "c0_cal_ppd": round(c0_s["cal_ppd"], 4),
            "cbl_cal_ppd": round(cbl_s["cal_ppd"], 4),
            "delta_vs_c0": round(d_c0, 4),
            "delta_vs_cbl": round(d_cbl, 4),
            "gate_pass": bool(oos_gate),
        },
        "rounds": winner_hist,
        "gap_info": gap_info,
        "code_fingerprint": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()[:24],
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    out_path = OUTPUT_DIR / f"phasef_legacy_{pair}_2013_2026_result.json"
    out_path.write_text(json.dumps(out, indent=2, default=str))
    print(f"Result written: {out_path}")


if __name__ == "__main__":
    main()

