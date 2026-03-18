#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

PHASEF_DIR = Path(
    r"C:\Users\temen\5 joulu\TrailingStopLoss pivot peak trading 6 march 2026\ATR 14 only mode\part 4\part 4 - For ATR 14 mode use\phase_f_honing_master"
)
import sys
sys.path.insert(0, str(PHASEF_DIR))
import phase_f_defs as pf  # type: ignore


PAIR = "usdjpy"
TRAIN_START, TRAIN_END = "2013-01-01", "2016-12-31"
ADJ_START, ADJ_END = "2017-01-01", "2022-12-31"
OOS_START, OOS_END = "2023-01-01", "2026-03-06"

OUT = Path(__file__).parent / "usdjpy_no_tp_trailing_search_2013_2026.json"


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
    return (
        m["Datetime"].to_numpy(),
        m["Open"].to_numpy(float),
        m["High"].to_numpy(float),
        m["Low"].to_numpy(float),
        m["Close"].to_numpy(float),
    )


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
        raise RuntimeError(f"No rows in {start}..{end}")
    return rows


def _n_cal(rows: pd.DataFrame) -> int:
    return len(rows["Trade_cmd"].apply(lambda x: pd.Timestamp(x).date()).unique())


def main() -> None:
    # Force true no-TP behavior in this process.
    pf._main_tp_price = lambda *a, **k: None
    pf._addon_tp_price = lambda *a, **k: None

    cfgp = pf.PAIR_CFG[PAIR]
    pip = cfgp["pip"]
    cost = cfgp["cost_pips"]

    daily = _load_daily(Path(cfgp["daily"]))
    h_ = daily["High"].values.astype(float)
    l_ = daily["Low"].values.astype(float)
    c_ = daily["Close"].values.astype(float)
    atr14 = pf._atr_series(h_, l_, c_, 14)
    adxv, pdi, ndi = pf._adx_series(h_, l_, c_, 14)
    times, opens, highs, lows, closes = _load_hourly(Path(cfgp["h1"]))

    tr_rows = _build_rows(daily, TRAIN_START, TRAIN_END, atr14, adxv, pdi, ndi)
    ad_rows = _build_rows(daily, ADJ_START, ADJ_END, atr14, adxv, pdi, ndi)
    oos_rows = _build_rows(daily, OOS_START, OOS_END, atr14, adxv, pdi, ndi)

    tr_slm = np.where(pf._trending_mask(tr_rows["ADX14"].values, tr_rows["PDI14"].values, tr_rows["NDI14"].values), 2.5, 2.0)
    ad_slm = np.where(pf._trending_mask(ad_rows["ADX14"].values, ad_rows["PDI14"].values, ad_rows["NDI14"].values), 2.5, 2.0)
    oos_slm = np.where(pf._trending_mask(oos_rows["ADX14"].values, oos_rows["PDI14"].values, oos_rows["NDI14"].values), 2.5, 2.0)

    n_tr = _n_cal(tr_rows)
    n_ad = _n_cal(ad_rows)
    n_oos = _n_cal(oos_rows)

    parent = dict(pf.F_PARENTS[PAIR])
    parent["sar_tp_removed"] = True

    # tiny fast grid (72 configs) for immediate guidance
    sar_trail_k_grid = [0.10, 0.20, 0.30]
    sar_trail_act_k_grid = [0.00, 0.10]
    sar_sl_frac_grid = [0.35, 0.70]
    addon_sl_scale_grid = [0.90, 1.10]
    skip_frac_grid = [0.25, 0.35, 0.40]

    tested = []
    best = None
    best_key = None

    total = (
        len(sar_trail_k_grid)
        * len(sar_trail_act_k_grid)
        * len(sar_sl_frac_grid)
        * len(addon_sl_scale_grid)
        * len(skip_frac_grid)
    )
    done = 0
    print(f"USDJPY no-TP trailing search: {total} configs")

    for tk in sar_trail_k_grid:
        for tak in sar_trail_act_k_grid:
            for sslf in sar_sl_frac_grid:
                for adssl in addon_sl_scale_grid:
                    for sf in skip_frac_grid:
                        cfg = dict(parent)
                        cfg.update(
                            {
                                "sar_trail_k": tk,
                                "sar_trail_act_k": tak,
                                "sar_sl_frac": sslf,
                                "addon_sl_scale": adssl,
                                "skip_frac": sf,
                                "sar_tp_removed": True,
                                "config_id": f"noTP_tk{tk}_tak{tak}_sslf{sslf}_adssl{adssl}_sf{sf}",
                                "threshold": f"noTP|tk={tk}|tak={tak}|sslf={sslf}|adssl={adssl}|sf={sf}",
                                "round": 0,
                            }
                        )

                        tr_res = pf._run_backtest_phasef(tr_rows, times, opens, highs, lows, closes, tr_slm, pip, cost, cfg)
                        ad_res = pf._run_backtest_phasef(ad_rows, times, opens, highs, lows, closes, ad_slm, pip, cost, cfg)
                        tr_s = pf._summarize_phasef(tr_res, n_tr, cost)
                        ad_s = pf._summarize_phasef(ad_res, n_ad, cost)

                        row = {
                            "cfg": {
                                "sar_trail_k": tk,
                                "sar_trail_act_k": tak,
                                "sar_sl_frac": sslf,
                                "addon_sl_scale": adssl,
                                "skip_frac": sf,
                            },
                            "train_cal_ppd": tr_s["cal_ppd"],
                            "adjust_cal_ppd": ad_s["cal_ppd"],
                            "adjust_time_exit_rate": ad_s["time_exit_rate"],
                            "adjust_sar_fire_rate": ad_s["sar_fire_rate"],
                            "adjust_addon_fire_rate": ad_s["addon_fire_rate"],
                        }
                        tested.append(row)

                        # Objective: maximize Adjust; tie-break on Train then lower TER.
                        key = (ad_s["cal_ppd"], tr_s["cal_ppd"], -ad_s["time_exit_rate"])
                        if best_key is None or key > best_key:
                            best_key = key
                            best = row

                        done += 1
                        if done % 20 == 0:
                            print(f"  progress {done}/{total} best_adj={best['adjust_cal_ppd']:+.4f}", flush=True)

    assert best is not None

    # One-shot OOS for best config.
    cfg = dict(parent)
    cfg.update(best["cfg"])
    cfg["sar_tp_removed"] = True
    cfg["config_id"] = "BEST_noTP"
    cfg["threshold"] = "BEST_noTP"
    cfg["round"] = 0
    oos_res = pf._run_backtest_phasef(oos_rows, times, opens, highs, lows, closes, oos_slm, pip, cost, cfg)
    oos_s = pf._summarize_phasef(oos_res, n_oos, cost)

    # Keep top 20 by adjust.
    tested_sorted = sorted(
        tested,
        key=lambda r: (r["adjust_cal_ppd"], r["train_cal_ppd"], -r["adjust_time_exit_rate"]),
        reverse=True,
    )
    top20 = tested_sorted[:20]

    out = {
        "pair": PAIR,
        "mode": "NO_HARD_TP_TRAILING_ONLY",
        "split": {
            "train": [TRAIN_START, TRAIN_END],
            "adjust": [ADJ_START, ADJ_END],
            "oos": [OOS_START, OOS_END],
        },
        "configs_tested": total,
        "best": best,
        "best_oos": {
            "oos_cal_ppd": oos_s["cal_ppd"],
            "oos_time_exit_rate": oos_s["time_exit_rate"],
            "oos_sar_fire_rate": oos_s["sar_fire_rate"],
            "oos_addon_fire_rate": oos_s["addon_fire_rate"],
        },
        "top20": top20,
    }
    OUT.write_text(json.dumps(out, indent=2))

    print("\nBEST (Train/Adjust):")
    print(best)
    print("\nBEST OOS:")
    print(out["best_oos"])
    print(f"\nSaved: {OUT}")


if __name__ == "__main__":
    main()

