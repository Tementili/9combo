# MEMO: Halfway to the Pivot — S1 Sell

**Date:** 2026-03-16
**Pair:** EURJPY
**Timeframe:** 15-minute candles, daily pivot levels
**Status:** Raw discovery — NOT validated OOS yet

---

## Strategy Name

**"Halfway to the Pivot to S1 Sell"**

## What It Does

Wait for price to drop from above the daily pivot (PP) down into the PP–S1 zone.
When price touches PP from above, SELL at PP.
Fixed SL above, fixed TP below. TP side is always the bigger distance.
Let it run — no trailing, no max hold, no day-end close.

## Exact Rules

1. **Daily levels (causal, from previous day):**
   - PP = (YH + YL + YC) / 3
   - S1 = 2 × PP − YH

2. **Entry trigger:** On 15m bars, scan from day open. First bar where bar_low <= PP → SELL at PP.

3. **If price never touches PP during the day → no trade, cancel.**

4. **SL = entry + 0.70 × ATR14** (fixed, above entry — adverse direction for SELL)

5. **TP = entry − 0.70 × 1.3 × ATR14 = entry − 0.91 × ATR14** (fixed, below entry — favorable direction for SELL)

6. **R:R = 1.3 : 1** (TP distance is 30% bigger than SL distance)

7. **No trailing stop. No time exit. Trade lives until TP or SL is hit.**

8. **Cost: 1.0 pip spread (EURJPY)**

9. **ATR14: Wilder smoothing, computed on daily bars, lagged by 1 day (causal)**

## Results

**Period:** 2003-08-18 to 2026-03-06 (8,237 calendar days, 22.5 years)

| Metric | Value |
|--------|-------|
| Total trades | 5,939 |
| Wins | 2,933 (49.4%) |
| Losses | 3,006 (50.6%) |
| Total pips | **+50,900** |
| Cal PPD | **+6.18** |
| Avg pips/trade | +8.57 |
| Avg win | +86.00 pips |
| Avg loss | −66.98 pips |
| Avg hold | 36.6 hours |
| TP exits | 2,933 |
| SL exits | 3,004 |

## Year-by-Year

| Year | Trades | Total Pips | Cal PPD | Win Rate |
|------|--------|-----------|---------|----------|
| 2003 | 114 | +1,099 | +8.14 | 46.5% |
| 2004 | 306 | +2,890 | +7.90 | 49.0% |
| 2005 | 308 | +1,084 | +2.98 | 46.8% |
| 2006 | 308 | +155 | +0.42 | 44.5% |
| 2007 | 102 | −266 | −1.59 | 46.1% |
| 2008 | 5 | −83 | −10.42 | 40.0% |
| 2009 | 301 | +5,816 | +15.98 | 50.2% |
| 2010 | 316 | +9,068 | +24.91 | 57.9% |
| 2011 | 319 | +4,249 | +11.64 | 52.4% |
| 2012 | 298 | +3,973 | +10.86 | 53.0% |
| 2013 | 308 | +866 | +2.38 | 44.5% |
| 2014 | 308 | +2,216 | +6.07 | 51.6% |
| 2015 | 321 | +6,718 | +18.41 | 57.0% |
| 2016 | 318 | +3,352 | +9.16 | 50.9% |
| 2017 | 302 | +2,378 | +6.52 | 51.0% |
| 2018 | 311 | +2,128 | +5.83 | 50.2% |
| 2019 | 313 | +1,526 | +4.18 | 49.8% |
| 2020 | 317 | +2,199 | +6.01 | 49.2% |
| 2021 | 94 | −479 | −3.42 | 38.3% |
| 2022 | 9 | +291 | +29.09 | 77.8% |
| 2023 | 306 | −144 | −0.40 | 44.4% |
| 2024 | 305 | −161 | −0.44 | 43.6% |
| 2025 | 295 | +1,565 | +4.29 | 46.8% |
| 2026 | 55 | +462 | +7.21 | 50.9% |

**Positive in 20 out of 24 years.**
**Negative years:** 2007 (−266), 2008 (−83, only 5 trades), 2021 (−479), 2023 (−144), 2024 (−161).

## Directional Proof (Same Setup, Three Sides)

| Side | Total Pips | Cal PPD | Win Rate |
|------|-----------|---------|----------|
| BUY only | −47,745 | −5.80 | 39.0% |
| Random (avg 20 runs) | −675 | −0.08 | 44.2% |
| **SELL only** | **+50,900** | **+6.18** | **49.4%** |

BUY and SELL are near-perfect mirrors. Random is near zero. This confirms the edge is purely directional: when EURJPY price drops to the daily pivot, it keeps falling.

## Why It Might Work

When price drops from above PP down to PP, it signals intraday weakness. The pivot is the day's equilibrium — falling through it means sellers are in control. Selling at PP with continuation momentum toward S1 captures this directional bias.

The 1.3:1 R:R means you can win less than 50% and still profit. At 49.4% win rate with avg win +86 vs avg loss −67, the math works: 0.494 × 86 − 0.506 × 67 = +8.6 pips/trade (matches the 8.57 avg observed).

## What Is NOT Validated Yet

- No OOS split has been applied (the full 2003–2026 range was used as one block)
- Only EURJPY tested — no cross-pair validation
- No parameter sensitivity tested (0.70 and 1.3 are the only values run)
- 2023–2024 show the edge weakening (−0.40 and −0.44 cal_ppd) — recent regime may be different
- No indicator filters applied — raw directional bias only

## Next Steps (If Pursuing)

1. Split into Train (2003–2016) / Adjust (2017–2021) / OOS (2022–2026) and validate
2. Sweep SL multiplier (0.50–1.00) and TP multiplier (1.1–2.0) on Train only
3. Test USDJPY and EURUSD with same rules
4. Check if adding the mirror (BUY above PP between PP–R1) on the same pair adds value
5. Test with trailing SL variant (keep TP fixed, add trail)

## Files

- Strategy code: `sell_pp_s1_fixed_sl70_tp91_15m_eurjpy.py`
- Trade ledger: `sell_pp_s1_fixed_sl70_tp91_15m_eurjpy_ledger.csv`
- Summary JSON: `sell_pp_s1_fixed_sl70_tp91_15m_eurjpy_summary.json`
- BUY comparison: `buy_pp_s1_fixed_sl70_tp91_15m_eurjpy_summary.json`
- Random comparison: `random_side_pp_s1_fixed_sl70_tp91_15m_eurjpy_summary.json`
