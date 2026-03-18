"""
_diagnose_be.py - verify BE_TP actually fires and BE_OFF/BE_ON differ.

Key geometry (slm=2.0, smf=0.5, atr=1.0, ao_frac=0.50 of sl_dist):
  sl_dist  = 2.0 * 0.5 * 1.0 = 1.0
  main_sl  = 100.0 - 1.0 = 99.0        (BUY)
  ao_entry = 100.0 - 0.50*1.0 = 99.5   (addon fills if L <= 99.5)
  main_tp  = 100.0 + 1.5*1.0 = 101.5   (TP_ATR_1.5)
  be_tp_px = 100.0                      (BE_ON: exit when H >= 100.0, from bar+1)

For BE_TP to fire:
  1. Addon fills: need L <= 99.5 but L > 99.0 (otherwise SL fires same bar)
  2. be_tp fires next bar: need H >= 100.0 but L > 99.0 (otherwise SL fires)
  3. No TP on that bar (H < 101.5)
  So: need a bar where 99.0 < L and H in [100.0, 101.4]

  For BUY addon fill bar:  L must be in (99.0, 99.5]   <- addon fills, SL not hit
  For BE_TP bar (next bar): H >= 100.0, L > 99.0, H < 101.5
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
import numpy as np
from bl_layer4_defs import _run_one_trade_layer4

pip=0.01; cost=2.0
kwargs = dict(wo=100.0, yh=102.0, yl=98.0, yc=100.0, pp=101.0,
              atr=1.0, yr=4.0, bar_idx=0, pip=pip, cost_pips=cost,
              sf=0.0, slm=2.0, smf=0.5, tpm="TP_ATR_1.5", ao_frac=0.50)

# sl_dist=1.0, ao_entry=99.5, main_sl=99.0, main_tp=101.5, be_tp_px=100.0

def show(label, r):
    print(f"  {label}: reason={r['main_reason']:12s}  "
          f"main={r['main_net']:+8.1f}  addon={r['addon_net']:+8.1f}  "
          f"total={r['total']:+8.1f}  addon_filled={r['addon_filled']}")

def trace_bars(n_show, highs, lows):
    print("  Bar trace (first few):")
    sl_dist=1.0; ao_entry=99.5; main_sl=99.0; be_tp_px=100.0; main_tp=101.5
    addon_filled=False; be_active=False; be_from=999
    for j in range(min(n_show, len(highs))):
        h,l=highs[j],lows[j]
        af=""
        if not addon_filled and l<=ao_entry:
            addon_filled=True; be_active=True; be_from=j+1
            af=" [ADDON FILL, be_from_bar="+str(be_from)+"]"
        be_hit = be_active and j>=be_from and h>=be_tp_px
        sl_hit = l<=main_sl
        tp_hit = h>=main_tp
        evts=[]
        if sl_hit: evts.append("SL")
        if tp_hit: evts.append("TP")
        if be_hit and not sl_hit and not tp_hit: evts.append("BE_TP")
        print(f"    j={j}: H={h:.2f} L={l:.2f}  events={evts or 'none'}{af}")

print("=" * 65)
print("Scenario A: addon fills bar1 (L=99.3), price back to 100.0 bar2,")
print("            then to TP bar3 (BUY: TP=101.5)")
print()
# bar0: entry bar. H=100.2, L=99.6 -> no addon (L=99.6>99.5), no SL, no TP
# bar1: H=100.0, L=99.3 -> addon fills (L=99.3<=99.5, L>99.0), be_from_bar=2
# bar2: H=100.2, L=99.4 -> be_hit (H=100.2>=100.0, j=2>=2, no SL, no TP)
# bar3: H=101.6, L=99.5 -> TP (H>=101.5)
highs = np.array([100.2, 100.0, 100.2, 101.6] + [100.5]*596, dtype=float)
lows  = np.array([99.6,  99.3,  99.4,  99.5]  + [99.5] *596, dtype=float)
closes= np.array([100.5]*600, dtype=float)
trace_bars(5, highs, lows)
r_off = _run_one_trade_layer4(**kwargs, highs=highs, lows=lows, closes=closes, be_on=False)
r_on  = _run_one_trade_layer4(**kwargs, highs=highs, lows=lows, closes=closes, be_on=True)
show("BE_OFF", r_off)
show("BE_ON ", r_on)
print()

print("=" * 65)
print("Scenario B: addon fills bar1 (L=99.3), price back to 100 bar2,")
print("            then DROPS to SL bar3 (L=98.8)")
print()
# Same as A but bar3 drops to SL
highs2 = np.array([100.2, 100.0, 100.2, 99.5] + [100.0]*596, dtype=float)
lows2  = np.array([99.6,  99.3,  99.4,  98.8] + [99.5]*596, dtype=float)
closes2= np.array([100.0]*600, dtype=float)
trace_bars(5, highs2, lows2)
r_off2 = _run_one_trade_layer4(**kwargs, highs=highs2, lows=lows2, closes=closes2, be_on=False)
r_on2  = _run_one_trade_layer4(**kwargs, highs=highs2, lows=lows2, closes=closes2, be_on=True)
show("BE_OFF", r_off2)
show("BE_ON ", r_on2)
print()

print("=" * 65)
print("Scenario C: NO addon fill (L never touches 99.5),")
print("            both go to TP directly")
highs3 = np.array([101.6] + [100.0]*599, dtype=float)
lows3  = np.array([99.6]  + [99.6] *599, dtype=float)
closes3= np.array([100.5]*600, dtype=float)
r_off3 = _run_one_trade_layer4(**kwargs, highs=highs3, lows=lows3, closes=closes3, be_on=False)
r_on3  = _run_one_trade_layer4(**kwargs, highs=highs3, lows=lows3, closes=closes3, be_on=True)
show("BE_OFF", r_off3)
show("BE_ON ", r_on3)

print()
print("=" * 65)
print("Summary:")
print("  Scenario A: BE_OFF=TP (win large), BE_ON=BE_TP (miss TP)?")
print("  Scenario B: BE_OFF=SL (lose large), BE_ON=BE_TP (lose small)?")
print("  The data shows whether BE actually fired as expected.")
