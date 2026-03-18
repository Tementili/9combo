from pathlib import Path
import pandas as pd
import glob
import os

BASE = Path(r"C:\Users\temen\5 joulu\TrailingStopLoss pivot peak trading 6 march 2026\ATR 14 only mode\part 4\part 4 - For ATR 14 mode use\baseline_discovery_v1")
out_lines = []
files = sorted(glob.glob(str(BASE / "*ledger*.csv")) + glob.glob(str(BASE / "*results*.csv")))
combined = []
for f in files:
    try:
        df = pd.read_csv(f)
    except Exception:
        continue
    if "net_pips" in df.columns:
        pos = df[df["net_pips"] > 0]
        for _, r in pos.iterrows():
            combined.append({
                "source": os.path.basename(f),
                "date": r.get("date", ""),
                "entry_px": r.get("entry_px", ""),
                "exit_px": r.get("exit_px", ""),
                "net_pips": r.get("net_pips", "")
            })

out_path = BASE / "positive_trades_combined.csv"
pd.DataFrame(combined).to_csv(out_path, index=False)
print("Wrote {} positive trades to {}".format(len(combined), out_path))
for row in combined:
    print(str(row.get("source","")) + "," + str(row.get("date","")) + "," + str(row.get("entry_px","")) + "," + str(row.get("exit_px","")) + "," + str(row.get("net_pips","")))

