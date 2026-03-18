"""Print artifact inventory and validation status."""
from pathlib import Path
import json
import pandas as pd

print("=== ARTIFACT FILES ===")
for p in sorted(Path(".").glob("layer*.csv")):
    sz = p.stat().st_size // 1024
    try:
        n = len(pd.read_csv(p))
    except Exception:
        n = "?"
    print(f"  {p.name:<58} {sz:5d}KB  {n} rows")

print()
for p in sorted(Path(".").glob("layer*frozen*.json")):
    sz = p.stat().st_size
    d = json.loads(p.read_text())
    cfg = d.get("frozen_cfg", {})
    print(f"  {p.name:<58} frozen={cfg}")

print()
print("=== VALIDATION ===")
for layer in ["layer1", "layer2", "layer3", "layer4"]:
    vf = Path(f"validation_checks_{layer}.json")
    if not vf.exists():
        print(f"  {layer}: MISSING")
        continue
    d = json.loads(vf.read_text())
    checks = d["checks"]
    n_pass = 0; n_fail = 0
    for pair, pchks in checks.items():
        if not isinstance(pchks, dict):
            continue
        for k, v in pchks.items():
            if isinstance(v, dict) and "pass" in v:
                if v["pass"]:
                    n_pass += 1
                else:
                    n_fail += 1
                    print(f"    FAIL [{layer}/{pair}/{k}]: {v.get('detail','')}")
    status = "ALL PASS" if n_fail == 0 else f"{n_fail} FAIL"
    print(f"  {layer}: {status}  ({n_pass} checks)")

print()
print("=== OOS LOCKS ===")
for p in sorted(Path(".").glob("layer*_oos_lock_*.json")):
    d = json.loads(p.read_text())
    print(f"  {p.name}  ts={d.get('timestamp','?')[:16]}")
