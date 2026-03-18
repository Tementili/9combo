"""Re-add pp=float to _dispatch kwargs (was accidentally removed with yc line)."""
from pathlib import Path
src = Path("bl_engine_v2.py").read_text(encoding="utf-8")
OLD = '               yh=float(row["YH"]), yl=float(row["YL"]),\n               atr=float(row["ATR14"])'
NEW = '               yh=float(row["YH"]), yl=float(row["YL"]),\n               pp=float(row["PP"]),\n               atr=float(row["ATR14"])'
if OLD in src:
    src = src.replace(OLD, NEW, 1)
    Path("bl_engine_v2.py").write_text(src, encoding="utf-8")
    print("pp restored OK.")
else:
    # Show context
    for i, ln in enumerate(src.splitlines(), 1):
        if "YH" in ln or "YL" in ln or "ATR14" in ln or "pp=float" in ln:
            if "dispatch" in "".join(src.splitlines()[max(0,i-5):i+5]):
                pass
            print(f"  {i}: {ln!r}")
import py_compile, ast
py_compile.compile("bl_engine_v2.py", doraise=True)
ast.parse(Path("bl_engine_v2.py").read_text(encoding="utf-8"))
print("Syntax OK.")
