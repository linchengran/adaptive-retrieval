# -*- coding: utf-8 -*-
"""读取 bakeoff JSON 并打印。用法：python benchmark/render_bakeoff.py <json路径>"""
import json
import os
import sys
from pathlib import Path

p = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(
    r"D:\claude code use\better research skill\experiments\_opt\reranker_bakeoff.json")
if not p.exists():
    print("pending: 文件尚未生成", p)
    raise SystemExit(0)
d = json.loads(p.read_text(encoding="utf-8"))
models = []
for v in d.values():
    for m in v:
        if m not in models:
            models.append(m)

for metric in ("hit@1", "ndcg@5", "p50_ms"):
    print(f"\n== {metric} ==")
    print(f"{'domain':16s} " + " ".join(f"{m:>14s}" for m in models))
    for dom, v in d.items():
        cells = []
        vals = {m: v.get(m, {}).get(metric) for m in models}
        numeric = [x for x in vals.values() if x is not None]
        best_high = max(numeric) if numeric else 0
        best_low = min(numeric) if numeric else 0
        for m in models:
            val = vals[m]
            if val is None:
                cells.append(f"{'-':>14s}")
                continue
            target = best_low if metric == "p50_ms" else best_high
            mark = "*" if val == target else " "
            cells.append(f"{val:>13.3f}{mark}" if isinstance(val, float) else f"{val:>13}{mark}")
        print(f"{dom:16s} " + " ".join(cells))
