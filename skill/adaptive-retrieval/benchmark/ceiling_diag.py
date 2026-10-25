# -*- coding: utf-8 -*-
"""诊断：天花板到底在哪一环？

分解 hit@1 的损失：
  A 第一跳命中率          —— 召回问题
  B 候选池命中率 pool@N    —— 召回问题（扩池能救）
  C 重排后命中率 hit@1     —— 排序问题
  D 完美重排上界 = pool@N  —— 排序问题（换/调重排能救）
若 A≈B，说明召回已饱和、扩池与查询改写无用，只能改重排。
"""
import ast
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _paths import REPO_ROOT as ROOT, SKILL_DIR  # noqa: E402
sys.path.insert(0, str(SKILL_DIR / "scripts"))
import retrieval_core as rc  # noqa: E402

DOMAINS = {
    "m5_dureader": "experiments/m5_dureader",
    "m7_cmrc2018": "experiments/m7_cmrc2018",
    "m7_t2retrieval": "experiments/m7_t2retrieval",
    "m7_cmedqa": "experiments/m7_cmedqa",
}
POOLS = (5, 10, 20, 50)


def load_items(dom):
    ts = ROOT / DOMAINS[dom] / "testset.json"
    data = json.loads(ts.read_text(encoding="utf-8"))
    items = data["items"] if isinstance(data, dict) else data
    out = []
    for it in items:
        rel = it.get("relevant_chunks") or it.get("relevant") or []
        if isinstance(rel, str):
            rel = ast.literal_eval(rel)
        q = it.get("query") or it.get("question")
        if q and rel:
            out.append({"query": q, "relevant": set(map(str, rel))})
    return out


def main():
    rc.ensure_utf8_stdout()
    for dom, rel_dir in DOMAINS.items():
        idx = ROOT / rel_dir / "skill_index"
        ret = rc.Retriever(str(idx))
        known = {c["chunk_id"] for c in ret.coarse}
        items = [it for it in load_items(dom) if it["relevant"] & known]
        n = len(items)
        print(f"\n== {dom} ({ret.corpus_type}, {len(ret.coarse)} chunks, n={n})")

        first1 = np.mean([1.0 if it["relevant"] & {c["chunk_id"] for c, _ in
                          ret.first_pass(it["query"], 5)} else 0.0 for it in items])
        print(f"  A 第一跳 top5 命中率            {first1:.3f}")

        for pool in POOLS:
            poolrec = []
            rerank1 = []
            for it in items:
                cands = ret.slow_candidates(it["query"], pool)
                ids = {c["chunk_id"] for c, _ in cands}
                poolrec.append(1.0 if it["relevant"] & ids else 0.0)
                rr = ret.rerank(it["query"], cands, 1)
                rerank1.append(1.0 if rr and rr[0][0]["chunk_id"] in it["relevant"]
                               else 0.0)
            pr, r1 = np.mean(poolrec), np.mean(rerank1)
            print(f"  pool={pool:<3d} B 候选池命中 {pr:.3f}   C 重排后 hit@1 {r1:.3f}   "
                  f"D 可救回 {pr - r1:+.3f}   成本 {pool} 对")


if __name__ == "__main__":
    main()
