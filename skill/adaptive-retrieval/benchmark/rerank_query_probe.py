# -*- coding: utf-8 -*-
"""重排侧实验：候选池已饱和（pool_recall@10≈1.0），唯一可能是让重排排得更准。

假设：cross-encoder 读"带大量疑问虚词的原查询"时会被稀释。
测四种输入方式：
  A 原查询（现状）
  B 实词化查询（去疑问/虚词）
  C 原查询 + 实词（拼接）
  D A/B 两次打分取平均（一致性投票）
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
    "m7_cmedqa": "experiments/m7_cmedqa",
    "m7_t2retrieval": "experiments/m7_t2retrieval",
}


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


def content_query(ret, q):
    v = rc.Retriever.expand_queries(q, ret.tokenize, max_variants=1)
    return v[0] if v else q


def main():
    rc.ensure_utf8_stdout()
    for dom, rel_dir in DOMAINS.items():
        idx = ROOT / rel_dir / "skill_index"
        ret = rc.Retriever(str(idx))
        known = {c["chunk_id"] for c in ret.coarse}
        items = [it for it in load_items(dom) if it["relevant"] & known]
        rr = ret._get_reranker()
        n = len(items)
        rel_counts = [len(it["relevant"]) for it in items]
        print(f"\n== {dom} ({ret.corpus_type}, n={n}, "
              f"平均相关块 {np.mean(rel_counts):.2f})")

        acc = {"A 原查询": 0, "B 实词化": 0, "C 拼接": 0, "D 双打分平均": 0, "E 三路平均": 0}
        for it in items:
            q = it["query"]
            cands = ret.slow_candidates(q, 10)
            if not cands:
                continue
            texts = [c["text"] for c, _ in cands]
            ids = [c["chunk_id"] for c, _ in cands]
            cq = content_query(ret, q)

            s_a = np.asarray(rr.predict([[q, t] for t in texts], show_progress_bar=False))
            s_b = np.asarray(rr.predict([[cq, t] for t in texts], show_progress_bar=False)) \
                if cq != q else s_a
            s_c = np.asarray(rr.predict([[(q + " " + cq), t] for t in texts],
                                        show_progress_bar=False)) if cq != q else s_a
            s_d = (s_a + s_b) / 2

            s_e = (s_a + s_b + s_c) / 3
            for key, s in (("A 原查询", s_a), ("B 实词化", s_b),
                           ("C 拼接", s_c), ("D 双打分平均", s_d), ("E 三路平均", s_e)):
                if ids[int(np.argmax(s))] in it["relevant"]:
                    acc[key] += 1

        for key, v in acc.items():
            print(f"  {key:14s} hit@1 = {v / n:.3f}")


if __name__ == "__main__":
    main()
