# -*- coding: utf-8 -*-
"""对微调结果做显著性检验 + 判断"提升是否来自领域适配"（而非测试集泄漏）。

三个检验：
1. paired bootstrap：微调前后逐条命中向量的差值 95% 区间；
2. 泄漏排查：微调模型的训练集 query 与测试集 query 交集断言为 0（再验一次）；
3. 跨域对照：微调只在 DuReader 训练集上做过，若 cmedqa/t2 也涨说明是"通用变好"，
   若只有 dureader 涨说明确实是领域适配（这也是预期结果）。
"""
from __future__ import annotations

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
    data = json.loads((ROOT / DOMAINS[dom] / "testset.json").read_text(encoding="utf-8"))
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


def per_query_hit1(ret, reranker, items, pool=10):
    out = []
    for it in items:
        cands = ret.slow_candidates(it["query"], pool)
        if not cands:
            out.append(0.0)
            continue
        scores = np.asarray(reranker.predict([[it["query"], c["text"]] for c, _ in cands],
                                             show_progress_bar=False), dtype="float32")
        best = cands[int(np.argmax(scores))][0]["chunk_id"]
        out.append(1.0 if best in it["relevant"] else 0.0)
    return out


def main() -> None:
    rc.ensure_utf8_stdout()
    from sentence_transformers import CrossEncoder
    import torch

    tuned = CrossEncoder(str(ROOT / "experiments" / "_rerank_train" / "model"),
                         max_length=512, device="cuda" if torch.cuda.is_available() else "cpu")

    # 泄漏排查：训练数据里的 query 是否出现在测试集中
    train_qs = set()
    with (ROOT / "experiments" / "_rerank_train" / "train.jsonl").open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                train_qs.add(json.loads(line)["query"])
    print(f"训练集 query 数（去重后）: {len(train_qs)}")

    report = {}
    for dom in DOMAINS:
        idx = ROOT / DOMAINS[dom] / "skill_index"
        if not (idx / "coarse.json").exists():
            continue
        ret = rc.Retriever(str(idx))
        known = {c["chunk_id"] for c in ret.coarse}
        items = [it for it in load_items(dom) if it["relevant"] & known]
        overlap = train_qs & {it["query"] for it in items}
        print(f"\n{dom}: 测试 query {len(items)}，与训练集重复 {len(overlap)} 条"
              + ("  ← 有泄漏风险！" if overlap else "  ✓ 无泄漏"))
        base = ret._get_reranker()
        a = per_query_hit1(ret, base, items)
        b = per_query_hit1(ret, tuned, items)
        bs = rc._paired_bootstrap(a, b, rounds=5000)
        report[dom] = {"before": round(float(np.mean(a)), 4), "after": round(float(np.mean(b)), 4),
                       "bootstrap": bs, "query_overlap": len(overlap)}
        print(f"  hit@1 {np.mean(a):.3f} -> {np.mean(b):.3f}  Δ={bs['delta']:+.4f}  "
              f"95%CI [{bs['ci95'][0]:+.4f}, {bs['ci95'][1]:+.4f}]  P(≤0)={bs['p_not_better']:.4f}")

    (ROOT / "experiments" / "_rerank_train" / "significance.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    print("\n已写入 experiments/_rerank_train/significance.json")


if __name__ == "__main__":
    main()
