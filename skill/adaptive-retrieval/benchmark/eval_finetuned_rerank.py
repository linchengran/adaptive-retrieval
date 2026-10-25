# -*- coding: utf-8 -*-
"""对比微调前后重排器在**测试集**上的 hit@1（严格隔离验证）。

评测口径与 retrieval_core 的 slow path 一致：候选池 = 第一跳 top-10 → 重排。
同时报告"完美重排上界"（候选池 recall@10），这样能看出微调吃掉多少可救回空间。

用法：python benchmark/eval_finetuned_rerank.py [--domain m5_dureader] [--model <dir>]
"""
from __future__ import annotations

import argparse
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
    "m7_cmrc2018": "experiments/m7_cmrc2018",
}


def load_items(dom: str) -> list:
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


def score_with(ret, reranker, items, pool=10):
    hit1 = hit5 = 0
    poolrec = 0
    nd = 0.0
    for it in items:
        cands = ret.slow_candidates(it["query"], pool)
        if not cands:
            continue
        ids = [c["chunk_id"] for c, _ in cands]
        if it["relevant"] & set(ids):
            poolrec += 1
        scores = np.asarray(reranker.predict([[it["query"], c["text"]] for c, _ in cands],
                                             show_progress_bar=False), dtype="float32")
        order = np.argsort(scores)[::-1]
        ranked = [ids[i] for i in order][:5]
        if ranked and ranked[0] in it["relevant"]:
            hit1 += 1
        if it["relevant"] & set(ranked[:5]):
            hit5 += 1
        dcg = sum(1.0 / np.log2(r + 1) for r, cid in enumerate(ranked[:5], 1)
                  if cid in it["relevant"])
        ideal = sum(1.0 / np.log2(r + 1) for r in range(1, min(len(it["relevant"]), 5) + 1))
        nd += dcg / ideal if ideal else 0.0
    n = len(items)
    return {"hit@1": round(hit1 / n, 4), "hit@5": round(hit5 / n, 4),
            "ndcg@5": round(nd / n, 4), "pool_recall@10": round(poolrec / n, 4), "n": n}


def main() -> None:
    rc.ensure_utf8_stdout()
    ap = argparse.ArgumentParser()
    ap.add_argument("--domains", nargs="*", default=list(DOMAINS))
    ap.add_argument("--model", default=str(ROOT / "experiments" / "_rerank_train" / "model"))
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    from sentence_transformers import CrossEncoder
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"

    tuned_dir = Path(a.model)
    if not (tuned_dir / "config.json").exists():
        raise SystemExit(f"找不到微调模型: {tuned_dir}（先跑 finetune_reranker.py）")
    print(f"微调模型: {tuned_dir}")
    tuned = CrossEncoder(str(tuned_dir), max_length=512, device=device)

    report = {}
    for dom in a.domains:
        idx = ROOT / DOMAINS[dom] / "skill_index"
        if not (idx / "coarse.json").exists():
            print(f"{dom}: 跳过（无 skill_index）")
            continue
        ret = rc.Retriever(str(idx))
        known = {c["chunk_id"] for c in ret.coarse}
        items = [it for it in load_items(dom) if it["relevant"] & known]
        base = ret._get_reranker()
        before = score_with(ret, base, items)
        after = score_with(ret, tuned, items)
        report[dom] = {"before": before, "after": after,
                       "delta_hit@1": round(after["hit@1"] - before["hit@1"], 4)}
        print(f"\n== {dom}（n={before['n']}, 候选池 recall@10={before['pool_recall@10']:.3f}）")
        print(f"  微调前  hit@1={before['hit@1']:.3f} hit@5={before['hit@5']:.3f} "
              f"ndcg@5={before['ndcg@5']:.3f}")
        print(f"  微调后  hit@1={after['hit@1']:.3f} hit@5={after['hit@5']:.3f} "
              f"ndcg@5={after['ndcg@5']:.3f}   Δhit@1={report[dom]['delta_hit@1']:+.3f}")

    if a.out:
        Path(a.out).write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"\n已写入 {a.out}")


if __name__ == "__main__":
    main()
