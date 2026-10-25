# -*- coding: utf-8 -*-
"""对比不同重排器底座在四域上的效果（候选池已饱和，纯看排序能力）。

对比对象：
- bge-reranker-base（现状默认，278M）
- 微调后的 base（experiments/_rerank_train/model）
- bge-reranker-v2-m3（568M）
- v2-m3 + 同上训练集微调（可选，--finetune）

用法：python benchmark/reranker_bakeoff.py [--include-finetune-v2m3]
"""
from __future__ import annotations

import argparse
import ast
import json
import sys
import time
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


def build_cache(ret, items, pool=10):
    """候选池只算一次，所有重排器共用。原来每个模型都重算召回，是真正的瓶颈。"""
    cached = []
    for it in items:
        cands = ret.slow_candidates(it["query"], pool)
        if cands:
            cached.append({"it": it,
                           "texts": [c["text"] for c, _ in cands],
                           "ids": [c["chunk_id"] for c, _ in cands]})
    return cached


def evaluate(cached, reranker):
    h1 = h5 = 0
    nd = 0.0
    lat = []
    for row in cached:
        it, texts, ids = row["it"], row["texts"], row["ids"]
        t0 = time.perf_counter()
        scores = np.asarray(reranker.predict([[it["query"], t] for t in texts],
                                             show_progress_bar=False), dtype="float32")
        lat.append((time.perf_counter() - t0) * 1000)
        top = [ids[i] for i in np.argsort(scores)[::-1]][:5]
        if top and top[0] in it["relevant"]:
            h1 += 1
        if it["relevant"] & set(top):
            h5 += 1
        dcg = sum(1.0 / np.log2(r + 1) for r, c in enumerate(top, 1) if c in it["relevant"])
        ideal = sum(1.0 / np.log2(r + 1) for r in range(1, min(len(it["relevant"]), 5) + 1))
        nd += dcg / ideal if ideal else 0.0
    n = len(cached)
    if not n:
        return {}
    lat.sort()
    return {"hit@1": round(h1 / n, 4), "hit@5": round(h5 / n, 4),
            "ndcg@5": round(nd / n, 4), "p50_ms": round(lat[len(lat) // 2], 1), "n": n}


def main() -> None:
    rc.ensure_utf8_stdout()
    ap = argparse.ArgumentParser()
    ap.add_argument("--domains", nargs="*", default=list(DOMAINS))
    ap.add_argument("--include-finetune-v2m3", action="store_true")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    from sentence_transformers import CrossEncoder
    import torch
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    models = {
        "base(默认)": str(ROOT / "models" / "bge-reranker-base"),
        "base+微调": str(ROOT / "experiments" / "_rerank_train" / "model"),
        "v2-m3": str(ROOT / "models" / "bge-reranker-v2-m3"),
    }
    if a.include_finetune_v2m3:
        p = ROOT / "experiments" / "_rerank_train" / "v2m3_model2"
        if (p / "config.json").exists():
            models["v2-m3+微调"] = str(p)

    loaded = {}
    for name, path in models.items():
        if not (Path(path) / "config.json").exists():
            print(f"跳过 {name}：{path} 不存在")
            continue
        loaded[name] = CrossEncoder(path, max_length=512, device=dev, num_labels=1)

    report = {}
    for dom in a.domains:
        idx = ROOT / DOMAINS[dom] / "skill_index"
        if not (idx / "coarse.json").exists():
            continue
        ret = rc.Retriever(str(idx))
        known = {c["chunk_id"] for c in ret.coarse}
        items = [it for it in load_items(dom) if it["relevant"] & known]
        report[dom] = {}
        print(f"\n== {dom}（n={len(items)}）")
        print(f"  {'重排器':14s} {'hit@1':>7s} {'hit@5':>7s} {'ndcg@5':>7s} {'p50ms':>8s}")
        cached = build_cache(ret, items, pool=10)
        for name, rr in loaded.items():
            r = evaluate(cached, rr)
            report[dom][name] = r
            print(f"  {name:14s} {r['hit@1']:7.3f} {r['hit@5']:7.3f} "
                  f"{r['ndcg@5']:7.3f} {r['p50_ms']:8.1f}")

    if a.out:
        Path(a.out).write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"\n已写入 {a.out}")


if __name__ == "__main__":
    main()
