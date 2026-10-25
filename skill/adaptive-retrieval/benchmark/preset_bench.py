# -*- coding: utf-8 -*-
"""策略预设四域对比 + 天花板分析（回答"multi-query 到底有没有把可救回率拉上去"）。

用法：
    python benchmark/preset_bench.py                 # 全部域、默认预设
    python benchmark/preset_bench.py --domains m7_t2retrieval m5_dureader
    python benchmark/preset_bench.py --presets base,multiquery,mq_rrf

输出：
1. 每域 × 每预设的 auto 档 hit@1 / hit@5 / ndcg@5 / p50
2. 相对 base 的配对 bootstrap（95% 区间 + P(不更好)）
3. **天花板分析**：第一跳漏掉、但候选池里其实有的比例（= 重排最多能救回多少）；
   这个值提升，才说明改造方向对了
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
    "m7_cmrc2018": "experiments/m7_cmrc2018",
    "m7_t2retrieval": "experiments/m7_t2retrieval",
    "m7_cmedqa": "experiments/m7_cmedqa",
    "wx": "experiments/m6_corpus_aware",
}
WX_TS = Path(__file__).resolve().parent / "data" / "testset_v2.json"


def load_items(dom: str) -> list:
    base = ROOT / DOMAINS[dom]
    ts = base / "testset.json" if (base / "testset.json").exists() else WX_TS
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


def index_dir(dom: str) -> Path:
    base = ROOT / DOMAINS[dom]
    return base / ("skill_index" if (base / "skill_index").exists() else "wx_skill_index")


def ndcg(ids, rel, k=5):
    dcg = sum(1.0 / np.log2(r + 1) for r, c in enumerate(ids[:k], 1) if c in rel)
    ideal = sum(1.0 / np.log2(r + 1) for r in range(1, min(len(rel), k) + 1))
    return dcg / ideal if ideal else 0.0


def run_domain(dom: str, presets: list) -> dict:
    idx = index_dir(dom)
    if not (idx / "coarse.json").exists():
        return {}
    items = load_items(dom)
    known = {c["chunk_id"] for c in json.loads(
        (idx / "coarse.json").read_text(encoding="utf-8"))}
    items = [it for it in items if it["relevant"] & known]
    if not items:
        return {}

    out = {"n": len(items), "presets": {}}
    for name in presets:
        ret = rc.Retriever(str(idx), options=rc.PRESETS.get(name, {}))
        per_q, lat, pooled = [], [], []
        for it in items:
            a = time.perf_counter()
            res = ret.retrieve(it["query"], k=5, profile="auto")
            lat.append((time.perf_counter() - a) * 1000)
            ids = [c["chunk_id"] for c in res]
            per_q.append(1.0 if it["relevant"] & set(ids[:1]) else 0.0)
            pooled.append(1.0 if it["relevant"] & set(ids) else 0.0)

        # 天花板：第一跳漏掉、但慢路径候选池里确实有正例的比例
        first, pool_hit = [], []
        long = ret.corpus_type == "long"
        for it in items:
            f = ret.first_pass(it["query"], 5)
            if it["relevant"] & {c["chunk_id"] for c, _ in f}:
                first.append(1.0)
                pool_hit.append(1.0)
                continue
            first.append(0.0)
            cands = ret.slow_candidates(it["query"], 10)
            pool_hit.append(1.0 if it["relevant"] & {c["chunk_id"] for c, _ in cands}
                            else 0.0)

        lat.sort()
        out["presets"][name] = {
            "hit@1": round(float(np.mean(per_q)), 4),
            "hit@5": round(float(np.mean(pooled)), 4),
            "ndcg@5": round(float(np.mean([
                ndcg([c["chunk_id"] for c in ret.retrieve(it["query"], k=5, profile="auto")],
                     it["relevant"]) for it in items
            ])), 4),
            "p50": round(lat[len(lat) // 2], 1),
            "_per_q": per_q,
            "pool_recall@10": round(float(np.mean(pool_hit)), 4),
            "ceiling_gap": round(float(np.mean(np.array(pool_hit) - np.array(first))), 4),
        }
    return out


def main():
    rc.ensure_utf8_stdout()
    ap = argparse.ArgumentParser()
    ap.add_argument("--domains", nargs="*", default=list(DOMAINS))
    ap.add_argument("--presets", default="base,multiquery,mq_rrf,fusion0.7,fast_rrf")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    presets = [x.strip() for x in a.presets.split(",") if x.strip()]
    for p in presets:
        if p not in rc.PRESETS:
            raise SystemExit(f"未知预设 {p}; 可选 {list(rc.PRESETS)}")

    report = {}
    for dom in a.domains:
        res = run_domain(dom, presets)
        if not res:
            print(f"{dom}: 跳过")
            continue
        report[dom] = res
        print(f"\n== {dom}（n={res['n']}）")
        base = res["presets"]["base"]
        print(f"  {'预设':12s} {'hit@1':>7s} {'Δ':>7s} {'hit@5':>7s} "
              f"{'pool_rec@10':>12s} {'可救回':>8s} {'p50ms':>8s}   95%CI")
        for name in presets:
            r = res["presets"][name]
            delta = r["hit@1"] - base["hit@1"]
            ci = ""
            if name != "base":
                bs = rc._paired_bootstrap(base["_per_q"], r["_per_q"], rounds=2000)
                ci = f"[{bs['ci95'][0]:+.3f},{bs['ci95'][1]:+.3f}] P≤0={bs['p_not_better']:.3f}"
            print(f"  {name:12s} {r['hit@1']:7.3f} {delta:+7.3f} {r['hit@5']:7.3f} "
                  f"{r['pool_recall@10']:12.3f} {r['ceiling_gap']:8.3f} {r['p50']:8.1f}   {ci}")

    if a.out:
        for d in report.values():
            for p in d["presets"].values():
                p.pop("_per_q", None)
        Path(a.out).write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"\n已写入 {a.out}")


if __name__ == "__main__":
    main()
