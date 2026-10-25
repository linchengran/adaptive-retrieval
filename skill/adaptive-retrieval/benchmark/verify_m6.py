# -*- coding: utf-8 -*-
"""M6 端到端验证：Skill Retriever 在两个语料上的语料分类与三档精度。

两个索引均由 convert_to_skill_index.py 从 benchmark 语料转换而来，
chunk 与 benchmark 完全一致，结果可与 benchmark 各臂直接对比。预期：
- wx 语料（长文档）: fast=hier(≈0.49) auto=agreement路由(≈0.71) best=hier+rerank(≈0.71)
- DuReader（短段落）: fast=dense(≈0.49) auto=best=dense+rerank(≈0.62，修复路由 -0.045 回归)
延迟为混跑参考值（M4 教训：延迟结论以 benchmark 独占测量为准）。
"""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _paths import REPO_ROOT as ROOT, SKILL_DIR  # noqa: E402
sys.path.insert(0, str(SKILL_DIR / "scripts"))
from retrieval_core import Retriever  # noqa: E402


def load_items(path):
    ts = json.loads(Path(path).read_text(encoding="utf-8"))
    return ts["items"] if isinstance(ts, dict) else ts


def eval_index(index_dir, testset_path, tag):
    ret = Retriever(str(index_dir))
    items = load_items(testset_path)
    print(f"\n=== {tag} ===")
    print(f"语料分类: {ret.corpus_type}  stats={ret.stats}")

    out = {}
    for profile in ("fast", "auto", "best"):
        hits, t0 = 0, time.perf_counter()
        for it in items:
            res = ret.retrieve(it["question"], k=5, profile=profile)
            hits += any(r["chunk_id"] in set(it["relevant_chunks"]) for r in res[:1])
        dt = (time.perf_counter() - t0) / len(items) * 1000
        out[profile] = round(hits / len(items), 3)
        print(f"  {profile:5s} hit@1={hits / len(items):.3f}  ({dt:.0f} ms/query, 混跑参考)")
    return out


def main():
    base = ROOT
    r1 = eval_index(
        base / "experiments" / "m6_corpus_aware" / "wx_skill_index",
        Path(__file__).resolve().parent / "data" / "testset_v2.json",
        "wx 知识库（长文档语料，55 查询）")
    r2 = eval_index(
        base / "experiments" / "m5_dureader" / "skill_index",
        base / "experiments" / "m5_dureader" / "testset.json",
        "DuReader（短段落语料，200 查询）")

    (base / "experiments" / "m6_corpus_aware").mkdir(parents=True, exist_ok=True)
    (base / "experiments" / "m6_corpus_aware" / "verify_results.json").write_text(
        json.dumps({"wx": {"corpus": "long", **r1},
                    "dureader": {"corpus": "short", **r2}},
                   ensure_ascii=False, indent=1), encoding="utf-8")
    print("\n结果已写入 experiments/m6_corpus_aware/verify_results.json")


if __name__ == "__main__":
    main()
