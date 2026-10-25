# -*- coding: utf-8 -*-
"""M1 基线跑分入口。

用法:
    python benchmark/run_benchmark.py                 # 全部三路
    python benchmark/run_benchmark.py --only bm25     # 只跑某一路 (dense/bm25/dense+rerank)

输出:
    experiments/m1_baseline/results.json  每路指标 + 每题明细
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from benchmark.corpus import TESTSET_PATH, load_testset
from benchmark.metrics import average, chunk_metrics
from benchmark.retrievers import (
    BM25Retriever,
    DensePlusRerank,
    DenseRetriever,
    HierarchicalPlusRerank,
    HierarchicalRetriever,
    HybridPlusRerank,
    HybridRRFRetriever,
    LearnedRouterRetriever,
    RoutedHierRetriever,
    RoutedRetriever,
    measure_latency,
)

K_LIST = (1, 3, 5)
TOP_K = 5
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _paths import REPO_ROOT  # noqa: E402

OUT_DIR = REPO_ROOT / "experiments"
TESTSET_V2 = Path(__file__).resolve().parent / "data" / "testset_v2.json"


def build_retrievers(only: str, index_dir=None, skip_learned=False):
    """按需构建检索臂；放在函数内以便 --only 跳过昂贵的模型加载。
    index_dir 非 None 时覆盖默认索引（跨域语料验证用）。
    only 支持逗号分隔多臂（如 "dense,hier"）或 "all"。"""
    names = [n.strip() for n in only.split(",")] if only != "all" else None
    want = lambda n: names is None or n in names
    kw = {"index_dir": index_dir} if index_dir else {}
    arms = []
    if want("dense"):
        arms.append(DenseRetriever(**kw))
    if want("bm25"):
        arms.append(BM25Retriever(**kw))
    if want("dense+rerank"):
        try:
            arms.append(DensePlusRerank(**kw))
        except Exception as exc:
            print(f"[跳过] dense+rerank 加载失败: {exc}")
    if want("hybrid"):
        arms.append(HybridRRFRetriever(**kw))
    if want("hybrid+rerank"):
        try:
            arms.append(HybridPlusRerank(**kw))
        except Exception as exc:
            print(f"[跳过] hybrid+rerank 加载失败: {exc}")
    if want("routed"):
        try:
            arms.append(RoutedRetriever(**kw))
        except Exception as exc:
            print(f"[跳过] routed 加载失败: {exc}")
    if want("routed-learned") and not skip_learned:
        try:
            arms.append(LearnedRouterRetriever(**kw))
        except Exception as exc:
            print(f"[跳过] routed-learned 加载失败: {exc}")
    if want("hier"):
        try:
            arms.append(HierarchicalRetriever(**kw))
        except Exception as exc:
            print(f"[跳过] hierarchical 加载失败: {exc}")
    if want("hier+rerank"):
        try:
            arms.append(HierarchicalPlusRerank(**kw))
        except Exception as exc:
            print(f"[跳过] hier+rerank 加载失败: {exc}")
    if want("routed-hier"):
        try:
            arms.append(RoutedHierRetriever(**kw))
        except Exception as exc:
            print(f"[跳过] routed-hier 加载失败: {exc}")
    return arms


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", default="all",
                        help="单臂名、逗号分隔多臂（如 dense,hier）或 all")
    parser.add_argument("--testset", default=str(TESTSET_V2))
    parser.add_argument("--tag", default=None, help="输出子目录名，默认按测试集命名")
    parser.add_argument("--index-dir", default=None, help="覆盖默认索引目录（跨域语料）")
    parser.add_argument("--skip-learned", action="store_true", help="跳过学习型路由臂")
    args = parser.parse_args()

    testset = load_testset(Path(args.testset))
    queries = [item["question"] for item in testset]

    tag = args.tag or ("m1_baseline" if "splits" in str(args.testset) else "m2_router")
    out_dir = OUT_DIR / tag
    results = {
        "meta": {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "testset": str(args.testset),
            "n_queries": len(testset),
            "top_k": TOP_K,
            "note": "语料与索引复用 wx 程序 server",
        },
        "arms": {},
    }

    for retriever in build_retrievers(args.only, index_dir=args.index_dir, skip_learned=args.skip_learned):
        print(f"\n===== 检索臂: {retriever.name} =====")
        per_query = []
        for item in testset:
            retrieved = retriever.retrieve(item["question"], top_k=TOP_K)
            ids = [c.chunk_id for c in retrieved]
            metrics = chunk_metrics(ids, item.get("relevant_chunks", []), K_LIST)
            per_query.append({
                "question": item["question"],
                "retrieved_ids": ids,
                "relevant_chunks": item.get("relevant_chunks", []),
                "metrics": metrics,
            })

        avg = average([q["metrics"] for q in per_query])
        route_stats = dict(retriever.stats) if hasattr(retriever, "stats") else None
        latency = measure_latency(retriever, queries, top_k=TOP_K)
        arm_result = {
            "avg_metrics": avg,
            "latency": latency,
            "per_query": per_query,
        }
        if route_stats is not None:
            arm_result["route_stats"] = route_stats
        results["arms"][retriever.name] = arm_result
        print(f"指标: {json.dumps(avg, ensure_ascii=False)}")
        print(f"延迟: {json.dumps(latency)}")
        if route_stats is not None:
            print(f"路由: {route_stats}")

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "results.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n结果已保存: {out_path}")


if __name__ == "__main__":
    main()
