# -*- coding: utf-8 -*-
"""路由器选型脚本：在 train 集上比较候选路由信号，保存选定配置。

候选信号：
1. margin  dense 余弦置信间隔（top1-top2 / top1-top5）——已否决：正确与
   错误 top1 的间隔分布几乎重合，无区分度；
2. agreement  dense 与 BM25 top1 一致性——一致组 dense 正确率显著高于先验，
   采用。快路径 = dense top5 免 rerank；慢路径 = dense 召回 + rerank。

全部比较只在 train 集上完成，测试集（testset_v2）不参与任何选型。

用法: python benchmark/tune_router.py
输出: experiments/m2_router/router.json
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from benchmark.corpus import MAIN_PROJECT_DIR
from benchmark.retrievers import BM25Retriever, DenseRetriever

TRAIN_PATH = MAIN_PROJECT_DIR / "experiments" / "splits" / "train.json"
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _paths import REPO_ROOT  # noqa: E402
OUT_DIR = REPO_ROOT / "experiments" / "m2_router"


def main():
    dense = DenseRetriever()
    bm25 = BM25Retriever()
    train = json.load(TRAIN_PATH.open(encoding="utf-8"))
    print(f"调参集: train {len(train)} 条")

    rows = []
    for item in train:
        rel = set(item.get("relevant_chunks", []))
        d_hits = dense.retrieve_with_scores(item["question"], top_k=5)
        b_hits = bm25.retrieve(item["question"], top_k=5)
        d_ids = [c.chunk_id for c, _ in d_hits]
        b_ids = [c.chunk_id for c in b_hits]
        rows.append({
            "dense_top1_ok": bool(d_ids) and d_ids[0] in rel,
            "m12": d_hits[0][1] - d_hits[1][1] if len(d_hits) > 1 else 1.0,
            "m15": d_hits[0][1] - d_hits[-1][1] if d_hits else 0.0,
            "agree": bool(d_ids and b_ids and d_ids[0] == b_ids[0]),
        })

    n = len(rows)
    import statistics as st
    margin_report = {
        "m12_median_correct": round(st.median([r["m12"] for r in rows if r["dense_top1_ok"]]), 4),
        "m12_median_wrong": round(st.median([r["m12"] for r in rows if not r["dense_top1_ok"]]), 4),
        "m15_median_correct": round(st.median([r["m15"] for r in rows if r["dense_top1_ok"]]), 4),
        "m15_median_wrong": round(st.median([r["m15"] for r in rows if not r["dense_top1_ok"]]), 4),
        "verdict": "无区分度，否决",
    }
    print("margin 信号:", margin_report)

    agree_group = [r for r in rows if r["agree"]]
    agree_ok = sum(1 for r in agree_group if r["dense_top1_ok"])
    agreement_report = {
        "fast_path_rate": round(len(agree_group) / n, 4),
        "fast_path_correct_rate": round(agree_ok / len(agree_group), 4) if agree_group else 0.0,
        "dense_prior_correct_rate": round(sum(r["dense_top1_ok"] for r in rows) / n, 4),
        "verdict": "采用",
    }
    print("agreement 信号:", agreement_report)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    config = {
        "strategy": "agreement",
        "rule": "dense 与 BM25 top1 一致 -> dense top5 直返（免 rerank）；否则 dense 召回 + rerank",
        "tuned_on": str(TRAIN_PATH),
        "tolerance_vs_full_rerank": 0.02,
        "signal_reports": {"margin": margin_report, "agreement": agreement_report},
    }
    with (OUT_DIR / "router.json").open("w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)
    print(f"已保存: {OUT_DIR / 'router.json'}")


if __name__ == "__main__":
    main()
