# -*- coding: utf-8 -*-
"""chunk 级检索指标：Hit@k / Recall@k / MRR@k / nDCG@k。

与 wx 程序 server/src/eval_retrieval.py 的口径保持一致，
保证与小程序侧已有实验数字可比。
"""
import math
from typing import Dict, List, Sequence


def chunk_metrics(retrieved_ids: List[str], relevant_ids: List[str],
                  k_list: Sequence[int] = (1, 3, 5)) -> Dict[str, float]:
    relevant = list(dict.fromkeys(relevant_ids))
    relevant_set = set(relevant)
    results: Dict[str, float] = {}
    if not relevant:
        return {f"{m}@{k}": 0.0 for k in k_list for m in ("hit", "recall", "mrr", "ndcg")}

    for k in k_list:
        topk = retrieved_ids[:k]
        hits_in_topk = sum(1 for doc_id in topk if doc_id in relevant_set)
        first_rank = next((r for r, d in enumerate(topk, 1) if d in relevant_set), None)
        dcg = sum(1.0 / math.log2(r + 1) for r, d in enumerate(topk, 1) if d in relevant_set)
        ideal_hits = min(len(relevant), k)
        idcg = sum(1.0 / math.log2(r + 1) for r in range(1, ideal_hits + 1))

        results[f"hit@{k}"] = 1.0 if first_rank else 0.0
        results[f"recall@{k}"] = hits_in_topk / len(relevant)
        results[f"mrr@{k}"] = 1.0 / first_rank if first_rank else 0.0
        results[f"ndcg@{k}"] = dcg / idcg if idcg else 0.0
    return results


def average(metrics_list: List[Dict[str, float]]) -> Dict[str, float]:
    if not metrics_list:
        return {}
    keys = metrics_list[0].keys()
    return {key: round(sum(m[key] for m in metrics_list) / len(metrics_list), 4)
            for key in keys}
