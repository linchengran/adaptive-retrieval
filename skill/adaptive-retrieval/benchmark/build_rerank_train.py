# -*- coding: utf-8 -*-
"""构造重排器微调用的训练集（严格测试集隔离）。

数据源：data_raw/dureader/dev.jsonl.gz（2000 条，含正例与难负例）。
测试集用的是 seed=42 抽的 200 条（见 build_dureader.py），所以这里：
  1. 先复现 build_dureader 的随机抽样，得到**测试集 query_id 集合**；
  2. 从剩余 query 里抽训练 query，保证与测试集 query_id 零交集；
  3. 每条训练 query 配 1 个正例 + N 个难负例 → cross-encoder 的 (query, passage, label) 三元组；
  4. 语料去重，并排除测试集语料里出现过的 docid（防止答案泄漏）。

输出：experiments/_rerank_train/{train.jsonl, meta.json}
用法：python benchmark/build_rerank_train.py --n-queries 800 --neg 4
"""
from __future__ import annotations

import argparse
import gzip
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _paths import REPO_ROOT as ROOT, SKILL_DIR  # noqa: E402
RAW = ROOT / "data_raw" / "dureader" / "dev.jsonl.gz"
OUT_DIR = ROOT / "experiments" / "_rerank_train"
TEST_META = ROOT / "experiments" / "m5_dureader" / "build_meta.json"
TEST_TS = ROOT / "experiments" / "m5_dureader" / "testset.json"


def load_rows() -> list:
    with gzip.open(RAW, "rt", encoding="utf-8") as f:
        rows = [json.loads(line) for line in f]
    return [r for r in rows if r.get("positive_passages")]


def test_query_ids(rows: list) -> set:
    """复现 build_dureader 的抽样，回推出测试集那条 200 条用的是哪些 query。"""
    random.seed(42)
    sampled = random.sample(rows, 200)
    return {r.get("query_id") for r in sampled}


def main() -> None:
    for _s in (sys.stdout, sys.stderr):
        try:
            if hasattr(_s, "reconfigure"):
                _s.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-queries", type=int, default=800)
    ap.add_argument("--neg", type=int, default=4, help="每条 query 的难负例数")
    ap.add_argument("--max-text", type=int, default=512,
                    help="passage 截断长度（cross-encoder max_length=512 token，中文约 1 字=1 token）")
    a = ap.parse_args()

    rows = load_rows()
    held_out = test_query_ids(rows)
    print(f"原始 {len(rows)} 条；测试集占用 {len(held_out)} 条 query_id（seed=42）")

    candidates = [r for r in rows if r.get("query_id") not in held_out]
    print(f"隔离后可用训练候选 {len(candidates)} 条")

    # 测试集语料里的 docid：训练时排除，避免答案泄漏
    test_docids = set()
    if TEST_TS.exists():
        for it in json.loads(TEST_TS.read_text(encoding="utf-8")):
            test_docids.update(it.get("relevant_sources") or [])
            for cid in it.get("relevant_chunks") or []:
                test_docids.add(cid.split("::")[0])
    print(f"测试集涉及 docid {len(test_docids)} 个（训练语料中排除）")

    random.seed(7)
    picked = random.sample(candidates, min(a.n_queries, len(candidates)))
    # 训练集内部再切一小块做 dev（用来看微调有没有过拟合），同样不与测试集混
    n_dev = max(50, len(picked) // 8)
    dev_queries, train_queries = picked[:n_dev], picked[n_dev:]

    pairs = {"train": [], "dev": []}
    seen_docids = set()
    for split, qs in (("dev", dev_queries), ("train", train_queries)):
        for r in qs:
            pos = r["positive_passages"][0]
            if pos["docid"] in test_docids:
                continue
            pairs[split].append({"query": r["query"], "passage": pos["text"][:a.max_text],
                                 "label": 1, "docid": pos["docid"], "qid": r.get("query_id")})
            seen_docids.add(pos["docid"])
            negs = [p for p in r.get("negative_passages", []) if p["docid"] not in test_docids]
            for p in negs[:a.neg]:
                pairs[split].append({"query": r["query"], "passage": p["text"][:a.max_text],
                                     "label": 0, "docid": p["docid"], "qid": r.get("query_id")})
                seen_docids.add(p["docid"])

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for split, rows_out in pairs.items():
        path = OUT_DIR / f"{split}.jsonl"
        with path.open("w", encoding="utf-8") as f:
            for item in rows_out:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")

    meta = {
        "source": str(RAW),
        "isolation": {
            "test_query_ids_held_out": len(held_out),
            "test_docids_excluded": len(test_docids),
            "overlap_check": "训练集 query_id 与测试集 query_id 交集应为 0",
        },
        "neg_per_query": a.neg,
        "max_text": a.max_text,
        "train_pairs": len(pairs["train"]),
        "dev_pairs": len(pairs["dev"]),
        "train_queries": len({p["qid"] for p in pairs["train"]}),
        "dev_queries": len({p["qid"] for p in pairs["dev"]}),
        "unique_docids": len(seen_docids),
    }
    (OUT_DIR / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1),
                                       encoding="utf-8")

    # 硬校验：训练/测试 query 零交集
    overlap = {p["qid"] for p in pairs["train"] + pairs["dev"]} & held_out
    assert not overlap, f"训练集与测试集 query 重叠 {len(overlap)} 条，必须修数据构造"
    print(json.dumps(meta, ensure_ascii=False, indent=1))
    print(f"\n✓ 隔离校验通过：训练/测试 query 交集 = 0")
    print(f"输出: {OUT_DIR}")


if __name__ == "__main__":
    main()
