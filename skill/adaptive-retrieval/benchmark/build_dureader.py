# -*- coding: utf-8 -*-
"""从 DuReader-retrieval ranking dev 集构建跨域验证语料与测试集。

设计：
- 抽样 N 条查询（固定种子可复现）；
- 语料 = 每条查询的 1 条正例段落 + 3 条难负例（天然干扰项），按 docid 去重；
- 每个段落按 512 字符切粗块，chunk_id = f"{docid}::{i}"；
- 标签：relevant_chunks = 正例段落覆盖的 chunk_id；
- 输出与 wx server 索引同格式（chunks.pkl + index.npy），benchmark 框架直接可用。

用法: python benchmark/build_dureader.py [--n 200]
输出: experiments/m5_dureader/{index/, testset.json, build_meta.json}
"""
import argparse
import gzip
import json
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from benchmark.corpus import Chunk
from benchmark.retrievers import BGE_SMALL

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _paths import REPO_ROOT  # noqa: E402
OUT_DIR = REPO_ROOT / "experiments" / "m5_dureader"
RAW_PATH = REPO_ROOT / "data_raw" / "dureader" / "dev.jsonl.gz"
CHUNK_SIZE = 512
NEG_PER_QUERY = 3


def chunk_text(text: str, size: int = CHUNK_SIZE, overlap: int = 64) -> list:
    """固定窗口切块（带 overlap），与既有 chunk_id 语义一致"""
    if len(text) <= size:
        return [text]
    out = []
    start = 0
    while start < len(text):
        out.append(text[start:start + size])
        start += size - overlap
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=200)
    args = parser.parse_args()

    import random
    random.seed(42)

    with gzip.open(RAW_PATH, "rt", encoding="utf-8") as f:
        rows = [json.loads(line) for line in f]
    rows = [r for r in rows if r.get("positive_passages")]
    print(f"dev 集共 {len(rows)} 条查询（含正例段落）")
    sample = random.sample(rows, args.n)

    # 语料：正例 + 难负例，去重
    passages = {}  # docid -> text
    labels = []    # (query, [positive chunk_ids])
    for row in sample:
        pos = row["positive_passages"][0]
        pos_docids = [pos["docid"]]
        for p in row.get("negative_passages", [])[:NEG_PER_QUERY]:
            passages.setdefault(p["docid"], p["text"])
        if pos["docid"] not in passages:
            passages[pos["docid"]] = pos["text"]
        # 标签在切块后填
        labels.append((row["query"], pos_docids))

    print(f"语料段落: {len(passages)} 篇（含难负例干扰）")

    # 切块
    chunks = []
    docid_to_chunkids = {}
    for docid, text in passages.items():
        ids = []
        for i, c in enumerate(chunk_text(text)):
            cid = f"{docid}::{i}"
            ids.append(cid)
            chunks.append(Chunk(text=c, metadata={"source": docid, "chunk_index": i, "chunk_id": cid}))
        docid_to_chunkids[docid] = ids

    testset = []
    for query, pos_docids in labels:
        rel = [cid for d in pos_docids for cid in docid_to_chunkids[d]]
        testset.append({
            "question": query,
            "relevant_chunks": rel,
            "relevant_sources": pos_docids,
            "origin": "dureader_dev",
        })

    # 编码 + FAISS 索引
    import numpy as np
    import faiss
    from sentence_transformers import SentenceTransformer
    import torch

    model = SentenceTransformer(BGE_SMALL, device="cuda" if torch.cuda.is_available() else "cpu")
    print(f"编码 {len(chunks)} 个 chunk ...")
    vecs = model.encode([c.text for c in chunks], normalize_embeddings=True, show_progress_bar=False)
    index = faiss.IndexFlatIP(vecs.shape[1])
    index.add(np.ascontiguousarray(np.asarray(vecs, dtype="float32")))

    idx_dir = OUT_DIR / "index"
    idx_dir.mkdir(parents=True, exist_ok=True)
    # 与 wx server 相同的落盘格式（绕开 faiss 中文路径问题的 npy 方案）
    arr = np.array(faiss.serialize_index(index), dtype=np.uint8)
    np.save(idx_dir / "index.tmp.npy", arr)
    (idx_dir / "index.tmp.npy").replace(idx_dir / "index.npy")
    with (idx_dir / "chunks.tmp.pkl").open("wb") as f:
        pickle.dump(chunks, f)
    (idx_dir / "chunks.tmp.pkl").replace(idx_dir / "chunks.pkl")

    with (OUT_DIR / "testset.json").open("w", encoding="utf-8") as f:
        json.dump(testset, f, ensure_ascii=False, indent=1)

    meta = {
        "source": "zyznull/dureader-retrieval-ranking dev",
        "n_queries": len(testset),
        "n_passages": len(passages),
        "n_chunks": len(chunks),
        "neg_per_query": NEG_PER_QUERY,
        "seed": 42,
        "chunk_size": CHUNK_SIZE,
    }
    with (OUT_DIR / "build_meta.json").open("w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print(json.dumps(meta, ensure_ascii=False))


if __name__ == "__main__":
    main()
