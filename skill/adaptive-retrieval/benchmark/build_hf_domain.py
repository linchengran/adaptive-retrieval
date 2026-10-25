# -*- coding: utf-8 -*-
"""M7 阈值校准：从 HF 数据集构建第 3/4 语料域（与 M5 build_dureader 同方法学）。

设计（三个域一致，保证可与 wx / DuReader 数字对比）：
- 抽样 N 条查询（seed=42 可复现），真实标注；
- 语料 = 每条查询的正例文档 + 3 条 BM25 难负例（从候选池选，天然干扰项），按文档去重；
- 每篇文档按 512 字符固定窗口切块（overlap 64），chunk_id = f"{docid}::{i}"；
- 标签：relevant_chunks = 正例文档覆盖的 chunk_id；
- 输出 benchmark 通用格式（chunks.pkl + index.npy），run_benchmark / convert_to_skill_index 直接可用。

域定义：
- cmrc2018   中文维基阅读理解（hfl/cmrc2018），边界探针：段落均值 ~505 字符
- t2retrieval 百度搜索真实查询（C-MTEB/T2Retrieval + mteb qrels），带轻量 HTML 清洗
- cmedqa     医疗问答检索（C-MTEB/CmedqaRetrieval + -qrels），极短段落均值 ~101 字符

用法: python benchmark/build_hf_domain.py --domain cmrc2018 [--n 200]
输出: experiments/m7_<domain>/{index/, testset.json, build_meta.json}
"""
import argparse
import json
import pickle
import random
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from benchmark.corpus import Chunk
from benchmark.retrievers import BGE_SMALL

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _paths import REPO_ROOT  # noqa: E402
ROOT = REPO_ROOT
CHUNK_SIZE = 512
NEG_PER_QUERY = 3
POOL_SIZE = 15000  # BM25 难负例候选池大小
NEARDUP_PREFIX = 60  # 前 60 字符相同的段落视为同一 passage 的截断/微调变体


def chunk_text(text: str, size: int = CHUNK_SIZE, overlap: int = 64) -> list:
    if len(text) <= size:
        return [text]
    out = []
    start = 0
    while start < len(text):
        out.append(text[start:start + size])
        start += size - overlap
    return out


def clean_t2(text: str) -> str:
    """T2 语料的轻量清洗：换行标记还原、去其余 HTML 标签（62% 段落带标记）"""
    text = re.sub(r"<br\s*/?>", "\n", text)
    return re.sub(r"<[^>]+>", "", text)


# ---------- 各域加载：返回 (queries, positives, texts) ----------
# queries: list[str]；positives: list[list[docid]] 与 queries 对齐；texts: dict docid->text

def load_cmrc2018(n):
    from datasets import load_dataset
    ds = load_dataset("hfl/cmrc2018", split="train")
    # 按语段去重：多个问题共享同一段落，只取每段一个问题
    first_q = {}
    for row in ds:
        first_q.setdefault(row["context"], row["question"])
    items = list(first_q.items())  # (context, question)
    print(f"cmrc2018 唯一语段: {len(items)}")
    sample = random.sample(items, n)
    texts, queries, positives = {}, [], []
    for i, (ctx, q) in enumerate(sample):
        docid = f"wiki_{i:05d}"
        texts[docid] = ctx
        queries.append(q)
        positives.append([docid])
    # 难负例池 = 全部唯一语段
    pool_texts = {f"wiki_{j:05d}": ctx for j, (ctx, _) in enumerate(items)}
    return queries, positives, texts, pool_texts


def load_t2retrieval(n):
    from datasets import load_dataset
    corpus = load_dataset("C-MTEB/T2Retrieval", split="corpus")
    queries_ds = load_dataset("C-MTEB/T2Retrieval", split="queries")
    qrels = load_dataset("mteb/T2Retrieval", "default", split="dev")
    qtext = {r["id"]: r["text"] for r in queries_ds}
    pos_by_q = {}
    for r in qrels:
        if r["score"] > 0:
            pos_by_q.setdefault(r["query-id"], []).append(r["corpus-id"])
    qids = [q for q in pos_by_q if q in qtext]
    print(f"t2retrieval 有标注查询: {len(qids)}")
    sample = random.sample(qids, n)
    corpus_text = {r["id"]: clean_t2(r["text"]) for r in corpus}
    queries, positives, texts = [], [], {}
    for qid in sample:
        pids = [p for p in pos_by_q[qid] if p in corpus_text]
        if not pids:
            continue
        queries.append(qtext[qid])
        positives.append(pids)
    for pids in positives:
        for pid in pids:
            texts.setdefault(pid, corpus_text[pid])
    # 难负例池 = 随机语料 + 全部正例
    all_ids = list(corpus_text)
    pool_ids = set(random.sample(all_ids, min(POOL_SIZE, len(all_ids)))) | set(texts)
    pool_texts = {i: corpus_text[i] for i in pool_ids if corpus_text[i].strip()}
    return queries, positives, texts, pool_texts


def load_cmedqa(n):
    from datasets import load_dataset
    corpus = load_dataset("C-MTEB/CmedqaRetrieval", split="corpus")
    queries_ds = load_dataset("C-MTEB/CmedqaRetrieval", split="queries")
    qrels = load_dataset("C-MTEB/CmedqaRetrieval-qrels", "default", split="dev")
    qtext = {r["id"]: r["text"] for r in queries_ds}
    pos_by_q = {}
    for r in qrels:
        if r["score"] > 0:
            pos_by_q.setdefault(r["qid"], []).append(r["pid"])
    qids = [q for q in pos_by_q if q in qtext]
    print(f"cmedqa 有标注查询: {len(qids)}")
    sample = random.sample(qids, n)
    corpus_text = {r["id"]: r["text"] for r in corpus}
    queries, positives, texts = [], [], {}
    for qid in sample:
        pids = [p for p in pos_by_q[qid] if p in corpus_text]
        if not pids:
            continue
        queries.append(qtext[qid])
        positives.append(pids)
    for pids in positives:
        for pid in pids:
            texts.setdefault(pid, corpus_text[pid])
    all_ids = list(corpus_text)
    pool_ids = set(random.sample(all_ids, min(POOL_SIZE, len(all_ids)))) | set(texts)
    pool_texts = {i: corpus_text[i] for i in pool_ids if corpus_text[i].strip()}
    return queries, positives, texts, pool_texts


LOADERS = {
    "cmrc2018": load_cmrc2018,
    "t2retrieval": load_t2retrieval,
    "cmedqa": load_cmedqa,
}


def dedup_positives(queries, positives, texts):
    """正例按前缀家族去重：每族保留最长一篇，族内其余文档的标签重映射到保留 docid。

    背景（M7 过程修复）：cmrc2018 语段存在大量截断变体（65% 文档在碰撞家族里），
    当正例与其变体同时入选时二者向量近乎重合，rank-1 在 FAISS 与 numpy 实现
    之间随机翻转，hit@1 不可复现。变体不是独立语段，去重而非保留。
    """
    fams = {}
    for ids in positives:
        for d in ids:
            fams.setdefault(texts[d][:NEARDUP_PREFIX], []).append(d)
    remap = {}
    kept = {}
    for p, ds in fams.items():
        best = max(ds, key=lambda d: len(texts[d]))
        kept[best] = texts[best]
        for d in ds:
            if d != best:
                remap[d] = best
    if remap:
        print(f"正例近重复去重: 合并 {len(remap)} 篇变体 -> 保留 {len(kept)} 篇")
    positives2 = []
    for ids in positives:
        merged = []
        for d in ids:
            nd = remap.get(d, d)
            if nd not in merged:
                merged.append(nd)
        positives2.append(merged)
    return queries, positives2, kept


def bm25_hard_negatives(queries, positives, pool_texts, texts):
    """每个查询从候选池 BM25 取 top 负例（排除正例与前缀近重复），分低时回退随机"""
    import jieba
    from rank_bm25 import BM25Okapi
    jieba.setLogLevel(60)

    pool_ids = list(pool_texts)
    tok = [jieba.lcut(pool_texts[i].lower()) for i in pool_ids]
    bm25 = BM25Okapi(tok)
    print(f"BM25 负例池: {len(pool_ids)} 篇")

    pos_set = [set(p) for p in positives]
    # 已入选文档（正例 + 已选负例）的前缀集合：变体不再重复入选
    used_prefixes = {t[:NEARDUP_PREFIX] for t in texts.values()}
    negatives = []
    for qi, q in enumerate(queries):
        scores = bm25.get_scores(jieba.lcut(q.lower()))
        order = [pool_ids[i] for i in np.argsort(scores)[::-1]]
        neg = []
        for d in order:
            if d in pos_set[qi]:
                continue
            pre = pool_texts[d][:NEARDUP_PREFIX]
            if pre in used_prefixes:
                continue
            neg.append(d)
            used_prefixes.add(pre)
            if len(neg) >= NEG_PER_QUERY:
                break
        if len(neg) < NEG_PER_QUERY:  # 兜底：随机补（同样排除近重复）
            rest = []
            for d in pool_ids:
                if d in pos_set[qi] or d in neg:
                    continue
                if pool_texts[d][:NEARDUP_PREFIX] in used_prefixes:
                    continue
                rest.append(d)
            for d in random.sample(rest, min(NEG_PER_QUERY - len(neg), len(rest))):
                neg.append(d)
                used_prefixes.add(pool_texts[d][:NEARDUP_PREFIX])
        negatives.append(neg)
    return negatives


import numpy as np  # noqa: E402  (bm25_hard_negatives 依赖，放模块级供其使用)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--domain", required=True, choices=list(LOADERS))
    parser.add_argument("--n", type=int, default=200)
    args = parser.parse_args()

    random.seed(42)
    queries, positives, texts, pool_texts = LOADERS[args.domain](args.n)
    queries, positives, texts = dedup_positives(queries, positives, texts)
    negatives = bm25_hard_negatives(queries, positives, pool_texts, texts)

    # 语料组装：正例 + 难负例，去重
    for neg in negatives:
        for d in neg:
            if d not in texts:
                texts[d] = pool_texts[d]
    print(f"语料文档: {len(texts)} 篇")

    # 切块
    chunks = []
    docid_to_chunkids = {}
    for docid, text in texts.items():
        ids = []
        for i, c in enumerate(chunk_text(text)):
            cid = f"{docid}::{i}"
            ids.append(cid)
            chunks.append(Chunk(text=c, metadata={"source": docid, "chunk_index": i, "chunk_id": cid}))
        docid_to_chunkids[docid] = ids

    testset = []
    for qi, (query, pos_docids) in enumerate(zip(queries, positives)):
        rel = [cid for d in pos_docids for cid in docid_to_chunkids.get(d, [])]
        testset.append({
            "question": query,
            "relevant_chunks": rel,
            "relevant_sources": pos_docids,
            "origin": args.domain,
        })

    # 编码 + FAISS 索引
    import faiss
    from sentence_transformers import SentenceTransformer
    import torch

    model = SentenceTransformer(BGE_SMALL, device="cuda" if torch.cuda.is_available() else "cpu")
    print(f"编码 {len(chunks)} 个 chunk ...")
    vecs = model.encode([c.text for c in chunks], normalize_embeddings=True, show_progress_bar=False)
    index = faiss.IndexFlatIP(vecs.shape[1])
    index.add(np.ascontiguousarray(np.asarray(vecs, dtype="float32")))

    out_dir = ROOT / "experiments" / f"m7_{args.domain}"
    idx_dir = out_dir / "index"
    idx_dir.mkdir(parents=True, exist_ok=True)
    arr = np.array(faiss.serialize_index(index), dtype=np.uint8)
    np.save(idx_dir / "index.tmp.npy", arr)
    (idx_dir / "index.tmp.npy").replace(idx_dir / "index.npy")
    with (idx_dir / "chunks.tmp.pkl").open("wb") as f:
        pickle.dump(chunks, f)
    (idx_dir / "chunks.tmp.pkl").replace(idx_dir / "chunks.pkl")
    with (out_dir / "testset.json").open("w", encoding="utf-8") as f:
        json.dump(testset, f, ensure_ascii=False, indent=1)

    lens = [len(c.text) for c in chunks]
    meta = {
        "source": {
            "cmrc2018": "hfl/cmrc2018 train",
            "t2retrieval": "C-MTEB/T2Retrieval + mteb/T2Retrieval qrels",
            "cmedqa": "C-MTEB/CmedqaRetrieval + CmedqaRetrieval-qrels",
        }[args.domain],
        "n_queries": len(testset),
        "n_docs": len(texts),
        "n_chunks": len(chunks),
        "neg_per_query": NEG_PER_QUERY,
        "neg_source": f"BM25 top-3 from pool {len(pool_texts)} (seed 42, 前缀{NEARDUP_PREFIX}字符近重复排除)",
        "seed": 42,
        "chunk_size": CHUNK_SIZE,
        "chunk_overlap": 64,
        "avg_chunk_chars": round(sum(lens) / len(lens), 1),
        "cleaning": "t2retrieval: <br>→\\n + strip HTML tags" if args.domain == "t2retrieval" else "无",
    }
    with (out_dir / "build_meta.json").open("w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print(json.dumps(meta, ensure_ascii=False))


if __name__ == "__main__":
    main()
