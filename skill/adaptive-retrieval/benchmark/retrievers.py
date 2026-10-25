# -*- coding: utf-8 -*-
"""可插拔检索器：每一路检索实现统一的 retrieve(query, top_k) 接口。

M1 包含三路：
- DenseRetriever   纯向量（bge-small-zh + FAISS，当前小程序的生产路径去 rerank）
- BM25Retriever    纯关键词（jieba 分词 + BM25Okapi）
- DensePlusRerank  向量召回 + bge-reranker 重排（当前小程序的完整生产管线）
"""
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List

import numpy as np

from .corpus import Chunk, INDEX_DIR, load_chunks, load_faiss_index

# 干净的本地模型副本（HF 缓存被旧 LoRA 实验污染，见 experiments/m1_baseline 报告）
import sys as _sys  # noqa: E402
_sys.path.insert(0, str(Path(__file__).resolve().parent))
from _paths import REPO_ROOT  # noqa: E402

MODELS_DIR = REPO_ROOT / "models"
BGE_SMALL = str(MODELS_DIR / "bge-small-zh-v1.5")
BGE_RERANKER = str(MODELS_DIR / "bge-reranker-base")


class BaseRetriever(ABC):
    name: str = "base"

    @abstractmethod
    def retrieve(self, query: str, top_k: int = 5) -> List[Chunk]:
        ...


class DenseRetriever(BaseRetriever):
    """纯向量检索：加载现成 FAISS 索引，查询时编码 + 精确内积"""

    def __init__(self, model_name: str = BGE_SMALL,
                 index_dir: Path = INDEX_DIR):
        from sentence_transformers import SentenceTransformer
        import torch

        self.name = "dense"
        self.index = load_faiss_index(index_dir)
        self.chunks: List[Chunk] = load_chunks(index_dir)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.embedder = SentenceTransformer(model_name, device=device)
        self.query_instruction = "为这个句子生成表示以用于检索相关文章："

    def _embed_query(self, query: str) -> np.ndarray:
        vec = self.embedder.encode(
            [self.query_instruction + query],
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return np.ascontiguousarray(vec.astype("float32"))

    def retrieve(self, query: str, top_k: int = 5) -> List[Chunk]:
        k = min(top_k, self.index.ntotal)
        scores, ids = self.index.search(self._embed_query(query), k)
        return [self.chunks[i] for i in ids[0] if i != -1]

    def retrieve_with_scores(self, query: str, top_k: int = 5):
        """返回 (chunk, score) 列表，供路由器计算置信度间隔"""
        k = min(top_k, self.index.ntotal)
        scores, ids = self.index.search(self._embed_query(query), k)
        return [(self.chunks[i], float(s)) for i, s in zip(ids[0], scores[0]) if i != -1]


class HybridRRFRetriever(BaseRetriever):
    """混合检索：dense top-R + BM25 top-R，倒数排名融合（RRF, k=60）后取 top_k"""

    RRF_K = 60

    def __init__(self, recall_k: int = 10, **dense_kwargs):
        self.name = "hybrid_rrf"
        self.recall_k = recall_k
        self.dense = DenseRetriever(**dense_kwargs)
        self.bm25 = BM25Retriever(dense_kwargs.get("index_dir", INDEX_DIR))

    def retrieve(self, query: str, top_k: int = 5) -> List[Chunk]:
        dense_hits = self.dense.retrieve_with_scores(query, top_k=self.recall_k)
        bm25_hits = self.bm25.retrieve(query, top_k=self.recall_k)

        # 注意：dense 与 bm25 各自加载了 chunks 列表，同一 chunk 是不同对象，
        # 去重必须按 chunk_id 而不是对象 id
        rrf: Dict[str, float] = {}
        for rank, (chunk, _) in enumerate(dense_hits, 1):
            rrf[chunk.chunk_id] = rrf.get(chunk.chunk_id, 0.0) + 1.0 / (self.RRF_K + rank)
        for rank, chunk in enumerate(bm25_hits, 1):
            rrf[chunk.chunk_id] = rrf.get(chunk.chunk_id, 0.0) + 1.0 / (self.RRF_K + rank)

        chunk_by_id = {c.chunk_id: c for c, _ in dense_hits}
        for c in bm25_hits:
            chunk_by_id.setdefault(c.chunk_id, c)
        ranked = sorted(rrf.items(), key=lambda kv: kv[1], reverse=True)[:top_k]
        return [chunk_by_id[cid] for cid, _ in ranked]


class RoutedRetriever(BaseRetriever):
    """一致性路由：dense 与 BM25 的 top1 一致 -> 快路径（免 rerank）；
    不一致 -> 慢路径（dense 召回 + rerank）。

    选型依据（benchmark/tune_router.py，在 train 集上做出，未接触测试集）：
    - 余弦置信间隔（top1-top2 / top1-top5）对"rerank 是否有必要"无区分度
      （正确/错误 top1 的间隔中位数 0.010 vs 0.009），已否决；
    - top1 一致信号：一致组 dense 正确率 0.90（20/72），显著高于先验 0.57。
    """

    def __init__(self, recall_k: int = 10, **dense_kwargs):
        self.name = "routed(agreement)"
        self.recall_k = recall_k
        self.dense = DenseRetriever(**dense_kwargs)
        self.bm25 = BM25Retriever(dense_kwargs.get("index_dir", INDEX_DIR))
        from sentence_transformers import CrossEncoder
        import torch
        self.reranker = CrossEncoder(
            BGE_RERANKER, device="cuda" if torch.cuda.is_available() else "cpu",
            max_length=512,
        )
        self.stats = {"fast_path": 0, "slow_path": 0}

    def retrieve(self, query: str, top_k: int = 5) -> List[Chunk]:
        dense_hits = self.dense.retrieve_with_scores(query, top_k=top_k)
        bm25_hits = self.bm25.retrieve(query, top_k=top_k)

        agree = (
            dense_hits and bm25_hits
            and dense_hits[0][0].chunk_id == bm25_hits[0].chunk_id
        )
        if agree:
            self.stats["fast_path"] += 1
            return [c for c, _ in dense_hits[:top_k]]

        self.stats["slow_path"] += 1
        candidates = [c for c, _ in self.dense.retrieve_with_scores(query, top_k=self.recall_k)]
        if not candidates:
            return []
        pairs = [[query, c.text] for c in candidates]
        scores = self.reranker.predict(pairs, show_progress_bar=False)
        order = np.argsort(scores)[::-1][:top_k]
        return [candidates[i] for i in order]


class BM25Retriever(BaseRetriever):
    """纯关键词检索：jieba 分词 + BM25Okapi"""

    def __init__(self, index_dir: Path = INDEX_DIR):
        import jieba
        from rank_bm25 import BM25Okapi

        self.name = "bm25"
        jieba.setLogLevel(60)  # 关掉词典加载日志
        self.tokenize = jieba.lcut
        self.chunks: List[Chunk] = load_chunks(index_dir)
        corpus_tokens = [self.tokenize(c.text.lower()) for c in self.chunks]
        self.bm25 = BM25Okapi(corpus_tokens)

    def retrieve(self, query: str, top_k: int = 5) -> List[Chunk]:
        scores = self.bm25.get_scores(self.tokenize(query.lower()))
        order = np.argsort(scores)[::-1][:top_k]
        # BM25 分数可能全为 0（查询词都不在语料中），此时返回空以免假命中
        if len(order) and scores[order[0]] <= 0:
            return []
        return [self.chunks[i] for i in order]


class DensePlusRerank(BaseRetriever):
    """向量召回 top-10 + bge-reranker 重排取 top_k（当前生产管线）"""

    def __init__(self, model_name: str = BGE_RERANKER,
                 recall_k: int = 10, **dense_kwargs):
        from sentence_transformers import CrossEncoder
        import torch

        self.name = "dense+rerank"
        self.recall_k = recall_k
        self.dense = DenseRetriever(**dense_kwargs)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.reranker = CrossEncoder(model_name, device=device, max_length=512)

    def retrieve(self, query: str, top_k: int = 5) -> List[Chunk]:
        candidates = self.dense.retrieve(query, top_k=self.recall_k)
        if not candidates:
            return []
        pairs = [[query, c.text] for c in candidates]
        scores = self.reranker.predict(pairs, show_progress_bar=False)
        order = np.argsort(scores)[::-1][:top_k]
        return [candidates[i] for i in order]


# ---------- 延迟测量 ----------
def measure_latency(retriever: BaseRetriever, queries: List[str],
                    top_k: int = 5, warmup: int = 2) -> Dict[str, float]:
    """先预热再计时，返回 p50 / p95 / mean（毫秒）"""
    for q in queries[:warmup]:
        retriever.retrieve(q, top_k=top_k)
    times = []
    for q in queries:
        t0 = time.perf_counter()
        retriever.retrieve(q, top_k=top_k)
        times.append((time.perf_counter() - t0) * 1000)
    times = np.array(times)
    return {
        "p50_ms": round(float(np.percentile(times, 50)), 2),
        "p95_ms": round(float(np.percentile(times, 95)), 2),
        "mean_ms": round(float(times.mean()), 2),
    }


# ---------- M3：共用 rerank 辅助 ----------
def _load_cross_encoder(model_name: str = BGE_RERANKER):
    from sentence_transformers import CrossEncoder
    import torch
    return CrossEncoder(model_name,
                        device="cuda" if torch.cuda.is_available() else "cpu",
                        max_length=512)


def _rerank_candidates(query: str, candidates: List[Chunk], reranker, top_k: int) -> List[Chunk]:
    if not candidates:
        return []
    pairs = [[query, c.text] for c in candidates]
    scores = reranker.predict(pairs, show_progress_bar=False)
    order = np.argsort(scores)[::-1][:top_k]
    return [candidates[i] for i in order]


class HybridPlusRerank(BaseRetriever):
    """混合召回（RRF top-R）喂 rerank 取 top_k。

    M2 发现 hybrid 的 hit@5 比 dense 高 0.145，喂给 reranker 的候选质量更高。
    """

    def __init__(self, recall_k: int = 10, **dense_kwargs):
        self.name = "hybrid+rerank"
        self.recall_k = recall_k
        self.hybrid = HybridRRFRetriever(recall_k=recall_k, **dense_kwargs)
        self.reranker = _load_cross_encoder()

    def retrieve(self, query: str, top_k: int = 5) -> List[Chunk]:
        candidates = self.hybrid.retrieve(query, top_k=self.recall_k)
        return _rerank_candidates(query, candidates, self.reranker, top_k)


class HierarchicalRetriever(BaseRetriever):
    """分层分块：细块（句级，~120 字符）召回 -> 父粗块（512 字符）返回。

    细索引在内存中现场构建（29 个粗块 -> ~百级细块，编码一次的开销可接受）。
    动机：M1/M2 中"双双落空"的查询多为答案藏在粗块中段的细粒度事实，
    细块能提高命中的定位精度，粗块保证返回上下文完整。
    """

    FINE_SIZE = 120

    def __init__(self, index_dir: Path = INDEX_DIR, **dense_kwargs):
        import re
        import faiss
        from sentence_transformers import SentenceTransformer
        import torch

        self.name = "hierarchical"
        self.coarse_chunks: List[Chunk] = load_chunks(index_dir)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.embedder = SentenceTransformer(dense_kwargs.get("model_name", BGE_SMALL), device=device)
        self.query_instruction = "为这个句子生成表示以用于检索相关文章："

        # 粗块 -> 细块（按句子边界合并到 ~FINE_SIZE 字符，记录父 chunk_id）
        self.fine_chunks: List[Chunk] = []
        for parent in self.coarse_chunks:
            sentences = [s for s in re.split(r"(?<=[。！？!?])\s*", parent.text) if s.strip()]
            buf, buf_len = [], 0
            for sent in sentences:
                if buf and buf_len + len(sent) > self.FINE_SIZE:
                    self._add_fine(buf, parent)
                    buf, buf_len = [], 0
                buf.append(sent)
                buf_len += len(sent)
            if buf:
                self._add_fine(buf, parent)

        # 细块向量索引（内存）
        texts = [c.text for c in self.fine_chunks]
        vecs = self.embedder.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        self.fine_index = faiss.IndexFlatIP(vecs.shape[1])
        self.fine_index.add(np.ascontiguousarray(vecs.astype("float32")))
        print(f"分层索引: {len(self.coarse_chunks)} 粗块 -> {len(self.fine_chunks)} 细块")

    def _add_fine(self, parts: List[str], parent: Chunk):
        text = "".join(parts)
        self.fine_chunks.append(Chunk(
            text=text,
            metadata={**parent.metadata,
                      "parent_chunk_id": parent.chunk_id,
                      "fine_index": len(self.fine_chunks)},
        ))

    def _embed_query(self, query: str) -> np.ndarray:
        vec = self.embedder.encode(
            [self.query_instruction + query],
            normalize_embeddings=True, show_progress_bar=False,
        )
        return np.ascontiguousarray(vec.astype("float32"))

    def retrieve(self, query: str, top_k: int = 5) -> List[Chunk]:
        return self.retrieve_coarse(query, top_k=top_k, oversample=3)

    def retrieve_coarse(self, query: str, top_k: int = 5, oversample: int = 3) -> List[Chunk]:
        """细块检索并映射回父粗块；oversample 控制细块候选倍数"""
        k = min(top_k * oversample, self.fine_index.ntotal)
        _, ids = self.fine_index.search(self._embed_query(query), k)
        coarse_by_id = {c.chunk_id: c for c in self.coarse_chunks}
        seen, out = set(), []
        for i in ids[0]:
            if i == -1:
                continue
            pid = self.fine_chunks[i].metadata["parent_chunk_id"]
            if pid in seen:
                continue
            seen.add(pid)
            out.append(coarse_by_id[pid])
            if len(out) >= top_k:
                break
        return out


class LearnedRouterRetriever(BaseRetriever):
    """学习型路由：逻辑回归预测"该查询是否需要 rerank"。

    模型与特征由 benchmark/train_router_m3.py 在 train 集上训练（5 折 CV），
    序列化到 experiments/m3_learned_router/model.pkl。标签定义：
    y=1 当且仅当 dense top1 错误且 rerank top1 正确（rerank 能救回）；
    dense 已正确（rerank 冗余或有害）与双错（rerank 无用）均为 y=0。
    """

    def __init__(self, recall_k: int = 10, threshold: float = 0.5, **dense_kwargs):
        import joblib
        model_path = REPO_ROOT / "experiments" / "m3_learned_router" / "model.pkl"
        bundle = joblib.load(model_path)
        self.name = f"routed-learned"
        self.clf = bundle["model"]
        self.feature_names = bundle["features"]
        self.threshold = threshold
        self.recall_k = recall_k
        self.dense = DenseRetriever(**dense_kwargs)
        self.bm25 = BM25Retriever(dense_kwargs.get("index_dir", INDEX_DIR))
        self.reranker = _load_cross_encoder()
        self.stats = {"fast_path": 0, "slow_path": 0}

    @staticmethod
    def features(query: str, d_hits, b_ids) -> List[float]:
        """d_hits: [(chunk, score)] top5；b_ids: bm25 top5 chunk_id 列表"""
        import math
        d_ids = [c.chunk_id for c, _ in d_hits]
        scores = [s for _, s in d_hits] or [0.0]
        probs = [max(s, 0.0) for s in scores]
        total = sum(probs) or 1.0
        entropy = -sum((p / total) * math.log(p / total + 1e-12) for p in probs)
        overlap3 = len(set(d_ids[:3]) & set(b_ids[:3]))
        return [
            1.0 if (d_ids and b_ids and d_ids[0] == b_ids[0]) else 0.0,  # top1 一致
            overlap3 / 3.0,                                              # top3 重叠
            scores[0] - (scores[1] if len(scores) > 1 else 0.0),         # 首二间隔
            scores[0] - scores[-1],                                      # 首末间隔
            scores[0],                                                   # dense top1 分数
            entropy,                                                     # dense 分数熵
            sum(1 for ch in query if ch.isascii()) / max(len(query), 1),  # ascii 占比
            float(len(query)),                                           # 查询长度
        ]

    def retrieve(self, query: str, top_k: int = 5) -> List[Chunk]:
        d_hits = self.dense.retrieve_with_scores(query, top_k=5)
        b_ids = [c.chunk_id for c in self.bm25.retrieve(query, top_k=5)]
        x = [self.features(query, d_hits, b_ids)]
        need_rerank = bool(self.clf.predict_proba(x)[0, 1] >= self.threshold)

        if not need_rerank:
            self.stats["fast_path"] += 1
            return [c for c, _ in d_hits[:top_k]]

        self.stats["slow_path"] += 1
        candidates = [c for c, _ in self.dense.retrieve_with_scores(query, top_k=self.recall_k)]
        return _rerank_candidates(query, candidates, self.reranker, top_k)


class HierarchicalPlusRerank(BaseRetriever):
    """分层召回（细块->父粗块 top-R）喂 rerank。M4 组合臂。"""

    def __init__(self, recall_k: int = 10, **hier_kwargs):
        self.name = "hier+rerank"
        self.recall_k = recall_k
        self.hier = HierarchicalRetriever(**hier_kwargs)
        self.reranker = _load_cross_encoder()

    def retrieve(self, query: str, top_k: int = 5) -> List[Chunk]:
        candidates = self.hier.retrieve_coarse(query, top_k=self.recall_k, oversample=4)
        return _rerank_candidates(query, candidates, self.reranker, top_k)


class RoutedHierRetriever(BaseRetriever):
    """分层版一致性路由：hier 与 BM25 top1 一致 -> hier 直返（~14ms）；
    不一致 -> hybrid+rerank 慢路径（~185ms）。

    信号在 train 集选型：一致组 44% 占比、hier top1 正确率 0.906（先验 0.847）。
    """

    def __init__(self, **hier_kwargs):
        self.name = "routed-hier(agreement)"
        self.hier = HierarchicalRetriever(**hier_kwargs)
        self.bm25 = BM25Retriever(hier_kwargs.get("index_dir", INDEX_DIR))
        self.slow = HybridPlusRerank(**hier_kwargs)
        self.stats = {"fast_path": 0, "slow_path": 0}

    def retrieve(self, query: str, top_k: int = 5) -> List[Chunk]:
        h_hits = self.hier.retrieve(query, top_k=top_k)
        h_ids = [c.chunk_id for c in h_hits]
        b_ids = [c.chunk_id for c in self.bm25.retrieve(query, top_k=top_k)]
        if h_ids and b_ids and h_ids[0] == b_ids[0]:
            self.stats["fast_path"] += 1
            return h_hits
        self.stats["slow_path"] += 1
        return self.slow.retrieve(query, top_k=top_k)
