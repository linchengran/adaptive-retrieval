# -*- coding: utf-8 -*-
"""adaptive-retrieval 检索核心：自包含实现，不依赖 benchmark 研究代码。

三档位按语料类型自动选择实现（M6 语料感知，M5 跨域证据驱动）：
- 长文档语料（平均粗块 >= 375 字符）：
  fast = 分层细块召回 -> 父粗块返回；auto = hier 与 BM25 top1 一致走 fast 否则升级；
  best = 分层召回 + bge-reranker 重排
- 短段落语料（平均粗块 < 375 字符）：
  fast = 纯 dense；auto = best（该类语料上 agreement 路由 hit@1 -0.045，精度不安全）；
  best = dense + bge-reranker 重排（此语料上分层细切有害）

另提供 dense 档：单路向量、无重排，仅用于 eval 时作对照基线，不属于三档契约。

索引格式（<index_dir>/，schema_version=2）：
- coarse.json     粗块文本与元数据（chunk_id = "<相对路径>::<序号>"）
- fine.npz        细块向量 + 父索引（仅长文档语料构建）
- coarse_vecs.npy 粗块向量（float32，行序与 coarse.json 一致）
- meta.json       语料元特征、embedder、维度、分块参数、逐文档摘要（增量与校验用）
schema_version=1 的旧索引（无 meta.json）仍可读，按需自动降级。

可选策略（默认全关，用 `tune` 在自己的评测集上选出后再开，见 PRESETS）：
分数融合 fusion、候选池混合 candidates=hybrid、首跳 BM25-RRF fast_rrf。
实测这些开关收益高度依赖语料（同一改动 +0.109 ~ −0.150），故不设全局默认。

评测除 chunk 级 hit/recall/ndcg/MRR 外，还支持：
- answers -> answer_recall@k（上下文里是否含答案原文）
- --judge-sheet / --judge-results（agent 打 answer_correct / faithful）
`tune` 用配对 bootstrap + 前半选择/后半复验给出策略建议，避免小样本误判。

命令行统一支持 --json，输出为单行或缩进 JSON，便于 agent / 脚本消费。
"""
from __future__ import annotations

import ast
import hashlib
import json
import math
import os
import re
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

SCHEMA_VERSION = 2

FINE_SIZE = 120        # 细块目标字符数
COARSE_SIZE = 512      # 粗块目标字符数
COARSE_OVERLAP = 64    # 超长无分隔段落的滑动窗口重叠

# 语料分类阈值：短段落语料上分层细切会把完整短段落进一步打碎（M5/M7 跨域证据：
# DuReader 315 字上 hier hit@1 比 dense 低 0.07，Cmedqa 169 字上低 0.045；而
# cmrc2018 368.8 字上打平，T2Retrieval 400.8 字上 hier(+rerank) 反超 dense 0.01），
# 且短段落语料上 BM25 偏弱使 agreement 路由不再精度安全（cmedqa -0.010）。
# 阈值 375 由五域校准：368.8 处 hier 已无害、400.8 处有益，375 落在过渡带内；
# 误分代价不对称——短判长会启用有害的 hier 细切，故压线语料从短更稳。
SHORT_CORPUS_AVG_CHARS = 375

DEFAULT_EXTS = (".txt", ".md", ".markdown")
MAX_FILE_BYTES = 20 * 1024 * 1024
# BGE 中文模型的检索指令前缀；英文/多语语料拼中文指令会掉点，故按语料语言切换
ZH_QUERY_INSTRUCTION = "为这个句子生成表示以用于检索相关文章："

# 查询扩展用的停用词（疑问词 + 高频虚词），只用于 multiquery 的派生变体
_ZH_STOP = set("""
的 了 是 在 和 有 就 不 人 都 一 一个 上 也 很 到 说 要 去 你 我 他 她 它 吗 呢 吧 啊
与 及 或 对 中 为 以 而 其 这 那 些 什么 怎么 如何 怎样 为什么 哪些 哪个 是否 可以 能否
请问 一下 介绍 告诉 关于 方面 大概 一般 通常 具体 应该 需要 有没有 多少 几个
""".split())
_CJK = re.compile(r"[\u4e00-\u9fff]")

PROFILE_FAST, PROFILE_AUTO, PROFILE_BEST, PROFILE_DENSE = "fast", "auto", "best", "dense"
PROFILES = (PROFILE_FAST, PROFILE_AUTO, PROFILE_BEST)


# ---------------------------------------------------------------- 基础工具
def ensure_utf8_stdout() -> None:
    """Windows 控制台默认 cp936，中文结果会乱码，agent 侧解析也会崩。

    CLI 入口统一调用；已配置为 UTF-8 或非 tty 场景下无副作用。
    """
    for stream in (sys.stdout, sys.stderr):
        try:
            if hasattr(stream, "reconfigure"):
                stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass


@contextmanager
def _quiet_stdout():
    """把第三方库的 print/进度条重定向到 stderr。

    MCP server 必须在 stdout 上只写 JSON-RPC 帧，任何库 print 都会污染协议。
    """
    old = sys.stdout
    sys.stdout = sys.stderr
    try:
        yield
    finally:
        sys.stdout = old


def _set_low_noise_env() -> None:
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")


def _silence_hf_progress() -> None:
    """transformers 5.x 的 "Loading weights" 进度条走 tqdm，会污染 stderr 与日志。"""
    try:
        import transformers.utils.logging as hf_logging
        hf_logging.set_verbosity_error()
        hf_logging.disable_progress_bar()
    except Exception:
        pass


def corpus_stats(coarse: Sequence[dict]) -> dict:
    """语料元特征，索引时即可计算，是档位实现的选型信号"""
    lens = [len(c["text"]) for c in coarse] or [0]
    ordered = sorted(lens)
    mid = len(ordered) // 2
    median = ordered[mid] if len(ordered) % 2 else (ordered[mid - 1] + ordered[mid]) / 2
    return {
        "n_chunks": len(lens),
        "avg_chunk_chars": round(sum(lens) / len(lens), 1),
        "median_chunk_chars": round(float(median), 1),
        "total_chars": int(sum(lens)),
    }


def classify_corpus(stats: dict) -> str:
    return "short" if stats["avg_chunk_chars"] < SHORT_CORPUS_AVG_CHARS else "long"


def _is_ascii_heavy(texts: Iterable[str], threshold: float = 0.7) -> bool:
    text = "".join(texts)[:20000]
    if not text:
        return False
    return sum(1 for ch in text if ch.isascii()) / len(text) >= threshold


# ---------------------------------------------------------------- 建库
def chunk_coarse(text: str, size: int = COARSE_SIZE,
                 overlap: int = COARSE_OVERLAP) -> List[str]:
    """按段落优先递归切粗块（与课题 chunking 递归策略同思路的简化版）"""
    text = (text or "").replace("\r\n", "\n").replace("\r", "\n").strip()
    if not text:
        return []
    separators = ["\n\n", "\n", "。", "！", "？", ".", "!", "?", "，", ",", " ", ""]

    def split(txt: str, seps: List[str]) -> List[str]:
        if len(txt) <= size:
            return [txt]
        sep = ""
        parts: List[str] = []
        for i, cand in enumerate(seps):
            if cand and cand in txt:
                sep = cand
                parts = [p for p in txt.split(cand) if p.strip()]
                break
        else:
            i = len(seps) - 1
            step = max(1, size - overlap)
            parts = [txt[j:j + size] for j in range(0, len(txt), step)]
        # 合并相邻片段到 size 以内
        merged: List[str] = []
        buf: List[str] = []
        blen = 0
        for p in parts:
            if buf and blen + len(p) > size:
                merged.append(sep.join(buf))
                buf, blen = [], 0
            buf.append(p)
            blen += len(p) + len(sep)
        if buf:
            merged.append(sep.join(buf))
        # 仍有超长块则用下一级分隔符继续
        out: List[str] = []
        for c in merged:
            if len(c) > size and i + 1 < len(seps):
                out.extend(split(c, seps[i + 1:]))
            else:
                out.append(c)
        return [c.strip() for c in out if c.strip()]

    return split(text, separators)


def fine_chunks(coarse: Sequence[dict]) -> Tuple[List[str], List[int]]:
    """句级细切粗块：返回 (fine_texts, fine_parent)，parent 指向 coarse 行号"""
    fine_texts: List[str] = []
    fine_parent: List[int] = []
    for pi, c in enumerate(coarse):
        sentences = [s for s in re.split(r"(?<=[。！？!?])\s*", c["text"]) if s.strip()]
        buf: List[str] = []
        blen = 0
        for sent in sentences:
            if buf and blen + len(sent) > FINE_SIZE:
                fine_texts.append("".join(buf))
                fine_parent.append(pi)
                buf, blen = [], 0
            buf.append(sent)
            blen += len(sent)
        if buf:
            fine_texts.append("".join(buf))
            fine_parent.append(pi)
    return fine_texts, fine_parent


def discover_docs(docs_dir: str, recursive: bool = True,
                  exts: Sequence[str] = DEFAULT_EXTS,
                  exclude_dirs: Sequence[str] = ()) -> List[Path]:
    """递归发现文档。旧实现只扫顶层，子目录文档会被静默漏掉。"""
    root = Path(docs_dir).expanduser()
    if not root.exists():
        raise FileNotFoundError(f"文档目录不存在: {root}")
    skip = {Path(d).resolve() for d in exclude_dirs if d}
    extset = {e.lower() if e.startswith(".") else "." + e.lower() for e in exts}
    found: List[Path] = []
    patterns = ("*", "**/*") if recursive else ("*",)
    seen = set()
    for pat in patterns:
        for fp in root.glob(pat):
            if not fp.is_file():
                continue
            if fp.suffix.lower() not in extset:
                continue
            if any(part.startswith(".") for part in fp.relative_to(root).parts[:-1]):
                continue
            if any(s and s in fp.resolve().parents for s in skip):
                continue
            key = fp.resolve()
            if key in seen:
                continue
            seen.add(key)
            found.append(fp)
    return sorted(found, key=lambda p: p.relative_to(root).as_posix())


def _file_sha1(path: Path) -> str:
    h = hashlib.sha1()
    with path.open("rb") as f:
        while True:
            block = f.read(1024 * 1024)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def _find_local_model(name: str) -> Optional[str]:
    """按 env -> 仓库 models/ -> 逐级向上查找 的顺序定位本地模型副本"""
    env = os.environ.get("AR_MODEL_DIR") or os.environ.get("ADAPTIVE_RETRIEVAL_MODEL_DIR")
    if env:
        p = Path(env).expanduser()
        if p.exists():
            return str(p)
    here = Path(__file__).resolve()
    for base in list(here.parents)[:6]:
        cand = base / "models" / name
        if cand.exists():
            return str(cand)
    return None


class _StubEmbedder:
    """无模型依赖的确定性占位编码器。

    仅供 CI / 自测（AR_EMBEDDER=stub）：用字符 n-gram 哈希出向量，
    能验证索引格式、路由与 JSON 契约，但没有任何语义检索能力。
    """

    def __init__(self, dim: int = 256):
        self.dim = dim

    def encode(self, texts, normalize_embeddings: bool = True,
               show_progress_bar: bool = False, batch_size: int = 32):
        vecs = []
        for text in texts:
            v = np.zeros(self.dim, dtype="float32")
            for i in range(len(text) - 1):
                gram = text[i:i + 2]
                v[hash(gram) % self.dim] += 1.0
            if not v.any():
                v[0] = 1.0
            if normalize_embeddings:
                n = float(np.linalg.norm(v)) or 1.0
                v = v / n
            vecs.append(v)
        return np.asarray(vecs, dtype="float32")


class _StubReranker:
    """无模型依赖的占位重排器（AR_EMBEDDER=stub 时使用）。

    仅按查询与候选的字符二元组重合度打分，让 auto/best 的重排分支在自测里
    也能被真实走到（而不是被整体跳过）。
    """

    def predict(self, pairs, show_progress_bar: bool = False):
        scores = []
        for q, text in pairs:
            qgrams = {q[i:i + 2] for i in range(max(len(q) - 1, 1))}
            tgrams = {text[i:i + 2] for i in range(max(len(text) - 1, 1))}
            overlap = len(qgrams & tgrams)
            scores.append(overlap / (len(qgrams) or 1))
        return np.asarray(scores, dtype="float32")


def _stub_mode() -> bool:
    return os.environ.get("AR_EMBEDDER", "").lower() == "stub"


# 同一进程内复用模型：build_index / Retriever / eval 连续调用时不要反复加载
_MODEL_CACHE: Dict[Tuple[str, str, str], object] = {}


def _load_embedder(model_dir: str = None, device: str = None):
    if _stub_mode():
        return _StubEmbedder()
    name = model_dir or _find_local_model("bge-small-zh-v1.5") or "BAAI/bge-small-zh-v1.5"
    if device is None:
        try:
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            device = "cpu"
    key = ("embedder", str(name), str(device))
    cached = _MODEL_CACHE.get(key)
    if cached is not None:
        return cached
    _set_low_noise_env()
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(name, device=device)
    _MODEL_CACHE[key] = model
    return model


def _embedder_id(name: str) -> str:
    p = Path(name)
    return p.name if p.exists() else name


def _encode(model, texts: Sequence[str], batch_size: int = 32) -> np.ndarray:
    if not texts:
        return np.zeros((0, 1), dtype="float32")
    vecs = model.encode(list(texts), normalize_embeddings=True,
                        show_progress_bar=False, batch_size=batch_size)
    return np.ascontiguousarray(np.asarray(vecs, dtype="float32"))


def _meta_matches(old_meta: Optional[dict], embedder_id: str, dim: int,
                  params: dict, corpus_type: str) -> bool:
    if not old_meta or old_meta.get("schema_version") != SCHEMA_VERSION:
        return False
    return (
        old_meta.get("embedder_id") == embedder_id
        and old_meta.get("dim") == dim
        and old_meta.get("chunk") == params
        and old_meta.get("corpus_type") == corpus_type
    )


def build_index(docs_dir: str, index_dir: str, model_dir: str = None,
                recursive: bool = True, exts: Sequence[str] = DEFAULT_EXTS,
                incremental: bool = True, force: bool = False,
                device: str = None, quiet: bool = False,
                batch_size: int = 32) -> dict:
    """读取目录内文档，切块编码，落盘索引。零配置入口。

    返回索引摘要（与 meta.json 内容一致），便于 CLI --json 直接输出。
    """
    out = Path(index_dir).expanduser()
    docs = discover_docs(docs_dir, recursive=recursive, exts=exts,
                         exclude_dirs=(str(out),))
    if not docs:
        raise FileNotFoundError(f"{docs_dir} 下没有匹配 {list(exts)} 的文档")

    root = Path(docs_dir).expanduser()
    params = {"coarse_size": COARSE_SIZE, "coarse_overlap": COARSE_OVERLAP,
              "fine_size": FINE_SIZE, "schema": SCHEMA_VERSION}

    model = _load_embedder(model_dir, device)
    embedder_id = "stub" if isinstance(model, _StubEmbedder) else _embedder_id(
        model_dir or _find_local_model("bge-small-zh-v1.5") or "BAAI/bge-small-zh-v1.5")

    # ---- 第一步：只做文本切块（便宜），确定语料类型 ----
    per_doc_chunks: Dict[str, List[str]] = {}
    doc_info: Dict[str, dict] = {}
    for fp in docs:
        rel = fp.relative_to(root).as_posix()
        size = fp.stat().st_size
        if size > MAX_FILE_BYTES:
            if not quiet:
                print(f"[跳过] {rel} 超过 {MAX_FILE_BYTES // 1024 // 1024}MB", file=sys.stderr)
            continue
        text = fp.read_text(encoding="utf-8", errors="ignore")
        chunks = chunk_coarse(text)
        if not chunks:
            continue
        per_doc_chunks[rel] = chunks
        doc_info[rel] = {"sha1": _file_sha1(fp), "bytes": size, "chunks": len(chunks)}
    if not per_doc_chunks:
        raise ValueError(f"{docs_dir} 下文档均为空，无法建索引")

    flat_coarse = [{"text": t, "source": rel, "chunk_index": i,
                    "chunk_id": f"{rel}::{i}"}
                   for rel, chunks in sorted(per_doc_chunks.items())
                   for i, t in enumerate(chunks)]
    stats = corpus_stats(flat_coarse)
    corpus_type = classify_corpus(stats)
    fine_texts, fine_parent = fine_chunks(flat_coarse) if corpus_type == "long" else ([], [])

    # ---- 第二步：尝试复用旧索引向量（增量） ----
    old_meta = None
    old_coarse: List[dict] = []
    old_coarse_vecs = None
    old_fine_vecs = None
    old_fine_parent = None
    if incremental and not force and (out / "coarse.json").exists():
        try:
            old_coarse = json.loads((out / "coarse.json").read_text(encoding="utf-8-sig"))
            old_coarse_vecs = np.load(out / "coarse_vecs.npy")
            if (out / "meta.json").exists():
                old_meta = json.loads((out / "meta.json").read_text(encoding="utf-8-sig"))
            if (out / "fine.npz").exists():
                _fine = np.load(out / "fine.npz")
                old_fine_vecs = _fine["fine_vecs"]
                old_fine_parent = _fine["fine_parent"]
        except Exception as exc:  # 旧索引损坏则不冒险，直接全量重建
            if not quiet:
                print(f"[提示] 旧索引不可复用（{exc}），改为全量重建", file=sys.stderr)
            old_meta, old_coarse, old_coarse_vecs = None, [], None
            old_fine_vecs = old_fine_parent = None

    reuse_ok = bool(
        old_meta and old_coarse_vecs is not None and old_coarse_vecs.ndim == 2
        and len(old_coarse) == len(old_coarse_vecs)
        and _meta_matches(old_meta, embedder_id, old_coarse_vecs.shape[1], params, corpus_type)
    )

    old_rows_by_doc: Dict[str, List[int]] = {}
    old_fine_by_doc: Dict[str, List[int]] = {}
    if reuse_ok:
        for i, c in enumerate(old_coarse):
            old_rows_by_doc.setdefault(c.get("source", ""), []).append(i)
        if old_fine_parent is not None:
            for fi, pi in enumerate(old_fine_parent):
                src = old_coarse[int(pi)].get("source", "")
                old_fine_by_doc.setdefault(src, []).append(fi)

    def _reuse_doc(rel: str) -> bool:
        if not reuse_ok:
            return False
        info = doc_info[rel]
        prev = (old_meta or {}).get("docs", {}).get(rel)
        if not prev or prev.get("sha1") != info["sha1"]:
            return False
        return len(old_rows_by_doc.get(rel, [])) == info["chunks"]

    # ---- 第三步：编码（只编码新增/变更文档） ----
    coarse_blocks: List[np.ndarray] = []
    fine_blocks: List[np.ndarray] = []
    reused_docs, encoded_docs = 0, 0
    dim = 0

    fine_idx_by_doc: Dict[str, List[int]] = {}
    for fi, pi in enumerate(fine_parent):
        fine_idx_by_doc.setdefault(flat_coarse[pi]["source"], []).append(fi)

    for rel, chunks in sorted(per_doc_chunks.items()):
        reuse = _reuse_doc(rel)
        if reuse:
            coarse_blocks.append(old_coarse_vecs[old_rows_by_doc[rel]])
            dim = old_coarse_vecs.shape[1]
            reused_docs += 1
        else:
            vecs = _encode(model, chunks, batch_size)
            coarse_blocks.append(vecs)
            dim = vecs.shape[1]
            encoded_docs += 1

        if corpus_type != "long":
            continue
        idxs_here = fine_idx_by_doc.get(rel, [])
        can_reuse_fine = (
            reuse and old_fine_vecs is not None
            and len(old_fine_by_doc.get(rel, [])) == len(idxs_here)
        )
        if can_reuse_fine:
            fine_blocks.append(old_fine_vecs[old_fine_by_doc[rel]])
        elif idxs_here:
            fine_blocks.append(_encode(model, [fine_texts[i] for i in idxs_here], batch_size))

    coarse_vecs = np.vstack([b for b in coarse_blocks if b.size]) if coarse_blocks \
        else np.zeros((0, dim), dtype="float32")
    coarse_vecs = np.ascontiguousarray(coarse_vecs.astype("float32"))

    # ---- 第四步：落盘 ----
    out.mkdir(parents=True, exist_ok=True)
    (out / "coarse.json").write_text(
        json.dumps(flat_coarse, ensure_ascii=False, indent=1), encoding="utf-8")
    np.save(out / "coarse_vecs.npy", coarse_vecs)
    if corpus_type == "long":
        fine_vecs = np.vstack([b for b in fine_blocks if b.size])
        np.savez_compressed(out / "fine.npz",
                            fine_vecs=np.ascontiguousarray(fine_vecs.astype("float32")),
                            fine_parent=np.asarray(fine_parent, dtype="int64"))
    elif (out / "fine.npz").exists():
        # 语料类型从 long 翻到 short 时必须清掉旧细块，否则会按旧索引误用
        (out / "fine.npz").unlink()

    meta = {
        "schema_version": SCHEMA_VERSION,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "corpus_type": corpus_type,
        "n_chunks": len(flat_coarse),
        "n_fine_chunks": len(fine_texts),
        "embedder_id": embedder_id,
        "dim": int(dim),
        "chunk": params,
        "docs": doc_info,
        "reused_docs": reused_docs,
        "encoded_docs": encoded_docs,
        **stats,
    }
    (out / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1),
                                   encoding="utf-8")
    if not quiet:
        print(f"索引完成: {len(flat_coarse)} 粗块 / {len(fine_texts)} 细块 "
              f"[{corpus_type} 语料, 平均 {stats['avg_chunk_chars']} 字符, "
              f"复用 {reused_docs} 篇 / 编码 {encoded_docs} 篇] -> {out}")
    return meta


# ---------------------------------------------------------------- 检索
class Retriever:
    """加载索引并执行三档检索。模型懒加载：只读元信息（stats/eval 准备）不触发编码器加载。"""

    def __init__(self, index_dir: str, model_dir: str = None, device: str = None,
                 lazy_embedder: bool = True, disable_rerank: bool = False,
                 options: Optional[dict] = None):
        self.dir = Path(index_dir).expanduser()
        if not (self.dir / "coarse.json").exists():
            raise FileNotFoundError(
                f"索引不存在: {self.dir}/coarse.json；先运行 "
                f"`retrieval_core.py index <文档目录> --index-dir {self.dir}`")
        self.coarse: List[dict] = json.loads(
            (self.dir / "coarse.json").read_text(encoding="utf-8-sig"))
        self.coarse_vecs = np.load(self.dir / "coarse_vecs.npy")
        if self.coarse_vecs.ndim != 2 or len(self.coarse_vecs) != len(self.coarse):
            raise ValueError(
                f"索引损坏：coarse_vecs.npy 行数 {len(self.coarse_vecs)} 与 "
                f"coarse.json 条数 {len(self.coarse)} 不一致，请重建索引")
        self._model_dir = model_dir
        self._device = device
        self._disable_rerank = disable_rerank or os.environ.get("AR_DISABLE_RERANK") == "1"
        self._embedder = None
        self._bm25_obj = None
        self._bm25_tok = None
        self._reranker_obj = None
        self.load_ms = 0.0   # 累计模型加载耗时，供 trace 区分"冷启动"与"稳态延迟"

        # 可选策略：默认全关（保持已验证的冠军组合）。用 `tune` 在自己的评测集上选出后再开，
        # 因为这些开关的收益高度依赖语料域（实测见 experiments/_opt/）。
        self.options: Dict[str, Any] = {"fusion": None, "fast_rrf": False, "candidates": "first"}
        if options:
            self.options.update({k: v for k, v in options.items() if v is not None})

        self.meta: Optional[dict] = None
        if (self.dir / "meta.json").exists():
            try:
                self.meta = json.loads((self.dir / "meta.json").read_text(encoding="utf-8-sig"))
            except Exception:
                self.meta = None

        # 语料分类：优先信 meta（索引时一次判定），旧索引则从自身数据推算
        self.stats = corpus_stats(self.coarse)
        self.corpus_type = (self.meta or {}).get("corpus_type") or classify_corpus(self.stats)
        self.dim = int(self.coarse_vecs.shape[1])

        fine_path = self.dir / "fine.npz"
        if fine_path.exists():
            fine = np.load(fine_path)
            self.fine_vecs, self.fine_parent = fine["fine_vecs"], fine["fine_parent"]
            if len(self.fine_parent) != len(self.fine_vecs):
                raise ValueError(f"索引损坏：fine_vecs 与 fine_parent 长度不一致（{fine_path}）")
        else:
            self.fine_vecs, self.fine_parent = None, None
        if self.corpus_type == "long" and self.fine_vecs is None:
            # 语料被判为长文档却没有细块（旧索引/被手工删过）-> 退化为 dense 并记录原因
            self.degraded = "missing_fine_index"
        else:
            self.degraded = None

        self._zh_corpus = not _is_ascii_heavy(c["text"] for c in self.coarse[:200])
        if lazy_embedder:
            pass
        else:
            self._ensure_embedder()

    # --- 资源 ---
    def _ensure_embedder(self):
        if self._embedder is None:
            t0 = time.perf_counter()
            self._embedder = _load_embedder(self._model_dir, self._device)
            if _stub_mode() and isinstance(self._embedder, _StubEmbedder):
                # 自测用的 stub 需要对齐索引维度，否则读真实模型建的索引会因维度校验失败
                self._embedder.dim = self.dim
            self.load_ms += _ms(t0)
            if not _stub_mode():
                # 注意：这一步会 import transformers（数秒），必须放在计时之后
                _silence_hf_progress()
            got = int(getattr(self._embedder, "dim", 0) or 0)
            if not got:
                import warnings
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    for attr in ("get_embedding_dimension", "get_sentence_embedding_dimension"):
                        fn = getattr(self._embedder, attr, None)
                        if fn is None:
                            continue
                        try:
                            got = int(fn())
                            break
                        except Exception:
                            continue
            if got and got != self.dim:
                raise ValueError(
                    f"embedder 维度 {got} 与索引维度 {self.dim} 不一致："
                    f"模型换了必须重建索引（index --force）")
        return self._embedder

    @property
    def embedder(self):
        return self._ensure_embedder()

    def _query_prefix(self) -> str:
        # 中文语料拼中文指令（BGE 中文模型的官方用法）；英文语料拼中文指令实测会掉点
        return ZH_QUERY_INSTRUCTION if self._zh_corpus else ""

    def _embed_query(self, q: str) -> np.ndarray:
        vec = self._encode_queries([self._query_prefix() + q])
        return np.ascontiguousarray(vec[0])

    def _encode_queries(self, texts: Sequence[str]) -> np.ndarray:
        model = self._ensure_embedder()
        vecs = model.encode(list(texts), normalize_embeddings=True,
                            show_progress_bar=False)
        return np.asarray(vecs, dtype="float32")

    def tokenize(self, text: str) -> List[str]:
        if self._is_english(text):
            return re.findall(r"[a-z0-9_]+", text.lower())
        import jieba
        jieba.setLogLevel(60)
        return jieba.lcut(text.lower())

    @staticmethod
    def _is_english(text: str) -> bool:
        letters = [c for c in text if c.isalpha()]
        if not letters:
            return False
        return sum(1 for c in letters if c.isascii()) / len(letters) > 0.8

    def _get_bm25(self):
        if self._bm25_obj is None:
            from rank_bm25 import BM25Okapi
            corpus_tokens = [self.tokenize(c["text"]) for c in self.coarse]
            self._bm25_obj = BM25Okapi(corpus_tokens)
        return self._bm25_obj

    def _get_reranker(self):
        if self._disable_rerank:
            return None
        if _stub_mode():
            if self._reranker_obj is None:
                self._reranker_obj = _StubReranker()
            return self._reranker_obj
        if self._reranker_obj is None:
            try:
                _set_low_noise_env()
                from sentence_transformers import CrossEncoder
                # 允许挂领域微调过的重排器：AR_RERANKER_DIR 指向微调产物目录。
                # 实测（docs/优化-按影响程度排序.md §11）：在 DuReader 上微调后
                # hit@1 0.615 -> 0.675（+0.060，95%CI [+0.005,+0.120]），
                # 因为该域候选池 recall@10=1.000，是纯粹的排序瓶颈。
                tuned = os.environ.get("AR_RERANKER_DIR")
                local = tuned or _find_local_model("bge-reranker-base")
                name = local or "BAAI/bge-reranker-base"
                device = self._device
                if device is None:
                    try:
                        import torch
                        device = "cuda" if torch.cuda.is_available() else "cpu"
                    except Exception:
                        device = "cpu"
                _t0 = time.perf_counter()
                rkey = ("reranker", str(name), str(device))
                cached = _MODEL_CACHE.get(rkey)
                if cached is not None:
                    self._reranker_obj = cached
                else:
                    self._reranker_obj = CrossEncoder(name, max_length=512, device=device,
                                                      num_labels=1)
                    _MODEL_CACHE[rkey] = self._reranker_obj
                _silence_hf_progress()
                self.load_ms += _ms(_t0)
            except Exception as exc:
                # 重排器不可用不应让检索整体失败：降级为不重排并记录
                self._reranker_obj = False
                self.degraded = f"reranker_unavailable: {exc}"
        return self._reranker_obj or None

    # --- 通道 ---
    def _topk(self, scores: np.ndarray, k: int) -> np.ndarray:
        if len(scores) <= k:
            return np.argsort(scores)[::-1]
        part = np.argpartition(scores, -k)[-k:]
        return part[np.argsort(scores[part])[::-1]]

    def dense_top(self, q: str, k: int = 5) -> List[Tuple[dict, float]]:
        scores = self.coarse_vecs @ self._embed_query(q)
        return [(self.coarse[i], float(scores[i])) for i in self._topk(scores, k)]

    def hier_top(self, q: str, k: int = 5) -> List[Tuple[dict, float]]:
        if self.fine_vecs is None:
            raise RuntimeError(
                "该索引为短段落语料构建（未含细块），本语料上 fast 档已自动改用 dense")
        scores = self.fine_vecs @ self._embed_query(q)
        order = np.argsort(scores)[::-1]
        out: List[Tuple[dict, float]] = []
        seen = set()
        limit = min(len(order), max(k * 3, k))
        start = 0
        while len(out) < k and start < len(order):
            for i in order[start:limit]:
                pi = int(self.fine_parent[i])
                if pi in seen:
                    continue
                seen.add(pi)
                out.append((self.coarse[pi], float(scores[i])))
                if len(out) >= k:
                    break
            start = limit
            limit = min(len(order), max(limit * 2, k))
            if start >= len(order):
                break
        return out

    def bm25_top(self, q: str, k: int = 5) -> List[Tuple[dict, float]]:
        bm25 = self._get_bm25()
        scores = np.asarray(bm25.get_scores(self.tokenize(q)), dtype="float32")
        if not len(scores) or float(scores.max()) <= 0:
            return []
        return [(self.coarse[i], float(scores[i])) for i in self._topk(scores, k)]

    def rerank(self, q: str, candidates: Sequence[Tuple[dict, float]],
               k: int = 5) -> List[Tuple[dict, float]]:
        if not candidates:
            return []
        reranker = self._get_reranker()
        if reranker is None:
            return list(candidates)[:k]
        scores = reranker.predict([[q, c["text"]] for c, _ in candidates],
                                  show_progress_bar=False)
        order = np.argsort(np.asarray(scores))[::-1][:k]
        return [(candidates[i][0], float(scores[i])) for i in order]

    # --- 可选策略（tune 选出后由 options 打开） ---
    @staticmethod
    def _rrf_merge(a: Sequence[Tuple[dict, float]], b: Sequence[Tuple[dict, float]],
                   k_rrf: int = 60) -> List[Tuple[dict, float]]:
        score: Dict[str, float] = {}
        keep: Dict[str, dict] = {}
        for rank, (c, _) in enumerate(a, 1):
            score[c["chunk_id"]] = score.get(c["chunk_id"], 0.0) + 1.0 / (k_rrf + rank)
            keep[c["chunk_id"]] = c
        for rank, (c, _) in enumerate(b, 1):
            score[c["chunk_id"]] = score.get(c["chunk_id"], 0.0) + 1.0 / (k_rrf + rank)
            keep.setdefault(c["chunk_id"], c)
        return [(keep[cid], s) for cid, s in
                sorted(score.items(), key=lambda kv: kv[1], reverse=True)]

    def first_pass(self, q: str, k: int = 5) -> List[Tuple[dict, float]]:
        """第一跳：长语料走细块→父块，短语料走 dense（也就是 fast 档的口径）。"""
        if self.corpus_type == "short" or self.fine_vecs is None:
            return self.dense_top(q, k)
        return self.hier_top(q, k)

    def fast_pairs(self, q: str, k: int = 5) -> List[Tuple[dict, float]]:
        if not self.options.get("fast_rrf"):
            return self.first_pass(q, k)[:k]
        return self._rrf_merge(self.first_pass(q, 10), self.bm25_top(q, 10))[:k]

    def slow_candidates(self, q: str, pool: int) -> List[Tuple[dict, float]]:
        """慢路径候选池。multiquery 打开时先做零模型查询扩展再合并。"""
        base = (self._rrf_merge(self.first_pass(q, pool), self.bm25_top(q, pool))[:pool]
                if self.options.get("candidates") == "hybrid"
                else self.first_pass(q, pool))
        if not self.options.get("multiquery"):
            return base
        variants = self.expand_queries(q, self.tokenize)
        if not variants:
            return base
        merged = list(base)
        for v in variants:
            extra = (self.hier_top(v, pool) if (self.corpus_type == "long"
                                                and self.fine_vecs is not None)
                     else self.dense_top(v, pool))
            merged = self._rrf_merge(merged, extra)
        return merged[:pool]

    @staticmethod
    def expand_queries(q: str, tokenize: Callable[[str], List[str]],
                       max_terms: int = 3, max_variants: int = 2) -> List[str]:
        """零模型的查询扩展：从原查询派生改写变体。

        不是 LLM 改写（那要额外调用、还要付延迟），而是三条便宜的派生规则：
        去疑问/虚词只留实词、取最长实词子集、原文后补关键词。
        针对的是"问句里塞太多虚词、向量被稀释"这类漏检。
        只在慢路径用，代价是 1-2 次额外编码（实测 +20ms 左右）。
        """
        tokens = [t for t in tokenize(q) if t.strip()]
        content = [t for t in tokens
                   if t not in _ZH_STOP and not t.isdigit() and len(t) >= 2]
        if not content:
            return []
        variants: List[str] = []
        stripped = " ".join(content)
        if stripped and stripped != q:
            variants.append(stripped)
        longs = sorted({t for t in content}, key=len, reverse=True)[:max_terms]
        if longs:
            v = " ".join(longs)
            if v not in variants and v != q and v != stripped:
                variants.append(v)
        return variants[:max_variants]

    def _ensemble_scores(self, q: str, texts: Sequence[str], reranker) -> np.ndarray:
        """重排打分。rerank_ensemble 打开时对同一批候选做多视角打分再平均。

        动机（benchmark/ceiling_diag.py）：候选池 recall@10 已达 0.98-1.00，
        说明瓶颈不在召回而在排序；于是尝试让 cross-encoder 从多个查询视角看同一对。
        实测（benchmark/rerank_query_probe.py，200 条/域）：
          dureader 0.615 -> 0.640（+0.025，四域里唯一明确收益）
          cmedqa   0.600 -> 0.570（−0.030）
          t2       0.965 -> 0.960（−0.005）
        域相关，故默认关闭，由 `tune` 决定是否启用。
        """
        if not self.options.get("rerank_ensemble"):
            return np.asarray(reranker.predict([[q, t] for t in texts],
                                              show_progress_bar=False), dtype="float32")
        variants = [q] + self.expand_queries(q, self.tokenize)
        accum = np.zeros(len(texts), dtype="float32")
        for v in variants:
            accum += np.asarray(reranker.predict([[v, t] for t in texts],
                                                 show_progress_bar=False), dtype="float32")
        return accum / len(variants)

    def rerank(self, q: str, candidates: Sequence[Tuple[dict, float]],
               k: int = 5, fusion: Optional[float] = None) -> List[Tuple[dict, float]]:
        """重排。fusion=None/1.0 为纯重排；0<fusion<1 时与召回分做 z-score 融合。

        融合的收益高度依赖语料：t2 上 w=0.7 让 hit@1 从 0.965 到 0.985，
        wx 上同样权重从 0.709 掉到 0.582。故默认不启用，由 `tune` 决定。
        """
        if not candidates:
            return []
        reranker = self._get_reranker()
        if reranker is None:
            return list(candidates)[:k]
        texts = [c["text"] for c, _ in candidates]
        scores = self._ensemble_scores(q, texts, reranker)
        w = self.options.get("fusion") if fusion is None else fusion
        if w is None or float(w) >= 1.0:
            order = np.argsort(scores)[::-1][:k]
            return [(candidates[i][0], float(scores[i])) for i in order]

        def _z(v: np.ndarray) -> np.ndarray:
            v = np.asarray(v, dtype="float64")
            s = v.std()
            return (v - v.mean()) / (s if s > 1e-9 else 1.0)

        recall = np.asarray([s for _, s in candidates], dtype="float32")
        fused = (1.0 - float(w)) * _z(recall) + float(w) * _z(scores)
        order = np.argsort(fused)[::-1][:k]
        return [(candidates[i][0], float(fused[i])) for i in order]

    # --- 三档位 ---
    def retrieve_with_trace(self, q: str, k: int = 5,
                            profile: str = PROFILE_AUTO) -> Tuple[List[dict], dict]:
        if profile not in (PROFILE_FAST, PROFILE_AUTO, PROFILE_BEST, PROFILE_DENSE):
            raise ValueError(f"未知档位: {profile}；可选 {PROFILES}（dense 仅作对照基线）")
        k = max(1, min(int(k), len(self.coarse)))
        short = self.corpus_type == "short"
        trace: Dict[str, Any] = {"profile": profile, "corpus_type": self.corpus_type,
                                 "route": None, "degraded": self.degraded}
        t0 = time.perf_counter()

        def _pack(pairs: Sequence[Tuple[dict, float]], stage: str) -> List[dict]:
            out = []
            for rank, (c, score) in enumerate(pairs, 1):
                out.append({
                    "rank": rank,
                    "chunk_id": c["chunk_id"],
                    "text": c["text"],
                    "source": c.get("source", ""),
                    "chunk_index": c.get("chunk_index"),
                    "score": round(float(score), 6),
                    "stage": stage,
                })
            return out

        def _timings() -> dict:
            return {"total_ms": _ms(t0), "model_load_ms": round(self.load_ms, 2)}

        if profile == PROFILE_DENSE:
            trace["route"] = "dense"
            res = _pack(self.dense_top(q, k), "dense")
        elif profile == PROFILE_FAST:
            if short or self.fine_vecs is None:
                trace["route"] = "dense"
            else:
                trace["route"] = "hier"
            if self.options.get("fast_rrf"):
                trace["route"] = f"{trace['route']}+bm25(rrf)"
            res = _pack(self.fast_pairs(q, k), trace["route"])
        else:
            if profile == PROFILE_AUTO and not short and self.fine_vecs is not None:
                h = self.hier_top(q, k)
                b = self.bm25_top(q, k)
                agree = bool(h and b and h[0][0]["chunk_id"] == b[0][0]["chunk_id"])
                trace["agreement"] = agree
                if agree:
                    trace["route"] = "fast_path(agreement)"
                    trace["timings"] = _timings()
                    return _pack(h, "hier"), trace
            cands_k = max(k * 2, 10)
            if short or self.fine_vecs is None:
                trace["route"] = "dense+rerank"
            else:
                trace["route"] = "hier+rerank"
            cands = self.slow_candidates(q, cands_k)
            if self.options.get("candidates") == "hybrid":
                trace["route"] = trace["route"].replace("+rerank", "+rrf+rerank")
            if self.options.get("fusion") is not None and float(self.options["fusion"]) < 1.0:
                trace["route"] += f"(fusion={self.options['fusion']})"
            reranker = self._get_reranker()
            if reranker is None:
                trace["rerank"] = "disabled_or_unavailable"
                res = _pack(cands[:k], trace["route"])
            else:
                res = _pack(self.rerank(q, cands, k), "rerank")
        trace["timings"] = _timings()
        trace["n_returned"] = len(res)
        return res, trace

    def retrieve(self, q: str, k: int = 5, profile: str = PROFILE_AUTO) -> List[dict]:
        """返回 [{"rank","chunk_id","text","source","chunk_index","score","stage"}]。

        与旧版返回的 chunk dict 兼容（chunk_id/text/source 字段不变），额外带分数与命中阶段。
        """
        return self.retrieve_with_trace(q, k=k, profile=profile)[0]

    def retrieve_batch(self, queries: Sequence[str], k: int = 5,
                       profile: str = PROFILE_AUTO) -> List[List[dict]]:
        return [self.retrieve(q, k=k, profile=profile) for q in queries]

    # --- 元信息 ---
    def describe(self) -> dict:
        return {
            "index_dir": str(self.dir),
            "schema_version": (self.meta or {}).get("schema_version", 1),
            "corpus_type": self.corpus_type,
            "n_chunks": len(self.coarse),
            "n_fine_chunks": 0 if self.fine_vecs is None else int(len(self.fine_vecs)),
            "avg_chunk_chars": self.stats["avg_chunk_chars"],
            "median_chunk_chars": self.stats["median_chunk_chars"],
            "total_chars": self.stats["total_chars"],
            "dim": self.dim,
            "embedder_id": (self.meta or {}).get("embedder_id"),
            "zh_corpus": self._zh_corpus,
            "degraded": self.degraded,
            "sources": sorted({c.get("source", "") for c in self.coarse}),
        }


def _ms(t0: float) -> float:
    return round((time.perf_counter() - t0) * 1000, 2)


# ---------------------------------------------------------------- 评测
def load_testset(path: str) -> List[dict]:
    """评测集格式：JSON list 或 {"items": [...]}，每条需含
    query/question 字段与 relevant/relevant_chunks 字段（chunk_id 列表）。

    可选 `answers`（字符串或列表）：给了就会额外算 answer_recall@k——
    检索到的上下文里是否真的包含答案原文，比 chunk 级 hit 更接近"能不能答对"。
    """
    data = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    items = data["items"] if isinstance(data, dict) else data
    out = []
    for i, it in enumerate(items):
        q = it.get("query") or it.get("question")
        rel = it.get("relevant") or it.get("relevant_chunks") or []
        if isinstance(rel, str):
            # 常见格式坑：relevant 存成了 "['a::0']" 这样的字符串
            text = rel.strip()
            try:
                parsed = ast.literal_eval(text)
                rel = list(parsed) if isinstance(parsed, (list, tuple, set)) else [text]
            except Exception:
                rel = [x.strip().strip("'\"") for x in text.strip("[]").split(",") if x.strip()]
        if not q or not rel:
            raise ValueError(
                f"评测集第 {i + 1} 条需含 query/question 与 relevant/relevant_chunks 字段")
        answers = it.get("answers") or it.get("answer") or []
        if isinstance(answers, str):
            answers = [answers]
        out.append({"query": q, "relevant": set(map(str, rel)),
                    "answers": [str(a) for a in answers if str(a).strip()]})
    return out


def _norm_text(s: str) -> str:
    """答案比对前归一化：去空白与标点、转小写（中文标点也去掉）。"""
    return re.sub(r"[\s\W_]+", "", str(s).lower())


def _contains_answer(contexts: Sequence[str], answers: Sequence[str]) -> bool:
    if not answers:
        return False
    haystack = _norm_text("".join(contexts))
    return any(_norm_text(a) and _norm_text(a) in haystack for a in answers)



def _ndcg(ids: Sequence[str], relevant: set, k: int) -> float:
    dcg = sum(1.0 / math.log2(r + 1) for r, cid in enumerate(ids[:k], 1) if cid in relevant)
    ideal = sum(1.0 / math.log2(r + 1) for r in range(1, min(len(relevant), k) + 1))
    return dcg / ideal if ideal else 0.0


def eval_index(index_dir: str, testset_path: str, profiles: Sequence[str] = PROFILES,
               k_values: Sequence[int] = (1, 5), model_dir: str = None,
               with_baseline: bool = True, quiet: bool = False,
               per_query: bool = False) -> dict:
    """离线评测：对已建索引逐档位算 hit@k / recall@k / mrr@k / ndcg@k。

    with_baseline=True 时额外跑 dense 档（单路向量、无重排）作为对照，
    这样用户不用自己拼基线口径就能看到"升级档位到底赚了多少"。
    """
    ret = Retriever(index_dir, model_dir)
    items = load_testset(testset_path)
    # 相关 id 必须是本索引内真实存在的 chunk_id：写错会静默得到 0 分，是最常见的评测坑
    known = {c["chunk_id"] for c in ret.coarse}
    unknown = sorted({cid for it in items for cid in it["relevant"] if cid not in known})
    if unknown:
        raise ValueError(
            "评测集里有 " + str(len(unknown)) + " 个 chunk_id 不在当前索引中，例如 "
            + ", ".join(unknown[:5])
            + "；跨系统对比请先用 benchmark/convert_to_skill_index.py 统一 chunk 口径")
    k_max = max(k_values)
    arm_list = list(profiles)
    if with_baseline and PROFILE_DENSE not in arm_list:
        arm_list.append(PROFILE_DENSE)
    if not quiet:
        print(f"[{ret.corpus_type} 语料: {len(ret.coarse)} 粗块, 平均 "
              f"{ret.stats['avg_chunk_chars']} 字符]  评测集: {len(items)} 条查询")

    results: Dict[str, dict] = {}
    has_answers = any(it.get("answers") for it in items)
    text_by_id = {c["chunk_id"]: c["text"] for c in ret.coarse}
    for prof in arm_list:
        hits = {k: 0 for k in k_values}
        recalls = {k: 0.0 for k in k_values}
        ndcgs = {k: 0.0 for k in k_values}
        answer_hits = {k: 0 for k in k_values}
        answer_n = 0
        rr = 0.0
        rows = []
        for it in items:
            ids = [c["chunk_id"] for c in ret.retrieve(it["query"], k=k_max, profile=prof)]
            for k in k_values:
                topk = ids[:k]
                if it["relevant"].intersection(topk):
                    hits[k] += 1
                if topk:
                    recalls[k] += len(it["relevant"].intersection(topk)) / len(it["relevant"])
                ndcgs[k] += _ndcg(ids, it["relevant"], k)
            if it.get("answers"):
                answer_n += 1
                for k in k_values:
                    if _contains_answer([text_by_id.get(cid, "") for cid in ids[:k]],
                                        it["answers"]):
                        answer_hits[k] += 1
            rank = next((r for r, cid in enumerate(ids, 1) if cid in it["relevant"]), None)
            if rank:
                rr += 1.0 / rank
            if per_query:
                rows.append({"query": it["query"], "retrieved": ids, "first_hit_rank": rank})
        n = len(items)
        row: Dict[str, Any] = {}
        for k in k_values:
            row[f"hit@{k}"] = round(hits[k] / n, 4)
            row[f"recall@{k}"] = round(recalls[k] / n, 4)
            row[f"ndcg@{k}"] = round(ndcgs[k] / n, 4)
            if has_answers and answer_n:
                row[f"answer_recall@{k}"] = round(answer_hits[k] / answer_n, 4)
        row["mrr"] = round(rr / n, 4)
        row["n_queries"] = n
        if has_answers and answer_n:
            row["n_with_answers"] = answer_n
        if per_query:
            row["per_query"] = rows
        results[prof] = row
        if not quiet:
            printable = {k: v for k, v in row.items() if k != "per_query"}
            print(f"  {prof:6s} " + "  ".join(f"{k}={v}" for k, v in printable.items()))
    return {"corpus": ret.describe(), "testset": str(testset_path),
            "k_values": list(k_values), "profiles": results}


def make_eval_template(index_dir: str, out_path: str, n: int = 20,
                       seed: int = 0) -> str:
    """生成待标注评测集骨架：随机抽 n 个 chunk，query 留给人工/LLM 填写。

    相关 chunk 先填好，使用者只需写查询；这是"零门槛自证"的最小闭环。
    """
    ret = Retriever(index_dir)
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(ret.coarse), size=min(n, len(ret.coarse)), replace=False)
    items = [{"query": "", "relevant_chunks": [ret.coarse[int(i)]["chunk_id"]],
              "source": ret.coarse[int(i)].get("source", "")}
             for i in sorted(idx)]
    out = Path(out_path)
    out.write_text(json.dumps({"items": items}, ensure_ascii=False, indent=1),
                   encoding="utf-8")
    return str(out)


def make_judge_sheet(index_dir: str, testset_path: str, out_path: str,
                     k: int = 5, profiles: Sequence[str] = (PROFILE_AUTO, PROFILE_BEST)) -> str:
    """生成"答案级"评测表：把检索上下文摊给 agent/人打分。

    chunk 级 hit 回答不了"最终能不能答对"，而接一个外部 LLM judge 又要额外 API。
    这个 skill 本身就运行在 agent 里，所以把打分交给 agent：这里只负责把
    查询 + 各档位检索到的上下文 + 评分位（留空）排好，agent 填完再用
    grade_judge_sheet 聚合。
    """
    ret = Retriever(index_dir)
    items = load_testset(testset_path)
    text_by_id = {c["chunk_id"]: c["text"] for c in ret.coarse}
    sheet = {
        "index_dir": str(index_dir),
        "testset": str(testset_path),
        "instruction": (
            "对每条 query，分别看 profiles 下检索到的上下文，给两个 0/1 分："
            "answer_correct=上下文是否足以答对（1 是），"
            "faithful=上下文与 query 是否相关且无噪声（1 是）。只依据上下文判断，不要用外部知识。"),
        "scale": {"answer_correct": "0 或 1", "faithful": "0 或 1"},
        "profiles": list(profiles),
        "items": [],
    }
    for it in items:
        entry = {"query": it["query"], "answers": it.get("answers", []),
                 "contexts": {}, "scores": {}}
        for prof in profiles:
            res = ret.retrieve(it["query"], k=k, profile=prof)
            entry["contexts"][prof] = [
                {"chunk_id": c["chunk_id"], "text": text_by_id.get(c["chunk_id"], c["text"])}
                for c in res]
            entry["scores"][prof] = {"answer_correct": None, "faithful": None}
        sheet["items"].append(entry)
    out = Path(out_path)
    out.write_text(json.dumps(sheet, ensure_ascii=False, indent=1), encoding="utf-8")
    return str(out)


def grade_judge_sheet(path: str) -> dict:
    """聚合已填分的评测表：各档位 answer_correct / faithful 均值 + 未填数量。"""
    sheet = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    profiles = sheet.get("profiles") or []
    acc: Dict[str, Dict[str, List[float]]] = {p: {"answer_correct": [], "faithful": []}
                                              for p in profiles}
    unfilled = 0
    for it in sheet.get("items", []):
        for prof, sc in (it.get("scores") or {}).items():
            for key in ("answer_correct", "faithful"):
                val = (sc or {}).get(key)
                if val is None:
                    unfilled += 1
                else:
                    acc.setdefault(prof, {"answer_correct": [], "faithful": []})[key].append(float(val))
    out: Dict[str, Any] = {"n_items": len(sheet.get("items", [])), "unfilled_scores": unfilled,
                            "profiles": {}}
    for prof, d in acc.items():
        row = {key: (round(sum(v) / len(v), 4) if v else None) for key, v in d.items()}
        row["n_scored"] = len(d.get("answer_correct") or [])
        out["profiles"][prof] = row
    return out


# 可调策略预设：默认全关；用 `tune` 在自己的评测集上选出再启用。
# 实测（4 个域 × 200 条，见 experiments/_opt/）说明收益高度依赖语料，不存在全局最优：
#   fast_rrf   wx +0.109 / t2 −0.015 / cmedqa −0.150
#   fusion0.7  t2 +0.020 / wx −0.127 / dureader −0.080
#   cand_rrf   t2 +0.005，其余持平
PRESETS: Dict[str, dict] = {
    "base": {},
    "fusion0.5": {"fusion": 0.5},
    "fusion0.7": {"fusion": 0.7},
    "fusion0.85": {"fusion": 0.85},
    "cand_rrf": {"candidates": "hybrid"},
    "fast_rrf": {"fast_rrf": True},
    "multiquery": {"multiquery": True},
    "mq_rrf": {"multiquery": True, "candidates": "hybrid"},
    "rerank_ens": {"rerank_ensemble": True},
}
DEFAULT_TUNE_PRESETS = ("base", "fusion0.7", "cand_rrf", "fast_rrf", "rerank_ens")


def _profile_metrics(ret: "Retriever", items: Sequence[dict], profile: str,
                     k_values: Sequence[int] = (1, 5)) -> dict:
    hits = {k: 0 for k in k_values}
    ndcgs = {k: 0.0 for k in k_values}
    k_max = max(k_values)
    for it in items:
        ids = [c["chunk_id"] for c in ret.retrieve(it["query"], k=k_max, profile=profile)]
        for k in k_values:
            if it["relevant"].intersection(ids[:k]):
                hits[k] += 1
            ndcgs[k] += _ndcg(ids, it["relevant"], k)
    n = len(items)
    row: Dict[str, float] = {}
    for k in k_values:
        row[f"hit@{k}"] = round(hits[k] / n, 4)
        row[f"ndcg@{k}"] = round(ndcgs[k] / n, 4)
    return row


def _per_query_hits(ret: "Retriever", items: Sequence[dict], profile: str,
                    k: int = 1) -> List[float]:
    """逐条命中向量（0/1），用于配对比较——同一批 query 上的差值才有意义。"""
    out = []
    for it in items:
        ids = [c["chunk_id"] for c in ret.retrieve(it["query"], k=k, profile=profile)]
        out.append(1.0 if it["relevant"].intersection(ids[:k]) else 0.0)
    return out


def _paired_bootstrap(a: Sequence[float], b: Sequence[float], seed: int = 0,
                      rounds: int = 2000) -> dict:
    """配对自助法：估计 mean(b) - mean(a) 的 95% 区间与 P(差值<=0)。

    单看两个小数谁大就下结论是这类小样本评测最常见的错误，所以 tune 必须给区间。
    """
    a = np.asarray(a, dtype="float64")
    b = np.asarray(b, dtype="float64")
    if len(a) != len(b) or not len(a):
        return {"delta": None}
    d = b - a
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(d), size=(rounds, len(d)))
    means = d[idx].mean(axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    return {
        "delta": round(float(d.mean()), 4),
        "ci95": [round(float(lo), 4), round(float(hi), 4)],
        "p_not_better": round(float((means <= 0).mean()), 4),
        "n": int(len(d)),
    }


def tune_index(index_dir: str, testset_path: str,
               profiles: Sequence[str] = PROFILES, model_dir: str = None,
               presets: Optional[Sequence[str]] = None,
               max_queries: Optional[int] = None,
               write_config: Optional[str] = None,
               holdout: float = 0.5,
               bootstrap_rounds: int = 2000) -> dict:
    """在你的评测集上比较策略预设，给出建议配置 + 差值置信区间。

    动机：四个域的实验表明任何全局默认都会在某个域上掉点（见 PRESETS 注释）。
    两个必须避免的坑，这里都处理了：
      1. 只看两个小数谁大 → 用**配对 bootstrap** 给差值区间与 P(差值<=0)；
      2. 在同一个集合上既选预设又报分数 → 默认按 query 切**前半选择 / 后半验证**，
         只有两边同向才推荐，否则建议保持 base。
    """
    items = load_testset(testset_path)
    if max_queries:
        items = items[:max_queries]
    base_ret = Retriever(index_dir, model_dir)
    known = {c["chunk_id"] for c in base_ret.coarse}
    unknown = sorted({cid for it in items for cid in it["relevant"] if cid not in known})
    if unknown:
        raise ValueError(f"评测集里有 {len(unknown)} 个 chunk_id 不在索引中，例如 "
                         + ", ".join(unknown[:5]))

    names = list(presets or DEFAULT_TUNE_PRESETS)
    bad = [n for n in names if n not in PRESETS]
    if bad:
        raise ValueError(f"未知预设 {bad}；可选 {list(PRESETS)}")
    if "base" not in names:
        names.insert(0, "base")

    n = len(items)
    split = int(n * (1.0 - holdout)) if (holdout and n >= 20) else n
    select_items, holdout_items = items[:split], items[split:]

    rets: Dict[str, "Retriever"] = {}
    for name in names:
        rets[name] = base_ret if not PRESETS[name] else Retriever(
            index_dir, model_dir, options=PRESETS[name])

    def _measure(subset: Sequence[dict]) -> Dict[str, Dict[str, List[float]]]:
        out = {}
        for name in names:
            out[name] = {p: _per_query_hits(rets[name], subset, p, 1) for p in profiles}
        return out

    sel = _measure(select_items)
    summary: Dict[str, Dict[str, dict]] = {}
    for name in names:
        summary[name] = {}
        for p in profiles:
            v = sel[name][p]
            summary[name][p] = {"hit@1": round(float(np.mean(v)) if v else 0.0, 4),
                                "n": len(v)}

    def _score(name: str) -> float:
        return sum(summary[name][p]["hit@1"] for p in profiles)

    ranked = sorted(names, key=_score, reverse=True)
    best = ranked[0]
    second = ranked[1] if len(ranked) > 1 else "base"

    payload: Dict[str, Any] = {
        "index_dir": str(index_dir),
        "testset": str(testset_path),
        "n_queries": n,
        "split": {"select": len(select_items), "holdout": len(holdout_items)},
        "profiles": list(profiles),
        "select_metrics": summary,
        "best_preset": best,
        "baseline_preset": "base",
        "bootstrap_rounds": bootstrap_rounds,
    }

    if best != "base":
        comparisons = {}
        for p in profiles:
            cmp = _paired_bootstrap(sel["base"][p], sel[best][p],
                                    rounds=bootstrap_rounds)
            comparisons[p] = cmp
        payload["vs_base"] = comparisons

        # 留出集复验：选择集上的赢家必须在留出集上也不掉点，否则视为噪声
        confirmed = None
        if holdout_items:
            ho = _measure(holdout_items)
            deltas = {p: round(float(np.mean(ho[best][p]) - np.mean(ho["base"][p])), 4)
                      for p in profiles}
            payload["holdout_delta"] = deltas
            confirmed = all(d >= 0 for d in deltas.values())
            payload["holdout_confirmed"] = confirmed
        else:
            payload["holdout_confirmed"] = None

        significant = all((cm.get("ci95") or [0, 0])[0] > 0
                          for cm in comparisons.values() if cm.get("delta") is not None)
        payload["significant"] = significant
        if not significant or confirmed is False:
            payload["recommendation"] = "base"
            payload["why"] = ("差值 95% 区间跨 0" if not significant
                              else "留出集上未复现") + "：保持 base 更稳"
        else:
            payload["recommendation"] = best
            payload["why"] = "选择集上区间不跨 0，且留出集同向"
    else:
        payload["recommendation"] = "base"
        payload["why"] = "没有预设在选择集上优于 base"
        payload["runner_up"] = second

    payload["note"] = ("样本 <100 条时 0.01 级差异多在噪声内；建议 ≥200 条。"
                       "holdout 默认为按 query 顺序后半（请先把评测集打乱或保证无顺序偏置）。")

    if write_config:
        Path(write_config).write_text(
            json.dumps({"preset": payload["recommendation"]}, ensure_ascii=False, indent=1),
            encoding="utf-8")
        payload["config_written"] = str(write_config)
    return payload


def diagnose(index_dir: str, testset_path: str, pool: int = 10,
             model_dir: str = None) -> dict:
    """诊断瓶颈：召回够不够、重排排得准不准、还有多少可救回空间。

    这是"先诊断再优化"的核心工具（对应 benchmark/ceiling_diag.py 的库化版本）：
    - `first_pass_hit`：第一跳 top5 命中率 → 召回能力
    - `pool_recall`：候选池 top-N 里是否含正例 → 重排的输入质量
    - `hit_at_1`：重排后第一名命中率 → 排序能力
    - `recoverable`：pool_recall − hit_at_1，即"重排最多还能救回多少"

    判读规则：
    - pool_recall ≈ 1.0 且 recoverable 大 → **排序瓶颈**，加召回手段无效，去微调重排器；
    - pool_recall 明显低于 first_pass → 索引/切分异常，先查建库；
    - first_pass 低 → 召回不足，扩池/混合召回/查询改写才有意义。
    """
    ret = Retriever(index_dir, model_dir)
    items = load_testset(testset_path)
    known = {c["chunk_id"] for c in ret.coarse}
    items = [it for it in items if it["relevant"] & known]
    if not items:
        raise ValueError("评测集与本索引没有交集：relevant 里的 chunk_id 都不在索引中")

    n = len(items)
    first = pool_hit = rerank1 = 0
    for it in items:
        q = it["query"]
        if it["relevant"] & {c["chunk_id"] for c, _ in ret.first_pass(q, 5)}:
            first += 1
        cands = ret.slow_candidates(q, pool)
        ids = {c["chunk_id"] for c, _ in cands}
        if it["relevant"] & ids:
            pool_hit += 1
        rr = ret.rerank(q, cands, 1)
        if rr and rr[0][0]["chunk_id"] in it["relevant"]:
            rerank1 += 1

    first_r, pool_r, hit1 = first / n, pool_hit / n, rerank1 / n
    recoverable = max(0.0, pool_r - hit1)
    if pool_r >= 0.95 and recoverable > 0.05:
        bottleneck, advice = "ranking", (
            f"排序瓶颈：候选池已经几乎必有正例（{pool_r:.3f}），但只有 {hit1:.3f} 排到第一。"
            "再加召回手段（multi-query/扩池/混合召回）收益上限很低，"
            "应当微调或更换重排器（见 docs 优化报告 §11）。")
    elif first_r < 0.85:
        bottleneck, advice = "recall", (
            f"召回瓶颈：第一跳只有 {first_r:.3f} 命中。此时扩候选池、混合召回、"
            "查询改写才有意义；先确认切块与语料分类是否合理。")
    else:
        bottleneck, advice = "balanced", (
            f"召回与排序都较好（第一跳 {first_r:.3f} / 候选池 {pool_r:.3f} / "
            f"hit@1 {hit1:.3f}），可救回空间仅 {recoverable:.3f}，继续优化收益有限。")

    return {
        "corpus": ret.describe(),
        "n_queries": n,
        "pool_size": pool,
        "first_pass_hit@5": round(first_r, 4),
        "pool_recall@%d" % pool: round(pool_r, 4),
        "hit@1": round(hit1, 4),
        "recoverable": round(recoverable, 4),
        "bottleneck": bottleneck,
        "advice": advice,
    }


# ---------------------------------------------------------------- CLI
def _print_or_json(payload, as_json: bool, pretty: bool = False) -> None:
    if as_json:
        print(json.dumps(payload, ensure_ascii=False, indent=2 if pretty else None))
    else:
        if isinstance(payload, list):
            for row in payload:
                print(row)
        elif isinstance(payload, dict):
            for k, v in payload.items():
                print(f"{k}: {v}")
        else:
            print(payload)


def main(argv: Optional[Sequence[str]] = None) -> int:
    import argparse

    ensure_utf8_stdout()

    # --json 既接受写在子命令前，也接受写在子命令后（agent 常直接跟在参数尾部）
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--json", action="store_true", default=argparse.SUPPRESS,
                        help="输出单行 JSON（agent/脚本用）")
    common.add_argument("--json-pretty", action="store_true", default=argparse.SUPPRESS,
                        help="输出缩进 JSON")
    p = argparse.ArgumentParser(
        prog="retrieval_core.py", parents=[common],
        description="adaptive-retrieval：本地文档零配置自适应检索（CLI 全命令支持 --json）")
    sub = p.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("index", parents=[common], help="对目录内文档建索引（默认递归、增量）")
    b.add_argument("docs_dir")
    b.add_argument("--index-dir", default=".retrieval_index")
    b.add_argument("--model-dir", default=None)
    b.add_argument("--ext", default=",".join(DEFAULT_EXTS),
                   help=f"逗号分隔扩展名，默认 {','.join(DEFAULT_EXTS)}")
    b.add_argument("--no-recursive", action="store_true", help="只扫顶层目录")
    b.add_argument("--force", action="store_true", help="忽略增量缓存，全量重建")
    b.add_argument("--batch-size", type=int, default=32)

    r = sub.add_parser("retrieve", parents=[common], help="检索")
    r.add_argument("query", nargs="?", default=None)
    r.add_argument("--batch", default=None, help="查询文件，一行一条")
    r.add_argument("--index-dir", default=os.environ.get("RETRIEVAL_INDEX_DIR", ".retrieval_index"))
    r.add_argument("--profile", choices=list(PROFILES) + [PROFILE_DENSE], default="auto")
    r.add_argument("--k", type=int, default=5)
    r.add_argument("--min-score", type=float, default=None, help="低于该分数不返回")
    r.add_argument("--no-rerank", action="store_true")
    r.add_argument("--with-text", action="store_true", help="文末附原文（人读模式）")
    r.add_argument("--preset", choices=list(PRESETS), default=None,
                   help="启用 tune 选出的策略预设（默认 base）")
    r.add_argument("--config", default=None, help="tune --out 写出的配置文件")
    r.add_argument("--model-dir", default=None)

    e = sub.add_parser("eval", parents=[common], help="离线评测：各档位 hit/recall/ndcg/MRR")
    e.add_argument("--index-dir", default=os.environ.get("RETRIEVAL_INDEX_DIR", ".retrieval_index"))
    e.add_argument("--testset", default=None,
                   help="JSON 评测集；或用 --template 生成骨架")
    e.add_argument("--template", default=None, help="生成待标注评测集骨架到该路径")
    e.add_argument("--n", type=int, default=20, help="--template 抽样条数")
    e.add_argument("--profiles", default="fast,auto,best", help="逗号分隔，默认三档")
    e.add_argument("--k", default="1,5", help="逗号分隔的 k 值，默认 1,5")
    e.add_argument("--no-baseline", action="store_true", help="不跑 dense 对照臂")
    e.add_argument("--out", default=None, help="把评测结果写入 JSON 文件")
    e.add_argument("--per-query", action="store_true", help="结果内含逐条查询明细")
    e.add_argument("--judge-sheet", default=None,
                   help="生成答案级评测表（含各档位上下文，留空待 agent/人打分）")
    e.add_argument("--judge-results", default=None, help="聚合已填分的答案级评测表")
    e.add_argument("--model-dir", default=None)

    s = sub.add_parser("stats", parents=[common], help="索引元信息（不加载模型）")
    s.add_argument("--index-dir", default=os.environ.get("RETRIEVAL_INDEX_DIR", ".retrieval_index"))

    t = sub.add_parser("tune", parents=[common],
                       help="在自己的评测集上比较策略预设（base/fusion/cand_rrf/fast_rrf）")
    t.add_argument("--index-dir", default=os.environ.get("RETRIEVAL_INDEX_DIR", ".retrieval_index"))
    t.add_argument("--testset", required=True)
    t.add_argument("--profiles", default="fast,auto,best")
    t.add_argument("--presets", default=",".join(DEFAULT_TUNE_PRESETS),
                   help=f"逗号分隔，可选 {list(PRESETS)}")
    t.add_argument("--max-queries", type=int, default=None, help="只跑前 N 条（快速试跑）")
    t.add_argument("--out", default=None, help="把最优预设写入配置文件，供 retrieve --config 使用")
    t.add_argument("--model-dir", default=None)

    args = p.parse_args(argv)
    as_json = bool(getattr(args, "json", False) or getattr(args, "json_pretty", False))
    pretty = bool(getattr(args, "json_pretty", False))
    try:
        if args.cmd == "index":
            meta = build_index(args.docs_dir, args.index_dir, args.model_dir,
                               recursive=not args.no_recursive,
                               exts=[x.strip() for x in args.ext.split(",") if x.strip()],
                               force=args.force, quiet=as_json,
                               batch_size=args.batch_size)
            _print_or_json(meta, as_json, pretty)

        elif args.cmd == "retrieve":
            queries: List[str] = []
            if args.query:
                queries.append(args.query)
            if args.batch:
                queries.extend([ln.strip() for ln in
                                Path(args.batch).read_text(encoding="utf-8-sig").splitlines()
                                if ln.strip()])
            if not queries:
                raise SystemExit("需要 query 或 --batch 提供查询")
            options: Optional[dict] = None
            if args.config:
                cfg = json.loads(Path(args.config).read_text(encoding="utf-8-sig"))
                options = PRESETS.get((cfg or {}).get("preset", "base"), {})
            if args.preset:
                options = PRESETS[args.preset]
            ret = Retriever(args.index_dir, args.model_dir,
                            disable_rerank=args.no_rerank, options=options)
            payload: List[dict] = []
            for q in queries:
                res, trace = ret.retrieve_with_trace(q, k=args.k, profile=args.profile)
                if args.min_score is not None:
                    res = [c for c in res if c["score"] >= args.min_score]
                payload.append({"query": q, "corpus": ret.describe(),
                                "route": trace, "results": res})
            if as_json:
                _print_or_json(payload[0] if len(payload) == 1 else payload,
                               True, pretty)
            else:
                for item in payload:
                    print(f"[{ret.corpus_type} 语料: {len(ret.coarse)} 粗块, 平均 "
                          f"{ret.stats['avg_chunk_chars']} 字符 | profile={args.profile} | "
                          f"route={item['route']['route']} | "
                          f"{item['route']['timings']['total_ms']}ms]")
                    for c in item["results"]:
                        print(f"{c['rank']}. [{c['chunk_id']}] score={c['score']} "
                              f"source={c['source']}")
                        if args.with_text:
                            print("   " + c["text"][:300].replace("\n", " "))

        elif args.cmd == "eval":
            if args.judge_results:
                _print_or_json(grade_judge_sheet(args.judge_results), as_json, pretty)
                return 0
            if args.judge_sheet:
                if not args.testset:
                    raise SystemExit("--judge-sheet 需要 --testset")
                path = make_judge_sheet(args.index_dir, args.testset, args.judge_sheet,
                                        k=max(int(x) for x in args.k.split(",") if x.strip()))
                _print_or_json({"judge_sheet": path,
                                "next": "让 agent/人给每条 profiles 打 answer_correct/faithful "
                                        "0/1，然后 eval --judge-results <该文件>"},
                               as_json, pretty)
                return 0
            if args.template:
                path = make_eval_template(args.index_dir, args.template, n=args.n)
                _print_or_json({"template": path,
                                "next": "填写每条的 query 后运行 eval --testset <该文件>"},
                               as_json, pretty)
                return 0
            if not args.testset:
                raise SystemExit("需要 --testset 或 --template")
            result = eval_index(args.index_dir, args.testset,
                                profiles=[x.strip() for x in args.profiles.split(",") if x.strip()],
                                k_values=[int(x) for x in args.k.split(",") if x.strip()],
                                model_dir=args.model_dir,
                                with_baseline=not args.no_baseline,
                                quiet=as_json, per_query=args.per_query)
            if args.out:
                Path(args.out).write_text(json.dumps(result, ensure_ascii=False, indent=2),
                                          encoding="utf-8")
            _print_or_json(result, as_json, pretty)

        elif args.cmd == "stats":
            ret = Retriever(args.index_dir)
            _print_or_json(ret.describe(), as_json, pretty)

        elif args.cmd == "tune":
            result = tune_index(
                args.index_dir, args.testset,
                profiles=[x.strip() for x in args.profiles.split(",") if x.strip()],
                model_dir=args.model_dir,
                presets=[x.strip() for x in args.presets.split(",") if x.strip()],
                max_queries=args.max_queries,
                write_config=args.out)
            _print_or_json(result, as_json, pretty)
            if not as_json:
                print(f"\n建议预设: {result['best_preset']}"
                      + (f"（已写入 {result['config_written']}）" if result.get("config_written") else ""))
        return 0
    except BrokenPipeError:
        return 0
    except Exception as exc:
        if as_json:
            print(json.dumps({"error": type(exc).__name__, "message": str(exc)},
                             ensure_ascii=False))
        else:
            print(f"[错误] {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
