# -*- coding: utf-8 -*-
"""语料加载：读取 wx 程序 server 建好的 FAISS 索引与 chunk 数据。

chunk.pkl 里存的是 src.chunking.TextChunk 对象，pickle 反序列化时
模块路径可能对不上（不同项目目录），所以用自定义 Unpickler 兜底。
"""
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List

import numpy as np
import faiss

# 默认路径：wx 程序 server 的索引与主项目的测试集
WX_SERVER_DIR = Path(r"D:\claude code use\rag-research-project\wx 程序\server")
MAIN_PROJECT_DIR = Path(r"D:\claude code use\rag-research-project")
INDEX_DIR = WX_SERVER_DIR / "data" / "processed" / "index"
TESTSET_PATH = MAIN_PROJECT_DIR / "experiments" / "splits" / "test.json"


@dataclass
class Chunk:
    """与 src.chunking.TextChunk 结构一致的本地副本"""
    text: str
    metadata: Dict = field(default_factory=dict)

    @property
    def chunk_id(self) -> str:
        return str(self.metadata.get("chunk_id", ""))


class _ChunkUnpickler(pickle.Unpickler):
    """把任何 TextChunk 类引用都映射到本地 Chunk，绕开模块路径差异"""

    def find_class(self, module, name):
        if name == "TextChunk":
            return Chunk
        return super().find_class(module, name)


def load_chunks(index_dir: Path = INDEX_DIR) -> List[Chunk]:
    index_dir = Path(index_dir)
    with (index_dir / "chunks.pkl").open("rb") as f:
        chunks = _ChunkUnpickler(f).load()
    print(f"语料: {len(chunks)} 个 chunk，来自 {index_dir}")
    return chunks


def load_faiss_index(index_dir: Path = INDEX_DIR, dim: int = 512) -> faiss.Index:
    index_dir = Path(index_dir)
    arr = np.load(index_dir / "index.npy")
    index = faiss.deserialize_index(arr)
    print(f"FAISS 索引: {index.ntotal} 条向量，维度 {index.d}")
    return index


def load_testset(path: Path = TESTSET_PATH) -> List[Dict]:
    """兼容两种格式：list（splits/test.json）或 {"items": [...]}（testset_v2）"""
    import json
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    testset = data["items"] if isinstance(data, dict) else data
    print(f"测试集: {len(testset)} 条查询，来自 {path}")
    return testset
