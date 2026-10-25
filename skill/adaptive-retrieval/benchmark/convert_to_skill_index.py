# -*- coding: utf-8 -*-
"""把 benchmark 格式索引（chunks.pkl）转成 Skill 格式索引。

适用于 wx server 索引与 M5 DuReader 索引；chunk 文本与 chunk_id 与 benchmark
语料完全一致，对应 testset 标注可直接复用。按 chunk 长度自动分类语料（M6）：
短段落语料不生成 fine.npz，写 meta.json 记录分类结果。

用法: python benchmark/convert_to_skill_index.py --src <chunks.pkl 目录> --out <输出目录>
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _paths import REPO_ROOT as ROOT, SKILL_DIR  # noqa: E402
# chunks.pkl 反序列化：wx server 的 src.chunking.TextChunk 模块路径对不上，
# 由 corpus._ChunkUnpickler 兜底，需 benchmark 包可导入
sys.path.insert(0, str(ROOT))
from benchmark.corpus import load_chunks  # noqa: E402

sys.path.insert(0, str(SKILL_DIR / "scripts"))
from retrieval_core import (  # noqa: E402
    COARSE_OVERLAP, COARSE_SIZE, FINE_SIZE, SCHEMA_VERSION,
    _embedder_id, _find_local_model, _load_embedder, classify_corpus,
    corpus_stats, ensure_utf8_stdout, fine_chunks,
)


def main():
    ensure_utf8_stdout()
    p = argparse.ArgumentParser(description="benchmark 索引 -> Skill 格式索引")
    p.add_argument("--src", required=True, help="chunks.pkl 所在目录")
    p.add_argument("--out", required=True, help="Skill 索引输出目录")
    a = p.parse_args()

    chunks = load_chunks(Path(a.src))
    coarse = [{
        "text": c.text,
        "source": (c.metadata or {}).get("source", ""),
        "chunk_index": (c.metadata or {}).get("chunk_index", 0),
        "chunk_id": c.chunk_id,
    } for c in chunks]

    stats = corpus_stats(coarse)
    corpus_type = classify_corpus(stats)

    model = _load_embedder()
    vecs = model.encode([c["text"] for c in coarse], normalize_embeddings=True,
                        show_progress_bar=False)

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "coarse.json").write_text(
        json.dumps(coarse, ensure_ascii=False, indent=1), encoding="utf-8")
    np.save(out / "coarse_vecs.npy", np.asarray(vecs, dtype="float32"))
    n_fine = 0
    if corpus_type == "long":
        fine_texts, fine_parent = fine_chunks(coarse)
        n_fine = len(fine_texts)
        fine_vecs = model.encode(fine_texts, normalize_embeddings=True,
                                 show_progress_bar=False)
        np.savez_compressed(out / "fine.npz",
                            fine_vecs=np.asarray(fine_vecs, dtype="float32"),
                            fine_parent=np.array(fine_parent))

    # 与 build_index 同 schema 的 meta：让 Retriever 能做维度/embedder 校验并报告真实来源。
    # docs 留空：外部 chunk 没有对应源文件，增量复用不适用（每次转换都全量编码）。
    meta = {
        "schema_version": SCHEMA_VERSION,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "corpus_type": corpus_type,
        "n_fine_chunks": n_fine,
        "embedder_id": _embedder_id(_find_local_model("bge-small-zh-v1.5")
                                    or "BAAI/bge-small-zh-v1.5"),
        "dim": int(np.asarray(vecs).shape[1]),
        "chunk": {"coarse_size": COARSE_SIZE, "coarse_overlap": COARSE_OVERLAP,
                  "fine_size": FINE_SIZE, "schema": SCHEMA_VERSION},
        "docs": {},
        "converted_from": str(a.src),
        **stats,
    }
    (out / "meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"Skill 索引完成: {meta['n_chunks']} 粗块 [{corpus_type} 语料] -> {out}")


if __name__ == "__main__":
    main()
