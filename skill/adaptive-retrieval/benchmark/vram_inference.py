# -*- coding: utf-8 -*-
"""8GB 显卡的实际边界：日常推理占用 vs 微调占用，以及更大模型是否可行。

用途：直接回答"8GB 能做什么/不能做什么"，不靠猜。
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _paths import REPO_ROOT as ROOT, SKILL_DIR  # noqa: E402


def measure_inference(model_path: str, label: str, max_len: int = 512) -> None:
    import torch
    from sentence_transformers import CrossEncoder
    try:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        m = CrossEncoder(model_path, max_length=max_len, device="cuda", num_labels=1)
        pairs = [["怎么申请奖学金", "文" * 400] for _ in range(10)]
        m.predict(pairs, show_progress_bar=False)   # 预热
        peak = torch.cuda.max_memory_allocated() / 1e9
        n = sum(p.numel() for p in m.model.parameters()) / 1e6
        print(f"  {label:38s} 参数 {n:6.0f}M  推理峰值 {peak:5.2f}GB  OK")
        del m
        torch.cuda.empty_cache()
    except RuntimeError as exc:
        print(f"  {label:38s} 失败: {str(exc).splitlines()[0][:60]}")
        torch.cuda.empty_cache()
    except Exception as exc:
        print(f"  {label:38s} 错误: {str(exc)[:60]}")


def main() -> None:
    for _s in (sys.stdout, sys.stderr):
        try:
            if hasattr(_s, "reconfigure"):
                _s.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass
    import torch
    props = torch.cuda.get_device_properties(0)
    print(f"GPU {props.name}  显存 {props.total_memory / 1e9:.2f}GB\n")

    print("【日常推理】检索时实际占用（skill 默认路径）：")
    measure_inference(str(ROOT / "models" / "bge-reranker-base"), "bge-reranker-base (默认)")
    measure_inference(str(ROOT / "experiments" / "_rerank_train" / "model"),
                      "微调后的 bge-reranker-base")

    print("\n【模型规模参照】不同尺寸重排器的推理占用估算（按参数线性估算）：")
    base_gb = 0.55   # 278M 权重 fp16 ≈ 0.55GB
    for name, params in (("bge-reranker-base", 278), ("bge-reranker-large", 560),
                         ("bge-reranker-v2-m3", 568), ("bge-reranker-v2-gemma", 9000)):
        est = params / 278 * base_gb
        verdict = "可行" if est < 6 else ("勉强" if est < 7.5 else "不可行")
        print(f"  {name:26s} ~{params:5d}M  推理约 {est:5.2f}GB  -> {verdict}")


if __name__ == "__main__":
    main()
