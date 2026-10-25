# -*- coding: utf-8 -*-
"""标定 cross-encoder 训练/推理速度，找出能跑完的配置。

之前 batch=16 实测 3.5s/step、batch=32 实测 131s/step，与 forward 4.8ms/pair 严重不符，
说明有隐藏开销。这里逐项测量：模型加载、单步前向、单步反传、优化器更新、显存峰值。
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _paths import REPO_ROOT as ROOT, SKILL_DIR  # noqa: E402


def main() -> None:
    for _s in (sys.stdout, sys.stderr):
        try:
            if hasattr(_s, "reconfigure"):
                _s.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass
    import torch
    from sentence_transformers import CrossEncoder

    print(f"torch {torch.__version__}  cuda={torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"gpu: {torch.cuda.get_device_name(0)}  "
              f"total {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB")

    t0 = time.time()
    model = CrossEncoder(str(ROOT / "models" / "bge-reranker-base"),
                         max_length=512, device="cuda", num_labels=1)
    print(f"模型加载 {time.time() - t0:.1f}s")
    tok = model.tokenizer
    print(f"参数 {sum(p.numel() for p in model.model.parameters()) / 1e6:.0f}M")

    for batch in (8, 16, 32):
        try:
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            texts = [["查什么", "文" * 400] for _ in range(batch)]
            enc = tok([x[0] for x in texts], [x[1] for x in texts], padding="max_length",
                      truncation=True, max_length=512, return_tensors="pt")
            enc = {k: v.cuda() for k, v in enc.items()}
            labels = torch.zeros(batch, device="cuda")

            # 纯前向
            model.model.eval()
            with torch.no_grad():
                for _ in range(2):
                    model.model(**enc)
                torch.cuda.synchronize()
                t = time.time()
                for _ in range(5):
                    model.model(**enc)
                torch.cuda.synchronize()
                fwd = (time.time() - t) / 5 * 1000

            # 前向+反传+更新
            model.model.train()
            opt = torch.optim.AdamW(model.model.parameters(), lr=1e-6)
            loss_fn = torch.nn.BCEWithLogitsLoss()
            def one_step():
                out = model.model(**enc)
                loss = loss_fn(out.logits.view(-1), labels)
                loss.backward()
                opt.step()
                opt.zero_grad(set_to_none=True)
            one_step()
            torch.cuda.synchronize()
            t = time.time()
            for _ in range(3):
                one_step()
            torch.cuda.synchronize()
            step = (time.time() - t) / 3 * 1000
            peak = torch.cuda.max_memory_allocated() / 1e9
            print(f"batch={batch:2d}: forward {fwd:7.0f}ms   train-step {step:8.0f}ms   "
                  f"peak {peak:.2f}GB   {step / batch:.0f}ms/pair")
        except RuntimeError as exc:
            print(f"batch={batch:2d}: OOM/错误 -> {str(exc)[:120]}")
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
