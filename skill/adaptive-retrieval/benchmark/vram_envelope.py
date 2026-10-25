# -*- coding: utf-8 -*-
"""在 8GB 显卡上标定可用配置：哪些 batch/max_length 组合能跑、哪些会爆。

结论用于填写 SKILL.md 的"显存边界"一表，避免用户反复试错。
每个组合只跑 2 步训练，失败即记录，不占用太久。
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
    from transformers import AutoTokenizer

    base = str(ROOT / "models" / "bge-reranker-base")
    props = torch.cuda.get_device_properties(0)
    print(f"GPU: {props.name}  total {props.total_memory / 1e9:.2f}GB")
    print(f"{'batch':>6s} {'maxlen':>7s} {'峰值GB':>8s} {'步耗时ms':>10s} {'结论':>6s}")

    tok = AutoTokenizer.from_pretrained(base)
    model = None
    # 每个组合独立测：不清空模型，但清空显存统计
    combos = [(8, 128), (8, 256), (8, 384), (8, 512),
              (16, 128), (16, 256), (16, 512),
              (4, 512), (2, 768)]
    for batch, maxlen in combos:
        try:
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            if model is None:
                model = CrossEncoder(base, max_length=maxlen, device="cuda", num_labels=1)
            model.model.train()
            model.model.max_length = maxlen
            opt = torch.optim.AdamW(model.model.parameters(), lr=1e-6)
            loss_fn = torch.nn.BCEWithLogitsLoss()

            enc = tok(["查什么"] * batch, ["文" * (maxlen * 2)] * batch,
                      padding="max_length", truncation=True, max_length=maxlen,
                      return_tensors="pt")
            enc = {k: v.cuda() for k, v in enc.items()}
            labels = torch.zeros(batch, device="cuda")

            # 先做一次，确认能过
            out = model.model(**enc)
            loss = loss_fn(out.logits.view(-1), labels)
            loss.backward()
            opt.step()
            opt.zero_grad(set_to_none=True)
            torch.cuda.synchronize()

            t = time.time()
            for _ in range(2):
                out = model.model(**enc)
                loss = loss_fn(out.logits.view(-1), labels)
                loss.backward()
                opt.step()
                opt.zero_grad(set_to_none=True)
            torch.cuda.synchronize()
            step = (time.time() - t) / 2 * 1000
            peak = torch.cuda.max_memory_allocated() / 1e9
            ok = "OK" if peak < props.total_memory / 1e9 * 0.92 else "临界"
            print(f"{batch:6d} {maxlen:7d} {peak:8.2f} {step:10.0f} {ok:>6s}")
        except RuntimeError as exc:
            msg = str(exc).split("\n")[0][:60]
            print(f"{batch:6d} {maxlen:7d} {'--':>8s} {'--':>10s} {'OOM':>6s}  {msg}")
            model = None
            torch.cuda.empty_cache()
        except Exception as exc:
            print(f"{batch:6d} {maxlen:7d} {'--':>8s} {'--':>10s} {'ERR':>6s}  {str(exc)[:60]}")
            model = None
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
