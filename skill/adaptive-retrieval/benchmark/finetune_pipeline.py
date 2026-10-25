# -*- coding: utf-8 -*-
"""领域微调重排器的一站式入口（把四个脚本串起来）。

    1. build   从带标注的语料构造训练集（隔离测试集）
    2. train   微调 bge-reranker-base
    3. eval    在留出测试集上对比微调前后
    4. report  出显著性结论

用法：
  python benchmark/finetune_pipeline.py --step all --domain m5_dureader
  python benchmark/finetune_pipeline.py --step train --batch 8 --max-length 256

显存提示：8GB 显卡务必 --batch 8 --max-length 256（batch 16 会触发显存换页，
实测每步从 0.16s 恶化到 10.8s；原因见 bench_train_speed.py）。
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _paths import REPO_ROOT as ROOT, SKILL_DIR  # noqa: E402
PY = sys.executable


def run(cmd: list, label: str) -> int:
    print(f"\n=== {label} ===")
    print(" ".join(str(c) for c in cmd))
    proc = subprocess.run([str(c) for c in cmd], cwd=str(ROOT))
    if proc.returncode != 0:
        print(f"[失败] {label} 退出码 {proc.returncode}")
    return proc.returncode


def main() -> None:
    for _s in (sys.stdout, sys.stderr):
        try:
            if hasattr(_s, "reconfigure"):
                _s.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass
    ap = argparse.ArgumentParser()
    ap.add_argument("--step", default="all",
                    choices=["all", "build", "train", "eval", "report"])
    ap.add_argument("--domain", default="m5_dureader",
                    help="评测域（build 固定用 dureader 的 dev 语料）")
    ap.add_argument("--n-queries", type=int, default=900)
    ap.add_argument("--neg", type=int, default=4)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--max-length", type=int, default=256)
    ap.add_argument("--lr", type=float, default=2e-5)
    a = ap.parse_args()

    steps = ["build", "train", "eval", "report"] if a.step == "all" else [a.step]
    rc = 0
    for step in steps:
        if step == "build":
            rc |= run([PY, "benchmark/build_rerank_train.py",
                       "--n-queries", a.n_queries, "--neg", a.neg], "构造训练集（含测试集隔离校验）")
        elif step == "train":
            rc |= run([PY, "benchmark/finetune_reranker.py",
                       "--epochs", a.epochs, "--batch", a.batch,
                       "--max-length", a.max_length, "--lr", a.lr], "微调重排器")
        elif step == "eval":
            rc |= run([PY, "benchmark/eval_finetuned_rerank.py",
                       "--domains", a.domain], "留出集评测")
        elif step == "report":
            rc |= run([PY, "benchmark/significance_check.py"], "显著性检验 + 泄漏排查")

    model_dir = ROOT / "experiments" / "_rerank_train" / "model"
    print(f"\n微调模型: {model_dir}")
    if (model_dir / "config.json").exists():
        print("挂上它（不改代码）：")
        print(f'  AR_RERANKER_DIR="{model_dir}" python skill/adaptive-retrieval/scripts/'
              f'retrieval_core.py retrieve "查询" --index-dir ./idx --profile best')
        print("或写进 MCP 配置的 env，让常驻服务一直用它。")
    sys.exit(0 if rc == 0 else 1)


if __name__ == "__main__":
    main()
