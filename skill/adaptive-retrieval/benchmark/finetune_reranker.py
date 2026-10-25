# -*- coding: utf-8 -*-
"""微调 bge-reranker-base（cross-encoder），验证"排序瓶颈"诊断。

background：ceiling_diag 显示 DuReader 候选池 recall@10 = 1.000，但 hit@1 只有 0.615，
说明漏检 100% 来自重排排序。那么把重排器在**同分布训练集**上微调，hit@1 应当上升；
若提不动，则 0.615 是标注/语料本身的上限。

数据隔离：训练集来自 2000 条 dev 中**排除测试集 200 条 query 与 200 个 docid**后的部分
（见 build_rerank_train.py，脚本内有 assert 零交集）。

用法：
  python benchmark/finetune_reranker.py --epochs 2 --batch 16
  python benchmark/finetune_reranker.py --dry-run          # 只打印数据与超参
输出：experiments/_rerank_train/{model/, train_log.json}
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _paths import REPO_ROOT as ROOT, SKILL_DIR  # noqa: E402
TRAIN_DIR = ROOT / "experiments" / "_rerank_train"
MODELS = ROOT / "models"


def load_pairs(split: str) -> list:
    path = TRAIN_DIR / f"{split}.jsonl"
    out = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def main() -> None:
    for _s in (sys.stdout, sys.stderr):
        try:
            if hasattr(_s, "reconfigure"):
                _s.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

    ap = argparse.ArgumentParser()
    ap.add_argument("--base-model", default=str(MODELS / "bge-reranker-base"))
    ap.add_argument("--out-dir", default=str(TRAIN_DIR / "model"))
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--batch", type=int, default=8,
                    help="8GB 显存请用 8：实测 batch16 峰值 8.75GB 会触发显存换页抖动（10801ms/step）")
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--max-length", type=int, default=256,
                    help="256 足够：query 很短，passage 关键信息在前 256 token；512 显存翻倍")
    ap.add_argument("--warmup-ratio", type=float, default=0.1)
    ap.add_argument("--freeze-layers", type=int, default=0,
                    help="冻结底层 encoder 层数；v2-m3 在 8GB 上必须设 22（只训最后 1 层）")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    train = load_pairs("train")
    dev = load_pairs("dev")
    print(f"训练对 {len(train)}（正例 {sum(p['label'] for p in train)}）")
    print(f"验证对 {len(dev)}（正例 {sum(p['label'] for p in dev)}）")
    if not train or not dev:
        raise SystemExit("训练/验证数据为空，先跑 benchmark/build_rerank_train.py")

    import torch
    from sentence_transformers import CrossEncoder
    from torch.utils.data import DataLoader
    from transformers import AutoTokenizer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device}  lr={a.lr}  epochs={a.epochs}  batch={a.batch}")
    if a.dry_run:
        # 走一遍前向，确认显存/形状没问题，不做优化
        model = CrossEncoder(a.base_model, max_length=a.max_length, device=device,
                             num_labels=1)
        batch = train[:4]
        scores = model.predict([[p["query"], p["passage"]] for p in batch],
                               show_progress_bar=False)
        print(f"dry-run 前向 OK，示例分数 {[round(float(s), 4) for s in scores]}")
        return

    model = CrossEncoder(a.base_model, max_length=a.max_length, device=device, num_labels=1)
    tokenizer = AutoTokenizer.from_pretrained(a.base_model)

    # 关键：每步在 collate 里分词会让 GPU 空转等 CPU（实测 3.5s/step）。
    # 这里先一次性把所有 pair 分词成定长张量，训练循环里只做切片，不再碰 tokenizer。
    def pretokenize(rows):
        t0 = time.time()
        enc = tokenizer([r["query"] for r in rows], [r["passage"] for r in rows],
                        padding="max_length", truncation="longest_first",
                        max_length=a.max_length, return_tensors="pt")
        labels = torch.tensor([float(r["label"]) for r in rows], dtype=torch.float32)
        print(f"  预分词 {len(rows)} 对，用时 {time.time() - t0:.1f}s，"
              f"张量 {tuple(enc['input_ids'].shape)}", flush=True)
        return enc, labels

    train_enc, train_y = pretokenize(train)
    dev_enc, dev_y = pretokenize(dev)

    class TokenDataset(torch.utils.data.Dataset):
        def __init__(self, enc, y):
            self.enc, self.y = enc, y

        def __len__(self):
            return len(self.y)

        def __getitem__(self, i):
            return {k: v[i] for k, v in self.enc.items()}, self.y[i]

    def collate(batch):
        items = [b[0] for b in batch]
        enc = {k: torch.stack([it[k] for it in items]) for k in items[0]}
        return enc, torch.stack([b[1] for b in batch])

    train_loader = DataLoader(TokenDataset(train_enc, train_y), batch_size=a.batch,
                              shuffle=True, collate_fn=collate, num_workers=0)
    dev_loader = DataLoader(TokenDataset(dev_enc, dev_y), batch_size=a.batch,
                            shuffle=False, collate_fn=collate)

    # 混合精度：cross-encoder 前向在 fp16/bf16 下足够稳，速度约 2-3 倍
    use_amp = device == "cuda"
    amp_dtype = torch.bfloat16 if (use_amp and torch.cuda.is_bf16_supported()) else torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=(use_amp and amp_dtype == torch.float16))

    # 可选：冻结底层 encoder，只训最后 N 层 + pooler/classifier。
    # 对 8GB 显卡跑 v2-m3 是必需的：全参数微调峰值 11.37GB（超物理显存），
    # 冻结后只要 2.71GB（实测）。同时小数据下也更不容易过拟合。
    if a.freeze_layers > 0:
        for p in model.model.parameters():
            p.requires_grad = False
        keep_suffixes = [f"encoder.layer.{i}." for i in
                         range(23 - a.freeze_layers + 1, 23)]
        suffix_whitelist = ("pooler", "classifier", "dense")
        n_train = 0
        for name, p in model.model.named_parameters():
            if (any(s in name for s in keep_suffixes)
                    or any(w in name for w in suffix_whitelist)):
                p.requires_grad = True
                n_train += p.numel()
        total_p = sum(p.numel() for p in model.model.parameters())
        print(f"冻结 {a.freeze_layers} 层：可训练 {n_train / 1e6:.1f}M / "
              f"总计 {total_p / 1e6:.1f}M 参数", flush=True)

    trainable = [p for p in model.model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=a.lr)
    total_steps = max(1, len(train_loader) * a.epochs)
    warmup = max(1, int(total_steps * a.warmup_ratio))
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda s: min(1.0, (s + 1) / warmup) if s < total_steps
        else max(0.0, (total_steps - s) / max(1, warmup)))
    loss_fn = torch.nn.BCEWithLogitsLoss()

    def evaluate():
        model.model.eval()
        correct = total = 0
        with torch.no_grad():
            for enc, labels in dev_loader:
                enc = {k: v.to(device) for k, v in enc.items()}
                with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=use_amp):
                    logits = model.model(**enc).logits.view(-1)
                pred = (torch.sigmoid(logits.float()) >= 0.5).float().cpu()
                correct += int((pred == labels).sum())
                total += len(labels)
        return correct / max(total, 1)

    log = {"base_model": a.base_model, "train_pairs": len(train), "dev_pairs": len(dev),
           "epochs": [], "device": device, "lr": a.lr, "batch": a.batch,
           "freeze_layers": a.freeze_layers, "max_length": a.max_length}

    def discrimination_probe():
        """关键健康检查：微调后必须仍能区分正负例。

        实测教训：v2-m3 冻结 20 层 + lr 5e-5 跑 2 epoch 后，正例与负例得分都塌到 0.1007
        （完全没有判别力），测试集 hit@1 从 0.645 崩到 0.105。train loss 正常下降、
        dev pair acc 也有 0.80，**从训练指标完全看不出来**，只有这个探针能提前发现。
        """
        model.model.eval()
        pairs = [["奖学金怎么申请", "奖学金每年九月开放申请，成绩要求专业前百分之二十。"],
                 ["奖学金怎么申请", "宿舍熄灯时间为晚上十一点，周末延长到十二点。"]]
        with torch.no_grad():
            s = model.predict(pairs, show_progress_bar=False)
        gap = float(s[0]) - float(s[1])
        model.model.train()
        return {"pos": round(float(s[0]), 4), "neg": round(float(s[1]), 4),
                "gap": round(gap, 4), "healthy": gap > 0.02}

    best_acc, best_state = -1.0, None
    for epoch in range(a.epochs):
        model.model.train()
        t0 = time.time()
        running, seen = 0.0, 0
        for step, (enc, labels) in enumerate(train_loader):
            enc = {k: v.to(device, non_blocking=True) for k, v in enc.items()}
            labels = labels.to(device, non_blocking=True)
            with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=use_amp):
                logits = model.model(**enc).logits.view(-1)
                loss = loss_fn(logits.float(), labels)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            running += float(loss) * len(labels)
            seen += len(labels)
            if step % 25 == 0:
                print(f"  epoch {epoch + 1} step {step}/{len(train_loader)} "
                      f"loss={running / max(seen, 1):.4f} "
                      f"({time.time() - t0:.0f}s)", flush=True)
        dev_acc = evaluate()
        probe = discrimination_probe()
        entry = {"epoch": epoch + 1, "train_loss": round(running / max(seen, 1), 4),
                 "dev_pair_acc": round(dev_acc, 4), "probe": probe,
                 "seconds": round(time.time() - t0, 1)}
        log["epochs"].append(entry)
        print(f"epoch {epoch + 1}: {entry}")
        if not probe["healthy"]:
            print("  ⚠️  判别力探针失败：正负例得分几乎相同，这个权重不可用。"
                  "常见原因：冻结层数过多 + lr 过高，把分类头推到了饱和区。"
                  "建议降低 lr（如 1e-5）或减少冻结层数。", flush=True)
        # 只有"判别力健康"的 epoch 才有资格成为最优权重
        if probe["healthy"] and dev_acc > best_acc:
            best_acc = dev_acc
            best_state = {k: v.detach().cpu().clone() for k, v in model.model.state_dict().items()}

    if best_state is None:
        raise SystemExit(
            "没有任何 epoch 通过判别力探针（正负例得分几乎相同），权重不可用。\n"
            "调整建议：降低 --lr（1e-5 或更低）、减少 --freeze-layers、或缩短 --epochs。")

    out_dir = Path(a.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model.model.load_state_dict(best_state)
    model.model.eval()
    model.save(str(out_dir))
    log["best_dev_pair_acc"] = round(best_acc, 4)
    log["saved_to"] = str(out_dir)
    (TRAIN_DIR / "train_log.json").write_text(json.dumps(log, ensure_ascii=False, indent=1),
                                              encoding="utf-8")
    print(f"\n最佳 dev pair acc = {best_acc:.4f}，权重已存 {out_dir}")
    print("下一步：python benchmark/eval_finetuned_rerank.py")


if __name__ == "__main__":
    main()
