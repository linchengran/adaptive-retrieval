# -*- coding: utf-8 -*-
"""学习型路由器训练：在 train 集上构造标签、5 折交叉验证、拟合全量模型并保存。

标签定义（只依赖 train 集，测试集零参与）：
    y = 1  当且仅当 dense top1 错误 且 rerank top1 正确（rerank 能救回该查询）
    y = 0  dense top1 正确（rerank 冗余甚至有害）或 双双错误（rerank 无用）

基线对照：all-fast（全走快路径）、all-slow（全量 rerank）、agreement 规则。

用法: python benchmark/train_router_m3.py
输出: experiments/m3_learned_router/{model.pkl, train_report.json}
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from benchmark.corpus import MAIN_PROJECT_DIR
from benchmark.metrics import average, chunk_metrics
from benchmark.retrievers import (
    BM25Retriever,
    DensePlusRerank,
    DenseRetriever,
    LearnedRouterRetriever,
)

TRAIN_PATH = MAIN_PROJECT_DIR / "experiments" / "splits" / "train.json"
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _paths import REPO_ROOT  # noqa: E402
OUT_DIR = REPO_ROOT / "experiments" / "m3_learned_router"
TOP_K = 5


def main():
    dense = DenseRetriever()
    bm25 = BM25Retriever()
    rerank_arm = DensePlusRerank()
    train = json.load(TRAIN_PATH.open(encoding="utf-8"))
    print(f"训练集: {len(train)} 条")

    X, y, extra = [], [], []
    for item in train:
        rel = set(item.get("relevant_chunks", []))
        d_hits = dense.retrieve_with_scores(item["question"], top_k=5)
        b_ids = [c.chunk_id for c in bm25.retrieve(item["question"], top_k=5)]
        r_ids = [c.chunk_id for c in rerank_arm.retrieve(item["question"], top_k=TOP_K)]
        dense_ok = bool(d_hits) and d_hits[0][0].chunk_id in rel
        rerank_ok = bool(r_ids) and r_ids[0] in rel
        X.append(LearnedRouterRetriever.features(item["question"], d_hits, b_ids))
        y.append(1 if (not dense_ok and rerank_ok) else 0)
        extra.append({"agree": LearnedRouterRetriever.features(item["question"], d_hits, b_ids)[0]})

    X, y = np.array(X), np.array(y)
    print(f"标签分布: y=1（需要 rerank 救回）{y.sum()}/{len(y)}")

    # ---- 交叉验证 ----
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold, cross_val_predict
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline
    from sklearn.metrics import roc_auc_score

    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, class_weight="balanced"))
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    proba = cross_val_predict(clf, X, y, cv=cv, method="predict_proba")[:, 1]
    auc = roc_auc_score(y, proba)

    pred = (proba >= 0.5).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum()); fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum()); tn = int(((pred == 0) & (y == 0)).sum())
    print(f"5折CV: AUC={auc:.4f}  预测需rerank {tp+fp} 条（对{tp}/错{fp}）  预测免rerank {tn+fn} 条（其中漏救{fn}）")

    # 基线对照（同一 CV 口径下用标签直接计算）
    y_arr = np.array(y)
    agree = np.array([e["agree"] for e in extra])
    baselines = {
        "all_fast(全免rerank)": {"accuracy": float((y_arr == 0).mean())},
        "all_slow(全量rerank)": {"accuracy": 1.0},
        "agreement规则": {
            "accuracy": float(((agree == 0) == (y_arr == 1)).mean()),
        },
    }
    baselines["learned_LR"] = {"accuracy": float((pred == y_arr).mean()), "auc": round(float(auc), 4)}
    for name, b in baselines.items():
        print(f"基线 {name}: {b}")

    # ---- 全量拟合并保存 ----
    final = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, class_weight="balanced"))
    final.fit(X, y)

    import joblib
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump({
        "model": final,
        "features": [
            "top1_agreement", "top3_overlap", "m12", "m15",
            "dense_top1_score", "dense_score_entropy", "ascii_ratio", "query_len",
        ],
        "label_definition": "y=1 iff dense top1 wrong AND rerank top1 correct",
        "trained_on": str(TRAIN_PATH),
    }, OUT_DIR / "model.pkl")

    report = {
        "n_train": len(train),
        "positive_rate": round(float(y.mean()), 4),
        "cv": {"auc": round(float(auc), 4), "tp": tp, "fp": fp, "fn": fn, "tn": tn},
        "baselines": baselines,
        "feature_importance": {
            name: round(float(w), 4) for name, w in zip(
                ["top1_agreement", "top3_overlap", "m12", "m15",
                 "dense_top1_score", "dense_score_entropy", "ascii_ratio", "query_len"],
                final[-1].coef_[0],
            )
        },
    }
    with (OUT_DIR / "train_report.json").open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"模型与报告已保存: {OUT_DIR}")


if __name__ == "__main__":
    main()
