# -*- coding: utf-8 -*-
"""端到端自测：不依赖标注数据，也不需要真的模型。

    AR_EMBEDDER=stub python scripts/selftest.py     # 秒级，跑契约与路由
    python scripts/selftest.py                       # 真实模型，额外验证语义链路

覆盖：建索引（递归/子目录）、增量复用、索引自描述与版本校验、三档 + dense 基线、
k 条数保证、分数单调、JSON 契约、评测（含 dense 对照、未知 chunk_id 报错）、
旧版索引（schema 1）兼容、MCP server 冒烟。
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import retrieval_core as rc  # noqa: E402

CHECKS: list = []


def check(name: str, ok: bool, detail=None) -> bool:
    CHECKS.append({"check": name, "ok": bool(ok), "detail": detail})
    print(f"  {'✓' if ok else '✗'} {name}" + (f"  {detail}" if detail is not None and not ok else ""))
    return bool(ok)


LONG_DOC = """# 奖学金与助学政策

## 申请条件

奖学金申请要求学业成绩位于专业前百分之二十，且无违纪记录。申请材料包括成绩单、
个人陈述与两封推荐信。评审分为书面材料初评与现场答辩两个环节，最终按加权总分排序。

## 时间安排

每年九月一日开放线上申请，九月三十日截止收材料，十一月十五日公示结果。
逾期提交的材料不予受理，补交材料需在公示期内提出并说明理由。

## 助学贷款

家庭经济困难学生可申请国家助学贷款，最高额度为每年一万二千元，在校期间利息由财政补贴。
申请需要提供家庭收入证明与所在地民政部门盖章的困难认定表。
"""

SHORT_DOC = """宿舍熄灯时间为晚上十一点，周末延长到十二点。禁止使用大功率电器，违者扣分。访客需在一楼登记。"""


def build_corpus(root: Path) -> Path:
    docs = root / "docs"
    (docs / "policy").mkdir(parents=True)
    (docs / "policy" / "scholarship.md").write_text(LONG_DOC, encoding="utf-8")
    (docs / "dorm.md").write_text(SHORT_DOC, encoding="utf-8")
    (docs / "note.txt").write_text("校历：秋季学期九月一日开学，一月十五日放假。",
                                   encoding="utf-8")
    return docs


def main() -> int:
    rc.ensure_utf8_stdout()
    stub = os.environ.get("AR_EMBEDDER", "").lower() == "stub"
    print(f"== selftest（{'stub 编码器' if stub else '真实模型'}）==")
    tmp = Path(tempfile.mkdtemp(prefix="ar_selftest_"))
    try:
        docs = build_corpus(tmp)
        idx = tmp / "idx"

        # 1) 建索引：递归 + 子目录 + 多扩展名
        meta = rc.build_index(str(docs), str(idx), quiet=True)
        check("index: 覆盖子目录与 txt/md", meta["n_chunks"] >= 3, meta["n_chunks"])
        check("index: 写出 meta/schema", meta["schema_version"] == rc.SCHEMA_VERSION)
        check("index: chunk_id 带相对路径",
              all("::" in c["chunk_id"] for c in json.loads(
                  (idx / "coarse.json").read_text(encoding="utf-8"))))

        # 2) 增量：不改文档 -> 全部复用
        meta2 = rc.build_index(str(docs), str(idx), quiet=True)
        check("index: 未变更时全部复用", meta2["encoded_docs"] == 0
              and meta2["reused_docs"] == meta2["docs"].__len__(), meta2["reused_docs"])

        # 3) 增量：改一篇 -> 只编码该类
        (docs / "dorm.md").write_text(SHORT_DOC + "电动车禁止进入宿舍楼。", encoding="utf-8")
        meta3 = rc.build_index(str(docs), str(idx), quiet=True)
        check("index: 改一篇只编码一篇", meta3["encoded_docs"] == 1, meta3["encoded_docs"])

        # 4) 检索契约
        ret = rc.Retriever(str(idx))
        for profile in ("fast", "auto", "best", "dense"):
            res, trace = ret.retrieve_with_trace("奖学金怎么申请", k=3, profile=profile)
            scores = [c["score"] for c in res]
            check(f"retrieve[{profile}]: 返回条数 <= k", len(res) <= 3, len(res))
            check(f"retrieve[{profile}]: 分数单调不增",
                  all(a >= b - 1e-6 for a, b in zip(scores, scores[1:])), scores)
            check(f"retrieve[{profile}]: 字段齐全",
                  all({"rank", "chunk_id", "text", "source", "score", "stage"} <= set(c)
                      for c in res))
            check(f"retrieve[{profile}]: trace 有 route/timings",
                  bool(trace.get("route")) and "total_ms" in (trace.get("timings") or {}))

        res_long = ret.retrieve("奖学金怎么申请", k=3, profile="auto")
        check("retrieve: 命中正确文档",
              bool(res_long) and res_long[0]["source"].endswith("scholarship.md"),
              res_long[0]["source"] if res_long else None)

        # 5) k 超过语料规模时不报错、不越界
        res_big = ret.retrieve("奖学金", k=99, profile="auto")
        check("retrieve: k 超语料规模被截断", len(res_big) == len(ret.coarse), len(res_big))

        # 6) 旧版索引（无 meta.json）兼容
        legacy = tmp / "idx_v1"
        shutil.copytree(idx, legacy)
        (legacy / "meta.json").unlink()
        legacy_desc = rc.Retriever(str(legacy)).describe()
        check("legacy: schema 1 索引可读", legacy_desc["schema_version"] == 1)

        # 7) 评测：模板 + 真标注 + dense 基线 + 未知 id 报错
        template = tmp / "ts.json"
        rc.make_eval_template(str(idx), str(template), n=2)
        tpl = json.loads(template.read_text(encoding="utf-8"))
        check("eval: 模板生成", len(tpl["items"]) == 2
              and all(it["relevant_chunks"] for it in tpl["items"]))

        good = tmp / "ts_good.json"
        items = []
        for c in ret.coarse:
            items.append({"query": c["text"][:30], "relevant_chunks": [c["chunk_id"]]})
        good.write_text(json.dumps({"items": items}, ensure_ascii=False), encoding="utf-8")
        result = rc.eval_index(str(idx), str(good), quiet=True)
        check("eval: 含 fast/auto/best + dense 基线",
              {"fast", "auto", "best", "dense"} <= set(result["profiles"]),
              list(result["profiles"]))
        row = result["profiles"]["auto"]
        check("eval: 指标完整",
              {"hit@1", "hit@5", "recall@5", "ndcg@5", "mrr"} <= set(row), list(row))
        check("eval: 自检索命中率 > 0.5", row["hit@1"] > 0.5, row["hit@1"])

        bad = tmp / "ts_bad.json"
        bad.write_text(json.dumps([{"query": "x", "relevant": ["不存在::0"]}],
                                  ensure_ascii=False), encoding="utf-8")
        try:
            rc.eval_index(str(idx), str(bad), quiet=True)
            check("eval: 未知 chunk_id 应报错", False)
        except ValueError as exc:
            check("eval: 未知 chunk_id 报错", "不在当前索引" in str(exc))

        # 8) CLI --json 契约（子进程，验证真实命令行入口）
        env = dict(os.environ)
        proc = subprocess.run(
            [sys.executable, str(HERE / "retrieval_core.py"), "retrieve", "奖学金怎么申请",
             "--index-dir", str(idx), "--profile", "auto", "--json", "--no-rerank"],
            capture_output=True, text=True, encoding="utf-8", env=env)
        ok_cli = proc.returncode == 0
        payload = {}
        if ok_cli:
            try:
                payload = json.loads(proc.stdout.strip().splitlines()[-1])
            except Exception:
                ok_cli = False
        check("cli: retrieve --json 可解析", ok_cli and "results" in payload,
              proc.stderr[-200:] if not ok_cli else None)

        # 9) MCP server 冒烟
        mcp = subprocess.run(
            [sys.executable, str(HERE / "mcp_server.py"), "--index-dir", str(idx), "--selftest"],
            capture_output=True, text=True, encoding="utf-8", env=env)
        check("mcp: selftest 通过", mcp.returncode == 0,
              (mcp.stdout + mcp.stderr)[-300:])

        failed = [c for c in CHECKS if not c["ok"]]
        print(f"\n== {len(CHECKS) - len(failed)}/{len(CHECKS)} 通过 ==")
        if failed:
            print(json.dumps(failed, ensure_ascii=False, indent=2))
        return 0 if not failed else 1
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
