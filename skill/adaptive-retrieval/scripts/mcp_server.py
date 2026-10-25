# -*- coding: utf-8 -*-
"""adaptive-retrieval 的常驻 MCP server（stdio 传输，零第三方依赖）。

为什么需要它：CLI 每次调用都要新起解释器，实测冷启动 ≈ 34-50s（import
sentence-transformers 与加载 bge 模型占绝大部分），而进程内检索只要 8-120ms。
MCP 形态让模型加载只发生一次，后续每次检索都是稳态延迟。

协议：newline-delimited JSON-RPC 2.0（MCP stdio），实现
initialize / notifications/initialized / tools/list / tools/call / ping。

工具：
- retrieval_query  检索（三档 + dense 基线）
- retrieval_index  建/更新索引（默认增量）
- retrieval_eval   离线评测（hit/recall/ndcg/MRR）
- retrieval_stats  索引元信息（不加载模型）

用法（由 harness 拉起，一般不用手动执行）：
    python mcp_server.py --index-dir ./.retrieval_index --preload

自检（不需要 MCP 客户端）：
    python mcp_server.py --selftest
"""
from __future__ import annotations

import argparse
import io
import json
import os
import sys
import threading
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent))

from retrieval_core import (  # noqa: E402
    PRESETS, PROFILES, PROFILE_AUTO, PROFILE_BEST, PROFILE_DENSE, PROFILE_FAST,
    Retriever, _quiet_stdout, build_index, diagnose, ensure_utf8_stdout, eval_index,
    tune_index,
)

SERVER_NAME = "adaptive-retrieval"
SERVER_VERSION = "1.0.0"
SUPPORTED_PROTOCOLS = ("2025-06-18", "2025-03-26", "2024-11-05")
DEFAULT_PROTOCOL = "2025-06-18"

TOOLS: List[dict] = [
    {
        "name": "retrieval_query",
        "description": (
            "在本地文档索引里检索相关段落。三档位自适应：fast 最快、auto 默认"
            "（长文档语料按一致性路由决定是否升级重排）、best 最准。"
            "返回 JSON：results[] 含 chunk_id/text/source/score/stage，"
            "route 说明实际走的检索路径，timings.model_load_ms 为首次调用的冷启动耗时。"
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "检索查询，自然语言"},
                "k": {"type": "integer", "default": 5, "minimum": 1,
                      "description": "返回条数"},
                "profile": {"type": "string", "enum": list(PROFILES) + [PROFILE_DENSE],
                            "default": PROFILE_AUTO,
                            "description": "档位；dense 为无重排对照基线"},
                "index_dir": {"type": "string",
                              "description": "索引目录，缺省用 server 启动参数 / 环境变量"},
                "preset": {"type": "string", "enum": list(PRESETS),
                           "description": "tune 选出的策略预设（默认 base；域相关，先 tune 再定）"},
                "min_score": {"type": "number",
                              "description": "低于该分数的结果不返回（过滤噪声）"},
                "include_text": {"type": "boolean", "default": True,
                                 "description": "是否返回原文；只要定位信息时可设 false"},
            },
            "required": ["query"],
        },
    },
    {
        "name": "retrieval_index",
        "description": (
            "对目录内 .txt/.md 文档建索引（默认递归、增量：只编码新增或变更的文档）。"
            "语料类型（长文档/短段落）在索引时自动判定，决定三档位的实现。"
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "docs_dir": {"type": "string", "description": "文档目录"},
                "index_dir": {"type": "string", "description": "索引输出目录"},
                "force": {"type": "boolean", "default": False,
                          "description": "忽略增量缓存，全量重建"},
                "recursive": {"type": "boolean", "default": True},
                "ext": {"type": "string", "description": "逗号分隔扩展名，默认 .txt,.md,.markdown"},
            },
            "required": ["docs_dir"],
        },
    },
    {
        "name": "retrieval_eval",
        "description": (
            "用标注集评测各档位检索质量（hit/recall/ndcg@k 与 MRR），"
            "并自动附 dense 基线做对照。评测集为 JSON：list 或 {items:[...]}，"
            "每条含 query/question 与 relevant/relevant_chunks（chunk_id 列表）。"
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "testset": {"type": "string", "description": "评测集 JSON 路径"},
                "index_dir": {"type": "string"},
                "profiles": {"type": "string", "default": "fast,auto,best"},
                "k": {"type": "string", "default": "1,5"},
            },
            "required": ["testset"],
        },
    },
    {
        "name": "retrieval_stats",
        "description": "索引元信息：语料类型、块数、平均块长、embedder、来源文件（不加载模型）。",
        "inputSchema": {
            "type": "object",
            "properties": {"index_dir": {"type": "string"}},
        },
    },
    {
        "name": "retrieval_tune",
        "description": (
            "在评测集上比较策略预设（base/fusion/cand_rrf/fast_rrf/multiquery/rerank_ens），"
            "给出建议配置。自带配对 bootstrap（95% 区间）与前半选择/后半复验，"
            "两边同向才推荐，否则建议保持 base。适合在接入新语料后调一次。"
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "testset": {"type": "string", "description": "评测集 JSON 路径"},
                "index_dir": {"type": "string"},
                "profiles": {"type": "string", "default": "fast,auto,best"},
                "presets": {"type": "string",
                            "description": "逗号分隔预设名，缺省用推荐组合"},
                "max_queries": {"type": "integer",
                                "description": "只跑前 N 条（快速试跑）"},
            },
            "required": ["testset"],
        },
    },
    {
        "name": "retrieval_diagnose",
        "description": (
            "诊断瓶颈在哪一环：算第一跳命中率、候选池 recall@10、重排后 hit@1。"
            "若候选池 recall 已接近 1 而 hit@1 低，说明是排序瓶颈（该去微调重排器），"
            "此时再加召回手段无效。调参前先跑这个。"
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "testset": {"type": "string", "description": "评测集 JSON 路径"},
                "index_dir": {"type": "string"},
                "pool": {"type": "integer", "default": 10,
                         "description": "候选池大小（用于评估可救回空间）"},
            },
            "required": ["testset"],
        },
    },
]


def _as_text(payload: Any) -> str:
    return json.dumps(payload, ensure_ascii=False)


class RetrievalServer:
    def __init__(self, index_dir: Optional[str] = None, model_dir: Optional[str] = None,
                 preload: bool = False, disable_rerank: bool = False):
        self.default_index = index_dir or os.environ.get(
            "RETRIEVAL_INDEX_DIR", ".retrieval_index")
        self.model_dir = model_dir
        self.disable_rerank = disable_rerank
        self._cache: Dict[str, Retriever] = {}
        self._cache_lock = threading.Lock()
        self._preload_error: Optional[str] = None
        if preload:
            threading.Thread(target=self._preload, daemon=True).start()

    def _preload(self) -> None:
        try:
            self.get(self.default_index)
        except Exception as exc:  # 索引还没建是正常状态，首次调用会再报一次
            self._preload_error = f"{type(exc).__name__}: {exc}"

    def get(self, index_dir: Optional[str] = None, preset: Optional[str] = None) -> Retriever:
        path = str(Path(index_dir or self.default_index).expanduser().resolve())
        options = PRESETS.get(preset or "base", {})
        key = f"{path}|{preset or 'base'}"
        with self._cache_lock:
            ret = self._cache.get(key)
            if ret is None:
                ret = Retriever(path, self.model_dir, disable_rerank=self.disable_rerank,
                                options=options)
                self._cache[key] = ret
            return ret

    # --- 工具实现 ---
    def tool_query(self, args: dict) -> dict:
        preset = args.get("preset")
        ret = self.get(args.get("index_dir"), preset)
        res, trace = ret.retrieve_with_trace(
            args["query"], k=int(args.get("k", 5)),
            profile=args.get("profile") or PROFILE_AUTO)
        if args.get("min_score") is not None:
            res = [c for c in res if c["score"] >= float(args["min_score"])]
        if args.get("include_text", True) is False:
            res = [{k: v for k, v in c.items() if k != "text"} for c in res]
        return {"query": args["query"], "results": res, "route": trace,
                "corpus": ret.describe()}

    def tool_index(self, args: dict) -> dict:
        exts = [x.strip() for x in (args.get("ext") or ".txt,.md,.markdown").split(",")
                if x.strip()]
        with _quiet_stdout():
            meta = build_index(args["docs_dir"], args.get("index_dir") or self.default_index,
                               self.model_dir,
                               recursive=bool(args.get("recursive", True)),
                               exts=exts, force=bool(args.get("force", False)),
                               quiet=True)
        # 索引变了，缓存里的旧 Retriever 必须失效
        with self._cache_lock:
            self._cache.clear()
        return meta

    def tool_eval(self, args: dict) -> dict:
        with _quiet_stdout():
            result = eval_index(
                args.get("index_dir") or self.default_index,
                args["testset"],
                profiles=[x.strip() for x in str(args.get("profiles", "fast,auto,best")).split(",") if x.strip()],
                k_values=[int(x) for x in str(args.get("k", "1,5")).split(",") if x.strip()],
                model_dir=self.model_dir, quiet=True)
        return result

    def tool_stats(self, args: dict) -> dict:
        return self.get(args.get("index_dir")).describe()

    def tool_tune(self, args: dict) -> dict:
        presets = [x.strip() for x in str(args.get("presets", "")).split(",") if x.strip()]
        with _quiet_stdout():
            result = tune_index(
                args.get("index_dir") or self.default_index,
                args["testset"],
                profiles=[x.strip() for x in str(args.get("profiles", "fast,auto,best")).split(",") if x.strip()],
                model_dir=self.model_dir,
                presets=presets or None,
                max_queries=args.get("max_queries"))
        with self._cache_lock:
            self._cache.clear()
        return result

    def tool_diagnose(self, args: dict) -> dict:
        pool = int(args.get("pool", 10))
        with _quiet_stdout():
            result = diagnose(
                args.get("index_dir") or self.default_index,
                args["testset"], pool=pool, model_dir=self.model_dir)
        return result

    def call_tool(self, name: str, args: dict) -> dict:
        mapping = {
            "retrieval_query": self.tool_query,
            "retrieval_index": self.tool_index,
            "retrieval_eval": self.tool_eval,
            "retrieval_stats": self.tool_stats,
            "retrieval_tune": self.tool_tune,
            "retrieval_diagnose": self.tool_diagnose,
        }
        if name not in mapping:
            raise KeyError(f"未知工具: {name}")
        return mapping[name](args)


def _error(msg_id, code: int, message: str, data: Any = None) -> dict:
    err: Dict[str, Any] = {"code": code, "message": message}
    if data is not None:
        err["data"] = data
    return {"jsonrpc": "2.0", "id": msg_id, "error": err}


def handle(server: RetrievalServer, msg: dict) -> Optional[dict]:
    method = msg.get("method")
    msg_id = msg.get("id")
    params = msg.get("params") or {}

    if method == "initialize":
        requested = params.get("protocolVersion")
        version = requested if requested in SUPPORTED_PROTOCOLS else DEFAULT_PROTOCOL
        return {"jsonrpc": "2.0", "id": msg_id, "result": {
            "protocolVersion": version,
            "capabilities": {"tools": {"listChanged": False}},
            "serverInfo": {"name": SERVER_NAME, "version": SERVER_VERSION},
            "instructions": (
                "本地文档自适应检索。先 retrieval_stats 看索引状态；"
                "没有索引就 retrieval_index 建；检索用 retrieval_query"
                "（profile=auto 默认，fast 更快，best 更准）。"),
        }}

    if method in ("notifications/initialized", "notifications/cancelled"):
        return None

    if method == "ping":
        return {"jsonrpc": "2.0", "id": msg_id, "result": {}}

    if method == "tools/list":
        return {"jsonrpc": "2.0", "id": msg_id, "result": {"tools": TOOLS}}

    if method == "tools/call":
        name = params.get("name")
        args = params.get("arguments") or {}
        try:
            payload = server.call_tool(name, args)
            return {"jsonrpc": "2.0", "id": msg_id, "result": {
                "content": [{"type": "text", "text": _as_text(payload)}],
                "structuredContent": payload,
                "isError": False,
            }}
        except Exception as exc:
            return {"jsonrpc": "2.0", "id": msg_id, "result": {
                "content": [{"type": "text", "text": _as_text({
                    "error": type(exc).__name__, "message": str(exc),
                    "hint": "索引不存在时先调用 retrieval_index；"
                            "模型/依赖缺失见 stderr 日志"})}],
                "isError": True,
            }}

    if method in ("resources/list", "prompts/list"):
        key = "resources" if method.startswith("resources") else "prompts"
        return {"jsonrpc": "2.0", "id": msg_id, "result": {key: []}}

    return _error(msg_id, -32601, f"未实现的方法: {method}")


def serve(server: RetrievalServer) -> int:
    ensure_utf8_stdout()
    stream = io.TextIOWrapper(sys.stdin.buffer, encoding="utf-8", errors="replace")
    out = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace", newline="\n")
    for line in stream:
        line = line.strip()
        if not line:
            continue
        try:
            msg = json.loads(line)
        except json.JSONDecodeError as exc:
            out.write(_as_text(_error(None, -32700, f"JSON 解析失败: {exc}")) + "\n")
            out.flush()
            continue
        try:
            resp = handle(server, msg)
        except Exception:
            resp = _error(msg.get("id"), -32603, "内部错误",
                          traceback.format_exc(limit=3))
        if resp is not None:
            out.write(_as_text(resp) + "\n")
            out.flush()
    return 0


def selftest(index_dir: str) -> int:
    """模拟一个 MCP 客户端走完整握手 + 四个工具，用于安装后验证。"""
    ensure_utf8_stdout()
    server = RetrievalServer(index_dir)
    checks: List[dict] = []

    def call(name: str, args: dict) -> dict:
        resp = handle(server, {"jsonrpc": "2.0", "id": 1, "method": "tools/call",
                               "params": {"name": name, "arguments": args}})
        assert resp is not None
        return resp["result"]

    init = handle(server, {"jsonrpc": "2.0", "id": 0, "method": "initialize",
                           "params": {"protocolVersion": "2025-06-18"}})
    checks.append({"step": "initialize",
                   "ok": init["result"]["serverInfo"]["name"] == SERVER_NAME,
                   "detail": init["result"]["protocolVersion"]})
    tools = handle(server, {"jsonrpc": "2.0", "id": 2, "method": "tools/list"})
    checks.append({"step": "tools/list", "ok": len(tools["result"]["tools"]) == len(TOOLS),
                   "detail": [t["name"] for t in tools["result"]["tools"]]})

    stats = call("retrieval_stats", {})
    checks.append({"step": "retrieval_stats", "ok": not stats.get("isError"),
                   "detail": json.loads(stats["content"][0]["text"]).get("n_chunks")})

    q = call("retrieval_query", {"query": "测试查询", "k": 3, "profile": "auto"})
    payload = json.loads(q["content"][0]["text"])
    checks.append({"step": "retrieval_query", "ok": not q.get("isError"),
                   "detail": {"n_results": len(payload.get("results", [])),
                              "route": (payload.get("route") or {}).get("route")}})

    ok = all(c["ok"] for c in checks)
    print(json.dumps({"selftest": "pass" if ok else "fail", "checks": checks},
                     ensure_ascii=False, indent=2))
    return 0 if ok else 1


def main(argv: Optional[list] = None) -> int:
    p = argparse.ArgumentParser(description="adaptive-retrieval MCP server (stdio)")
    p.add_argument("--index-dir", default=os.environ.get("RETRIEVAL_INDEX_DIR", ".retrieval_index"))
    p.add_argument("--model-dir", default=None)
    p.add_argument("--preload", action="store_true", help="启动后台预热模型，隐藏首次调用冷启动")
    p.add_argument("--no-rerank", action="store_true")
    p.add_argument("--selftest", action="store_true", help="本地自检并退出（不进入 MCP 循环）")
    p.add_argument("--tools", action="store_true", help="打印工具清单后退出")
    args = p.parse_args(argv)

    if args.tools:
        print(json.dumps(TOOLS, ensure_ascii=False, indent=2))
        return 0
    if args.selftest:
        return selftest(args.index_dir)

    server = RetrievalServer(args.index_dir, args.model_dir,
                             preload=args.preload, disable_rerank=args.no_rerank)
    return serve(server)


if __name__ == "__main__":
    raise SystemExit(main())
