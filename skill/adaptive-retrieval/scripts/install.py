# -*- coding: utf-8 -*-
"""把 adaptive-retrieval 安装进各 agent harness（Claude Code / Codex / ZCode / Pi / DeepSeek Harness）。

同一份 skill 目录按各 harness 的约定落地：

| target       | user scope                 | project scope        |
|--------------|----------------------------|----------------------|
| claude-code  | ~/.claude/skills           | ./.claude/skills     |
| codex        | ~/.codex/skills            | ./.codex/skills      |
| zcode        | ~/.zcode/skills            | ./.zcode/skills      |
| pi           | ~/.pi/agent/skills         | ./.pi/agent/skills   |
| dsh          | ~/.dsh/skills              | ./.dsh/skills        |
| agents       | ~/.agents/skills（共享约定）| ./.agents/skills     |

默认只装 skill（每个 harness 都能用，各自用命令行跑检索）。加 --mcp 才会把
常驻 MCP server 注册进 harness 配置——这一步能消掉每次 30s+ 的冷启动，
但会改动 harness 自己的配置文件，所以默认不动它。

常用：
    python install.py --list
    python install.py --target all --dry-run
    python install.py --target claude-code --mcp
    python install.py --target all --check
    python install.py --target all --uninstall
    python install.py --plugin-out ./dist/adaptive-retrieval   # 生成 Claude Code/ZCode 插件包
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

SKILL_NAME = "adaptive-retrieval"
MARKER = ".ar-install.json"
BUNDLE_DIR = Path(__file__).resolve().parent.parent          # .../skill/adaptive-retrieval
SCRIPT_DIR = BUNDLE_DIR / "scripts"


def _home() -> Path:
    return Path(os.environ.get("USERPROFILE") or os.environ.get("HOME") or Path.home())


TARGETS: Dict[str, dict] = {
    "claude-code": {
        "user": _home() / ".claude" / "skills",
        "project": Path(".claude") / "skills",
        "mcp": "claude-json",
        "note": "Claude Code 原生 skills 目录；插件市场形态用 --plugin-out",
    },
    "codex": {
        "user": _home() / ".codex" / "skills",
        "project": Path(".codex") / "skills",
        "mcp": "codex-toml",
        "note": "Codex 原生 skills 目录；MCP 写 ~/.codex/config.toml",
    },
    "zcode": {
        "user": _home() / ".zcode" / "skills",
        "project": Path(".zcode") / "skills",
        "mcp": "zcode-json",
        "note": "ZCode MCP 写 ~/.zcode/cli/config.json 的嵌套键 mcp.servers；插件形态用 --plugin-out",
    },
    "pi": {
        "user": _home() / ".pi" / "agent" / "skills",
        "project": Path(".pi") / "agent" / "skills",
        "mcp": None,
        "note": "Pi 的 skill 目录；MCP 配置格式随版本变化，未自动写入",
    },
    "dsh": {
        "user": _home() / ".dsh" / "skills",
        "project": Path(".dsh") / "skills",
        "mcp": "dsh-yaml",
        "note": "DeepSeek Harness；MCP 写 ~/.dsh/profiles/<profile>/cordis.patch.yml",
    },
    "agents": {
        "user": _home() / ".agents" / "skills",
        "project": Path(".agents") / "skills",
        "mcp": None,
        "note": "跨 harness 的共享约定，DSH 等会扫描该目录",
    },
}


# ------------------------------------------------------------------ 工具
def _log(msg: str) -> None:
    print(msg, flush=True)


def _skill_dest(target: str, scope: str, dest_override: Optional[str]) -> Path:
    if dest_override:
        return Path(dest_override).expanduser()
    root = TARGETS[target][scope]
    return (root if root.is_absolute() else Path.cwd() / root) / SKILL_NAME


def _copy_bundle(dest: Path, link: bool) -> str:
    """把 bundle 放到 dest；link=True 时用符号链接/目录联接（改代码即生效）。"""
    if dest.exists() or dest.is_symlink():
        if dest.is_symlink() or dest.is_junction():
            dest.unlink()
        else:
            shutil.rmtree(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    if link:
        try:
            if os.name == "nt":
                subprocess.run(["cmd", "/c", "mklink", "/J", str(dest), str(BUNDLE_DIR)],
                               check=True, capture_output=True)
            else:
                os.symlink(BUNDLE_DIR, dest, target_is_directory=True)
            return "junction" if os.name == "nt" else "symlink"
        except Exception:
            pass  # 无权限时静默退回复制
    shutil.copytree(BUNDLE_DIR, dest,
                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc", MARKER))
    return "copy"


def _write_marker(dest: Path, target: str, scope: str, mode: str) -> None:
    (dest / MARKER).write_text(json.dumps({
        "skill": SKILL_NAME, "target": target, "scope": scope, "mode": mode,
        "source": str(BUNDLE_DIR), "installed_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "version": _skill_version(),
    }, ensure_ascii=False, indent=1), encoding="utf-8")


def _skill_version() -> str:
    try:
        text = (BUNDLE_DIR / "SKILL.md").read_text(encoding="utf-8-sig")
        for line in text.splitlines():
            if line.strip().startswith("version:"):
                return line.split(":", 1)[1].strip().strip("\"'")
    except Exception:
        pass
    return "unknown"


def _frontmatter(path: Path) -> dict:
    """解析 SKILL.md 的 YAML frontmatter。

    能 import yaml 时用真解析器（顺手完成语法校验——description 里一个裸冒号就能让
    严格的 harness 整个拒绝加载该 skill），否则退回最小解析。
    解析失败返回 {"__invalid__": 原因}。
    """
    try:
        text = path.read_text(encoding="utf-8-sig")
    except Exception:
        return {}
    if not text.startswith("---"):
        return {}
    body = text.split("---", 2)
    if len(body) < 3:
        return {}
    try:
        import yaml  # type: ignore
        try:
            data = yaml.safe_load(body[1])
        except Exception as exc:
            return {"__invalid__": f"YAML 解析失败: {exc}"}
        return data if isinstance(data, dict) else {"__invalid__": "frontmatter 不是映射"}
    except ModuleNotFoundError:
        pass
    out: Dict[str, str] = {}
    for line in body[1].splitlines():
        if ":" in line and not line.startswith((" ", "\t")):
            k, v = line.split(":", 1)
            out[k.strip()] = v.strip().strip("\"'")
    return out


def _has_dep(name: str) -> bool:
    """find_spec 不真正 import，避免为了一次检查付 25s 的 sentence-transformers 导入。"""
    try:
        return importlib.util.find_spec(name) is not None
    except Exception:
        return False


# ------------------------------------------------------------------ MCP 注册
def _mcp_argv(index_dir: str) -> Tuple[str, List[str], str]:
    python = sys.executable
    args = [str(SCRIPT_DIR / "mcp_server.py"), "--index-dir", index_dir, "--preload"]
    return python, args, str(BUNDLE_DIR)


def _backup(path: Path) -> Optional[Path]:
    if not path.exists():
        return None
    bak = path.with_suffix(path.suffix + f".bak.{time.strftime('%Y%m%d%H%M%S')}")
    shutil.copy2(path, bak)
    return bak


def register_mcp(target: str, scope: str, index_dir: str, dry_run: bool) -> List[str]:
    """把 MCP server 写进对应 harness 的配置；返回人类可读的动作列表。"""
    kind = TARGETS[target]["mcp"]
    python, args, cwd = _mcp_argv(index_dir)
    notes: List[str] = []
    if not kind:
        notes.append(f"{target}: 未内置 MCP 配置写入（{TARGETS[target]['note']}）")
        return notes

    if kind == "claude-json":
        path = Path.cwd() / ".mcp.json" if scope == "project" else _home() / ".claude.json"
        entry = {"command": python, "args": args, "cwd": cwd}
        return _merge_json_mcp(path, entry, dry_run)

    if kind == "codex-toml":
        path = (_home() / ".codex" / "config.toml") if scope == "user" \
            else (Path.cwd() / ".codex" / "config.toml")
        block = (
            "\n# >>> adaptive-retrieval（由 install.py 写入，--uninstall 可移除）\n"
            f"[mcp_servers.{SKILL_NAME}]\n"
            f"command = {json.dumps(python)}\n"
            f"args = {json.dumps(args)}\n"
            f"cwd = {json.dumps(cwd)}\n"
            "startup_timeout_sec = 60\n"
            "tool_timeout_sec = 120\n"
            "# <<< adaptive-retrieval\n"
        )
        return _append_block(path, block, f"[mcp_servers.{SKILL_NAME}]", dry_run)

    if kind == "zcode-json":
        path = (Path.cwd() / ".zcode" / "config.json") if scope == "project" \
            else _home() / ".zcode" / "cli" / "config.json"
        entry = {"type": "stdio", "command": python, "args": args}
        return _merge_nested_json_mcp(path, entry, dry_run)

    if kind == "dsh-yaml":
        path = _dsh_patch_path()
        block = (
            "\n# >>> adaptive-retrieval（由 install.py 写入，--uninstall 可移除）\n"
            "- id: mcp-adaptive-retrieval\n"
            "  name: \"@deepseek-ai/dsh-mcp-client\"\n"
            "  config:\n"
            "    serverName: retrieval\n"
            "    transport: stdio\n"
            f"    command: {json.dumps(python)}\n"
            f"    args: {json.dumps(args)}\n"
            f"    cwd: {json.dumps(cwd)}\n"
            "    toolCallTimeoutMs: 120000\n"
            "    failOnStartupError: false\n"
            "# <<< adaptive-retrieval\n"
        )
        return _append_block(path, block, "mcp-adaptive-retrieval", dry_run)

    return notes


def _dsh_patch_path() -> Path:
    env = os.environ.get("DSH_PROFILE_PATCH")
    if env:
        return Path(env).expanduser()
    root = _home() / ".dsh" / "profiles"
    for name in ("web", "desktop", "default"):
        cand = root / name / "cordis.patch.yml"
        if cand.exists():
            return cand
    return root / "web" / "cordis.patch.yml"


def _merge_json_mcp(path: Path, entry: dict, dry_run: bool) -> List[str]:
    data: dict = {}
    if path.exists():
        try:
            data = json.loads(path.read_text(encoding="utf-8-sig"))
        except Exception as exc:
            return [f"{path}: 现有 JSON 无法解析（{exc}），跳过 MCP 注册"]
    servers = data.setdefault("mcpServers", {})
    if servers.get(SKILL_NAME) == entry:
        return [f"{path}: 已存在相同 MCP 配置，跳过"]
    servers[SKILL_NAME] = entry
    if dry_run:
        return [f"[dry-run] 将写入 {path}: mcpServers.{SKILL_NAME}"]
    path.parent.mkdir(parents=True, exist_ok=True)
    bak = _backup(path)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    return [f"{path}: 已写入 mcpServers.{SKILL_NAME}" + (f"（备份 {bak.name}）" if bak else "")]


def _merge_nested_json_mcp(path: Path, entry: dict, dry_run: bool) -> List[str]:
    """ZCode 形态：server 挂在嵌套键 mcp.servers 下（不是 Claude 的顶层 mcpServers）。"""
    data: dict = {}
    if path.exists():
        try:
            data = json.loads(path.read_text(encoding="utf-8-sig"))
        except Exception as exc:
            return [f"{path}: 现有 JSON 无法解析（{exc}），跳过 MCP 注册"]
    if not isinstance(data.get("mcp"), dict):
        data["mcp"] = {}
    servers = data["mcp"].setdefault("servers", {})
    if servers.get(SKILL_NAME) == entry:
        return [f"{path}: 已存在相同 MCP 配置，跳过"]
    servers[SKILL_NAME] = entry
    if dry_run:
        return [f"[dry-run] 将写入 {path}: mcp.servers.{SKILL_NAME}"]
    path.parent.mkdir(parents=True, exist_ok=True)
    bak = _backup(path)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    return [f"{path}: 已写入 mcp.servers.{SKILL_NAME}" + (f"（备份 {bak.name}）" if bak else "")]


def _append_block(path: Path, block: str, marker: str, dry_run: bool) -> List[str]:
    existing = path.read_text(encoding="utf-8-sig") if path.exists() else ""
    if marker in existing:
        return [f"{path}: 已包含 {marker}，跳过"]
    if dry_run:
        return [f"[dry-run] 将向 {path} 追加 {marker}"]
    path.parent.mkdir(parents=True, exist_ok=True)
    bak = _backup(path)
    with path.open("a", encoding="utf-8", newline="\n") as f:
        f.write(block)
    return [f"{path}: 已追加 {marker}" + (f"（备份 {bak.name}）" if bak else "")]


def unregister_mcp(target: str, scope: str, dry_run: bool) -> List[str]:
    kind = TARGETS[target]["mcp"]
    if not kind:
        return []
    if kind == "claude-json":
        path = Path.cwd() / ".mcp.json" if scope == "project" else _home() / ".claude.json"
        if not path.exists():
            return []
        try:
            data = json.loads(path.read_text(encoding="utf-8-sig"))
        except Exception:
            return [f"{path}: 解析失败，未改动"]
        if SKILL_NAME not in (data.get("mcpServers") or {}):
            return []
        if dry_run:
            return [f"[dry-run] 将从 {path} 移除 mcpServers.{SKILL_NAME}"]
        data["mcpServers"].pop(SKILL_NAME, None)
        _backup(path)
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        return [f"{path}: 已移除 mcpServers.{SKILL_NAME}"]

    if kind == "zcode-json":
        path = (Path.cwd() / ".zcode" / "config.json") if scope == "project" \
            else _home() / ".zcode" / "cli" / "config.json"
        if not path.exists():
            return []
        try:
            data = json.loads(path.read_text(encoding="utf-8-sig"))
        except Exception:
            return [f"{path}: 解析失败，未改动"]
        servers = (data.get("mcp") or {}).get("servers") or {}
        if SKILL_NAME not in servers:
            return []
        if dry_run:
            return [f"[dry-run] 将从 {path} 移除 mcp.servers.{SKILL_NAME}"]
        data["mcp"]["servers"].pop(SKILL_NAME, None)
        _backup(path)
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        return [f"{path}: 已移除 mcp.servers.{SKILL_NAME}"]

    path = _dsh_patch_path() if kind == "dsh-yaml" else (
        (_home() / ".codex" / "config.toml") if scope == "user"
        else Path.cwd() / ".codex" / "config.toml")
    if not path.exists():
        return []
    text = path.read_text(encoding="utf-8-sig")
    if "# >>> adaptive-retrieval" not in text:
        return []
    if dry_run:
        return [f"[dry-run] 将从 {path} 移除 adaptive-retrieval 配置块"]
    start = text.index("# >>> adaptive-retrieval")
    end = text.index("# <<< adaptive-retrieval") + len("# <<< adaptive-retrieval")
    new = (text[:start].rstrip("\n") + "\n" + text[end:].lstrip("\n"))
    _backup(path)
    path.write_text(new, encoding="utf-8")
    return [f"{path}: 已移除 adaptive-retrieval 配置块"]


# ------------------------------------------------------------------ 插件包
def build_plugin(out_dir: str, index_dir: str) -> Path:
    """生成 Claude Code / ZCode 兼容的插件目录（.claude-plugin/plugin.json + skills/ + .mcp.json）"""
    out = Path(out_dir).expanduser()
    if out.exists():
        shutil.rmtree(out)
    (out / ".claude-plugin").mkdir(parents=True)
    skills = out / "skills"
    shutil.copytree(BUNDLE_DIR, skills / SKILL_NAME,
                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc", MARKER))
    python, args, cwd = _mcp_argv(index_dir)

    manifest = {
        "name": SKILL_NAME,
        "version": _skill_version(),
        "description": "自适应本地检索：三档（fast/auto/best）按语料类型自动切换，零配置建索引，自带评测。",
        "author": {"name": "adaptive-retrieval"},
        "keywords": ["retrieval", "rag", "search", "rerank", "hybrid-search"],
    }
    (out / ".claude-plugin" / "plugin.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    (out / ".mcp.json").write_text(json.dumps({
        "mcpServers": {SKILL_NAME: {"command": python, "args": args, "cwd": cwd}}
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    commands = out / "commands"
    commands.mkdir()
    (commands / "retrieve.md").write_text(
        "---\n"
        "description: 在本地文档索引里检索相关段落（adaptive-retrieval）\n"
        "---\n\n"
        "用 adaptive-retrieval 的 MCP 工具 `retrieval_query` 回答：$ARGUMENTS\n\n"
        "1. 先调用 `retrieval_stats` 确认索引存在；没有就调用 `retrieval_index`，"
        "文档目录取用户指定或当前项目内最相关的文档目录。\n"
        "2. 调用 `retrieval_query`（profile=auto，k=5），按返回的 chunk_id/source 引用原文。\n"
        "3. 结果为空或明显不相关时，改用 profile=best 再试一次，并说明差异。\n",
        encoding="utf-8")
    (out / "README.md").write_text(
        f"# {SKILL_NAME}（插件包）\n\n"
        "由 `python scripts/install.py --plugin-out` 生成。\n\n"
        "Claude Code：`/plugin marketplace add <this dir>` 后 `/plugin install adaptive-retrieval`；\n"
        "或直接把本目录放进 `~/.claude/plugins/`。ZCode 兼容同一格式。\n",
        encoding="utf-8")
    return out


# ------------------------------------------------------------------ 检查 / 列表
def check(index_dir: str) -> int:
    python = sys.executable
    deps = {n: _has_dep(n) for n in ("numpy", "jieba", "rank_bm25", "sentence_transformers")}
    _log("== 依赖 ==")
    for name, ok in deps.items():
        _log(f"  {'✓' if ok else '✗'} {name}")
    _log(f"  python: {python}")

    _log("\n== skill 自身 ==")
    bundle_fm = _frontmatter(BUNDLE_DIR / "SKILL.md")
    if "__invalid__" in bundle_fm:
        _log(f"  ✗ {bundle_fm['__invalid__']}（严格解析的 harness 会拒绝加载）")
    else:
        _log(f"  {'✓' if bundle_fm.get('name') == SKILL_NAME and bundle_fm.get('description') else '✗'} "
             f"name={bundle_fm.get('name')} version={bundle_fm.get('version')} "
             f"desc_len={len(bundle_fm.get('description') or '')}")

    _log("\n== 各 harness 安装状态 ==")
    rows = []
    for target in TARGETS:
        for scope in ("user", "project"):
            dest = _skill_dest(target, scope, None)
            skill_md = dest / "SKILL.md"
            if not skill_md.exists():
                state = "-"
            else:
                fm = _frontmatter(skill_md)
                if "__invalid__" in fm:
                    state = "invalid-yaml"
                elif fm.get("name") == SKILL_NAME and fm.get("description"):
                    state = "ok"
                else:
                    state = "invalid"
            if state != "-":
                rows.append((target, scope, str(dest), state))
                _log(f"  {target:12s} {scope:8s} {state:8s} {dest}")
    if not rows:
        _log("  （尚未安装到任何 harness）")

    _log("\n== 索引 ==")
    idx = Path(index_dir).expanduser()
    if (idx / "coarse.json").exists():
        try:
            meta = json.loads((idx / "meta.json").read_text(encoding="utf-8-sig")) \
                if (idx / "meta.json").exists() else {}
            if meta:
                _log(f"  ✓ {idx}  {meta.get('n_chunks', '?')} 粗块 / "
                     f"{meta.get('corpus_type', '?')} 语料 / schema {meta.get('schema_version')}")
            else:
                n = len(json.loads((idx / "coarse.json").read_text(encoding="utf-8-sig")))
                _log(f"  ✓ {idx}  {n} 粗块 / 无 meta.json（schema 1 旧索引，可正常检索）")
        except Exception as exc:
            _log(f"  ! {idx} 元信息读取失败: {exc}")
    else:
        _log(f"  ✗ {idx} 不存在，先运行："
             f"python scripts/retrieval_core.py index <文档目录> --index-dir {idx}")

    _log("\n== MCP 自检 ==")
    if (idx / "coarse.json").exists():
        env = dict(os.environ, AR_EMBEDDER=os.environ.get("AR_EMBEDDER", ""))
        if env.get("AR_EMBEDDER") == "stub":
            # 自检要覆盖"真的能检索"，stub 只在没有模型时才用；有模型就不要它
            env.pop("AR_EMBEDDER")
        _log("  （真实模型自检，首次需加载模型，约 30s）")
        proc = subprocess.run([python, str(SCRIPT_DIR / "mcp_server.py"),
                               "--index-dir", str(idx), "--selftest"],
                              capture_output=True, text=True, encoding="utf-8", env=env)
        ok = proc.returncode == 0
        _log(f"  {'✓ 通过' if ok else '✗ 失败'}")
        if not ok:
            _log((proc.stdout or "")[-600:] + (proc.stderr or "")[-600:])
    else:
        _log("  （先建索引才能自检）")

    bad = ((not all(deps.values())) or any(r[3] != "ok" for r in rows)
           or "__invalid__" in bundle_fm or bundle_fm.get("name") != SKILL_NAME)
    return 1 if bad else 0


def list_targets() -> int:
    _log(f"{'target':12s} {'user scope':40s} project scope")
    for name, spec in TARGETS.items():
        user = str(spec["user"])
        proj = str(spec["project"])
        _log(f"{name:12s} {user:40s} {proj}")
        _log(f"{'':12s} MCP: {spec['mcp'] or '—'}  {spec['note']}")
    return 0


# ------------------------------------------------------------------ 主流程
def install(targets: List[str], scope: str, dest: Optional[str], link: bool,
            with_mcp: bool, index_dir: str, dry_run: bool) -> int:
    if not (BUNDLE_DIR / "SKILL.md").exists():
        _log(f"[错误] 找不到 skill bundle: {BUNDLE_DIR}")
        return 2

    for target in targets:
        dest_dir = _skill_dest(target, scope, dest)
        action = "链接" if link else "复制"
        if dry_run:
            _log(f"[dry-run] {target}: {action} {BUNDLE_DIR} -> {dest_dir}")
        else:
            mode = _copy_bundle(dest_dir, link)
            if mode == "copy":
                # 链接模式下写 marker 会顺着 junction 写回源目录，所以只对复制写入
                _write_marker(dest_dir, target, scope, mode)
            _log(f"✓ {target}: {mode} -> {dest_dir}")
        if with_mcp:
            for line in register_mcp(target, scope, index_dir, dry_run):
                _log(f"  MCP {line}")

    if not dry_run:
        _log("\n下一步：")
        _log(f"  1) 建索引: python \"{SCRIPT_DIR / 'retrieval_core.py'}\" index <文档目录> "
             f"--index-dir {index_dir}")
        _log(f"  2) 验证:   python \"{SCRIPT_DIR / 'install.py'}\" check --index-dir {index_dir}")
        if with_mcp:
            _log("  3) 重启对应 harness 使 MCP 配置生效（skill 本身无需重启）")
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    for _s in (sys.stdout, sys.stderr):
        try:
            if hasattr(_s, "reconfigure"):
                _s.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass
    p = argparse.ArgumentParser(
        description="adaptive-retrieval 多 harness 安装器",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="target 可选: " + ", ".join(TARGETS) + ", all")
    p.add_argument("--target", default=None, help="单个 target 或 all")
    p.add_argument("--targets", default=None, help="逗号分隔多个 target")
    p.add_argument("--scope", choices=["user", "project"], default="user")
    p.add_argument("--dest", default=None, help="直接指定 skill 安装目录（覆盖 scope）")
    p.add_argument("--link", action="store_true", help="用符号链接/目录联接代替复制（改代码即生效）")
    p.add_argument("--mcp", action="store_true", help="同时注册常驻 MCP server（消除冷启动）")
    p.add_argument("--index-dir", default=str(Path.cwd() / ".retrieval_index"))
    p.add_argument("--dry-run", action="store_true", help="只打印将要做什么")
    p.add_argument("--check", action="store_true", help="检查依赖 / 安装状态 / 索引 / MCP 自检")
    p.add_argument("--list", action="store_true", help="列出各 harness 路径与能力")
    p.add_argument("--uninstall", action="store_true", help="删除已安装的 skill（并清理 MCP 配置）")
    p.add_argument("--plugin-out", default=None, help="生成 Claude Code/ZCode 插件包到该目录")
    args = p.parse_args(argv)

    if args.list:
        return list_targets()
    if args.check:
        return check(args.index_dir)
    if args.plugin_out:
        out = build_plugin(args.plugin_out, args.index_dir)
        _log(f"✓ 插件包已生成: {out}")
        _log("  Claude Code: /plugin marketplace add <该目录> → /plugin install "
             f"{SKILL_NAME}")
        return 0

    if args.targets:
        targets = [t.strip() for t in args.targets.split(",") if t.strip()]
    elif args.target:
        targets = list(TARGETS) if args.target == "all" else [args.target]
    else:
        p.error("需要 --target / --targets / --check / --list / --plugin-out 之一")
        return 2
    unknown = [t for t in targets if t not in TARGETS]
    if unknown:
        p.error(f"未知 target: {unknown}；可选 {list(TARGETS)}")
        return 2

    if args.uninstall:
        for target in targets:
            dest = _skill_dest(target, args.scope, args.dest)
            if dest.exists() or dest.is_symlink():
                if args.dry_run:
                    _log(f"[dry-run] 将删除 {dest}")
                else:
                    if dest.is_symlink() or dest.is_junction():
                        dest.unlink()
                    else:
                        shutil.rmtree(dest)
                    _log(f"✓ {target}: 已删除 {dest}")
            for line in unregister_mcp(target, args.scope, args.dry_run):
                _log(f"  MCP {line}")
        return 0

    return install(targets, args.scope, args.dest, args.link, args.mcp,
                   args.index_dir, args.dry_run)


if __name__ == "__main__":
    raise SystemExit(main())
