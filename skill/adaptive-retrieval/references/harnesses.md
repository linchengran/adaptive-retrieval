# 各 harness 接入参考

本 skill 走 **agentskills.io 的 SKILL.md 约定**（`<root>/<name>/SKILL.md` + YAML frontmatter
的 `name` / `description`），这是 Claude Code、Codex、ZCode、Pi、DeepSeek Harness 目前
共同接受的形态。检索能力本身通过三种入口暴露：CLI（所有 harness 都能跑）、
MCP server（常驻、推荐）、Python 库。

> 下表的路径在 Windows / macOS / Linux 上都对应 `<home>` 下的同名目录；
> 标注「已核实」的是在本机实际存在并检查过布局的。

## 1. Skill 目录

| harness | user scope | project scope | 状态 |
|---|---|---|---|
| Claude Code | `~/.claude/skills/<name>/` | `<repo>/.claude/skills/<name>/` | 已核实 |
| Codex | `~/.codex/skills/<name>/` | `<repo>/.codex/skills/<name>/` | 已核实 |
| ZCode | `~/.zcode/skills/<name>/` | `<repo>/.zcode/skills/<name>/` | 已核实 |
| Pi | `~/.pi/agent/skills/<name>/` | `<repo>/.pi/agent/skills/<name>/` | 已核实 |
| DeepSeek Harness | `~/.dsh/skills/<name>/` | `<repo>/.dsh/skills/<name>/` | 官方文档确认（扫描 rank 400 / 100） |
| 共享约定 | `~/.agents/skills/<name>/` | `<repo>/.agents/skills/<name>/` | DSH 会扫描（rank 500 / 200），其他 harness 也可能读 |

DeepSeek Harness 的扫描优先级（来自 `@deepseek-ai/dsh-skill-filesystem` 文档）：

```
100 <project>/.dsh/skills
200 <project>/.agents/skills
300 自定义 customSkillDirs
400 <dshHome>/skills        # 默认 ~/.dsh/skills
500 <agentsHome>/skills     # 默认 ~/.agents/skills
```

DSH 的 frontmatter 严格要求：`name` 必须 kebab-case、`description` 必填；
`disable-model-invocation` / `user-invocable` 若写成非法布尔值会导致整个 skill 被丢弃。

## 2. MCP server（消除冷启动）

CLI 每次调用都要新起解释器加载模型，实测冷启动 34-50s；MCP 形态只加载一次，
之后每次检索是 8-120ms。注册命令：

```bash
python scripts/install.py --target <claude-code|codex|dsh> --mcp
```

写到哪里、长什么样：

**Claude Code**（project scope 写 `<repo>/.mcp.json`，user scope 写 `~/.claude.json`）

```json
{"mcpServers": {"adaptive-retrieval": {
  "command": "<python>",
  "args": ["<...>/scripts/mcp_server.py", "--index-dir", "<...>/.retrieval_index", "--preload"],
  "cwd": "<...>/skill/adaptive-retrieval"}}}
```

**Codex**（`~/.codex/config.toml`）

```toml
[mcp_servers.adaptive-retrieval]
command = "<python>"
args = ["<...>/scripts/mcp_server.py", "--index-dir", "<...>/.retrieval_index", "--preload"]
cwd = "<...>/skill/adaptive-retrieval"
startup_timeout_sec = 60
tool_timeout_sec = 120
```

**DeepSeek Harness**（`~/.dsh/profiles/<profile>/cordis.patch.yml`，追加到顶层数组）

```yaml
- id: mcp-adaptive-retrieval
  name: "@deepseek-ai/dsh-mcp-client"
  config:
    serverName: retrieval
    transport: stdio
    command: "<python>"
    args: ["<...>/scripts/mcp_server.py", "--index-dir", "<...>/.retrieval_index", "--preload"]
    cwd: "<...>/skill/adaptive-retrieval"
    toolCallTimeoutMs: 120000
```

工具名会以 `mcp__retrieval__retrieval_query` 之类出现（harness 各自加命名空间）。

**ZCode**（`~/.zcode/cli/config.json`，注意是嵌套键 `mcp.servers`，与 Claude 的顶层
`mcpServers` 不同；workspace 级为 `<repo>/.zcode/config.json`，用户级优先）

```json
{"mcp": {"servers": {"adaptive-retrieval": {
  "type": "stdio",
  "command": "<python>",
  "args": ["<...>/scripts/mcp_server.py", "--index-dir", "<...>/.retrieval_index", "--preload"]}}}}
```

安装器已内置（`--target zcode --mcp` / `--target all --mcp` 一并写入）；写完需重启
ZCode 会话，server 会在会话启动时自动连接。

**Pi**：安装器不自动改它的 MCP 配置（格式随版本变化）。可以直接用 skill + CLI；
若要用常驻服务，把上面的 `command`/`args` 填进它的 MCP 配置即可，
`mcp_server.py` 是标准 stdio MCP server，不需要任何第三方依赖。

## 3. 插件包形态

```bash
python scripts/install.py --plugin-out ./dist/adaptive-retrieval
```

生成的结构：

```
dist/adaptive-retrieval/
  .claude-plugin/plugin.json     # name / version / description / author / keywords
  skills/adaptive-retrieval/     # 完整 skill bundle
  commands/retrieve.md           # /retrieve 斜杠命令（走 MCP 工具）
  .mcp.json                      # 自带 MCP server 声明
  README.md
```

- **Claude Code**：`/plugin marketplace add <该目录>` → `/plugin install adaptive-retrieval`；
  或把目录放进 `~/.claude/plugins/`。
- **ZCode**：其插件市场列表里能看到 `claude-plugins-official`，说明兼容 Claude Code 插件格式；
  同样可用 marketplace 方式安装。
- **Codex**：不使用插件包；用 skill 目录 + `~/.codex/config.toml` 的 MCP 段。

## 4. 验证

```bash
python scripts/install.py --check      # 依赖 / 各 harness 安装状态 / 索引 / MCP 自检
python scripts/mcp_server.py --selftest --index-dir <索引>
```

`--check` 会打印每个 harness 的安装状态（`ok` / `invalid`，invalid 表示 SKILL.md
frontmatter 缺 `name` 或 `description`，对应 harness 会直接忽略该 skill）。

## 5. 卸载

```bash
python scripts/install.py --target all --uninstall
```

只删除它自己安装的 `<root>/adaptive-retrieval/` 目录，以及带
`# >>> adaptive-retrieval` 标记的配置块；`~/.claude.json`、`config.toml`、
`cordis.patch.yml` 改动前都会留 `.bak.<timestamp>` 备份。
