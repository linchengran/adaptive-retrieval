# 评审：adaptive-retrieval 的不足与改进

> 评审对象：`skill/adaptive-retrieval` + `benchmark/` + `experiments/`（M1–M7）
> 评审方式：读代码与报告 + 在本机实跑（Windows 11 / Python 3.13 / CUDA 可用）
> 结论口径：**"不足"指会影响真实使用或结论可信度的问题**，每条给证据与状态。

## 0. 一句话结论

方法本身（一致性路由 + 语料感知档位 + 分层分块）是扎实的，实验记录的负结果也诚实；
**主要短板不在算法，而在"交付形态"与"可复现性"**：

- 作为 CLI 交付时，单次调用冷启动 **34–50 秒**（实测），而文档宣传的是 **8–119ms**（进程内稳态），
  二者差 3 个数量级——这决定了它在 agent 场景里"能不能用"，而不是"好不好用"；
- 没有任何机器可读输出、没有接入说明、没有安装器，五个 harness 全靠用户手摆；
- 评测与索引缺少自描述，换模型/换语料时容易静默得到错误结论。

下面按 P0 / P1 / P2 分级，`已修` 表示本次已落到代码或文档。

---

## 1. P0：会直接让 skill 不可用

### 1.1 冷启动 34–50s，与宣传延迟差 3 个数量级 `已修（MCP 形态）`

实测（同机，反复三次）：

| 场景 | 实测 |
|------|------|
| `retrieve --profile fast`（单次 CLI） | **41.9s** |
| `retrieve --profile best`（单次 CLI） | **41.9s** |
| `retrieve --profile auto`（单次 CLI） | 42.0s / 34.2s |
| `import numpy` | 0.3s |
| `import torch` | 2.8s |
| `import sentence_transformers` | **22–25s** |
| 加载 bge-small-zh（含 import） | 30s |
| 进程内稳态检索（报告口径） | 8–119ms |

原因：CLI 每次新起解释器，`sentence-transformers` + torch 的导入与模型加载无法摊销。
影响：agent 每次检索要等 30s+，多轮问答直接不可用；而 SKILL.md/README 只写了 8ms，
使用者会认为"实测与宣传不符"。

处置：新增 `scripts/mcp_server.py`（stdio JSON-RPC，零第三方依赖），模型只加载一次，
`--preload` 后台预热；CLI 的 `--json` 里新增 `timings.model_load_ms`，
把"冷启动"和"稳态"分开报，文档也不再混用两个口径。

### 1.2 没有机器可读输出 `已修`

原 `retrieve` 只打印人读文本（`1. [chunk] source=...`），agent 无法稳定解析，只能正则。
且 `fast/auto/best` 三个 profile 的差异只体现在文本里，无法程序化判断"这次走的是哪条路"。

处置：全命令 `--json`（`index/retrieve/eval/stats`），返回固定 schema：
`results[].{rank,chunk_id,source,score,stage}` + `route.{route,agreement,timings}` + `corpus.*`。

### 1.3 Windows 控制台中文乱码 `已修`

默认 GBK 控制台打印中文全部变 `����`（已复现），agent 读到的就是乱码。
处置：CLI / 安装器 / MCP 入口统一 `reconfigure(encoding="utf-8")`；
另修了读文档时不剥 UTF-8 BOM 的问题（PowerShell 写的 `.md` 极常见，BOM 会进 chunk 正文）。

### 1.4 没有接入任何 harness 的能力 `已修`

改前：只有 `SKILL.md` + 两个脚本，用户要把目录复制到哪、MCP 怎么配全靠猜。
处置：`scripts/install.py`（六个 target：claude-code / codex / zcode / pi / dsh / agents，
user+project 两种 scope，copy/link 两种模式，`--check`/`--uninstall`/`--plugin-out`），
附 `references/harnesses.md` 写清各平台真实路径与配置模板。

---

## 2. P1：会让结果不可信或维护困难

### 2.1 索引无自描述、无校验 `已修`

改前 `meta.json` 只有 `{n_chunks, avg_chunk_chars, corpus_type}`：不记录 embedder、维度、
分块参数、schema 版本。换了模型但没重建索引时，只会得到"看起来正常但分数全错"的结果。
处置：`schema_version=2`，记录 `embedder_id/dim/chunk 参数/逐文档 sha1`；
`Retriever` 加载时校验行数与维度，模型维度不符直接报"必须重建索引"；
旧版（schema 1）索引仍可读，`describe()` 会标出来。

### 2.2 增量索引缺失 `已修`

改前改一个文档要全量重建（含 25s 导入 + 全部编码）。千级语料上是分钟级成本。
处置：按文档 sha1 复用旧向量（粗块与细块都复用），`reused_docs/encoded_docs` 写进 meta 与输出；
实测"未变更 → 0 篇编码，改一篇 → 只编码 1 篇"。

### 2.3 只扫顶层目录、chunk_id 会撞 `已修`

`Path(docs_dir).glob("*.txt")` 只扫一层，子目录文档静默丢失；`chunk_id` 用 `fp.name`，
不同子目录同名文件会撞 id（仓库里 `results_prefix_collision.json` 正是这类事故的痕迹）。
处置：递归发现（可关）、跳过隐藏目录与索引自身目录、`chunk_id = "<相对路径>::<序号>"`、
`eval` 校验评测集里的 id 必须存在于索引（写错 id 会静默 0 分，是最常见的评测坑）。

### 2.4 结果没有分数，无法过滤/解释 `已修`

改前三档都只返回 chunk dict。处置：每档都带 `score` + `stage`，CLI 加 `--min-score`；
`auto` 的 `route.agreement` 直接暴露路由依据（为什么这次快/慢）。

### 2.5 `hier_top` 可能返回不足 k 条 `已修`

细块 oversample=3 固定：若 top-3k 细块全落在同一父块，唯一父块数 < k，返回条数会缩水。
处置：自适应倍增扫描（3k → 6k → …）直到凑够 k 或扫完，并加了 `k > 语料规模` 的截断保护。

### 2.6 重排器不可用会整体崩 `已修`

改前 `_get_reranker` 抛异常即整个检索失败。处置：降级为不重排，`route.rerank`
标注 `disabled_or_unavailable`，并在 `degraded` 字段说明原因。

### 2.7 英文语料仍拼中文检索指令 `已修`

BGE 中文模型的指令前缀被硬编码，英文语料也拼中文前缀（掉点），英文分词也没处理。
处置：按语料 ASCII 占比判定，英文语料不加中文指令、BM25 走英文正则分词。

### 2.8 评测太薄，无法自证 `已修`

改前 eval 只有 `hit@k` + `MRR`，没有 nDCG/Recall，没有基线对照，用户无法回答
"升级档位到底赚了多少"——而这正是这个 skill 的差异化卖点。
处置：`hit/recall/ndcg@k + MRR`，**自动跑 dense 基线**（`--no-baseline` 可关），
`--template` 生成待标注骨架（随机抽 chunk、预填 relevant），`--out` 落 JSON，`--per-query` 出明细。

### 2.9 SKILL.md 承诺了不存在的东西 `已修`

改前文档说"用 `benchmark/convert_to_skill_index.py` 统一口径"，但该脚本不在 skill 分发包内；
独立安装 skill 的用户按文档操作必然失败。
处置：文档明确其在 benchmark/ 下（研究仓内），并说明"同口径"为什么必须这么做；
`eval` 的未知 id 报错直接给出这条提示。

### 2.10 frontmatter 只有中文 description、无版本/许可 `已修`

`description` 决定 skill 能否被触发，纯中文在英文 harness 上匹配差；
且没有 `version`，安装器无法记录装的是哪一版。
处置：中英双语 description + `version/license/metadata(tags, harnesses, entrypoints)`。

### 2.11 我用 YAML 解析器一验，frontmatter 是**非法 YAML** `已修`

这一条是本次评审里最危险、也最容易漏掉的：`description` 里写了
`(long docs: hierarchical locate ...)`，未加引号的标量中出现 **`: `** 会让 YAML 解析器
直接抛 `ScannerError: mapping values are not allowed here`。
后果不是"描述不好看"，而是**严格解析 frontmatter 的 harness 会整个拒绝加载这个 skill**
（DSH 的处理是"随警告被跳过"，即 skill 静默消失）。

处置：description 改为双引号包裹并去掉裸冒号；用 PyYAML 实测通过：

```
name: adaptive-retrieval   desc_len: 307   keys: [name, description, version, license, metadata]
```

建议把这条检查固化进发布流程（安装器的 `--check` 现在已用 PyYAML 真解析一次，
失败会报 `invalid-yaml`）。

### 2.12 模型定位硬编码、同进程内反复加载 `已修`

- `_load_embedder` 用 `Path(__file__).resolve().parents[3] / "models"` 定位模型：
  目录层级一变（pip 安装、复制到别处、打包进插件）就静默回退到在线 HuggingFace，
  离线环境直接失败。处置：`AR_MODEL_DIR` 环境变量 → 向上逐级查找 `models/` → HF 回退，
  每步可控。
- 同进程内 `build_index` 与 `Retriever` 各自 new 一个模型：自测里 3 次建索引 =
  3 次加载（实测多花约 90s）。处置：模块级模型缓存（embedder / reranker 分别按
  模型名 + device 缓存），真实模型自测从 3 分钟量级降到 102s。

### 2.13 文档里的"延迟"没有区分冷/热 `已修`

同一张表里混着 8ms（进程内）与用户实际感受的 40s（CLI），是"看起来很强、用起来很慢"
的典型来源。处置：SKILL.md 明写"表中为进程内稳态，CLI 冷启动 34-50s"，
`--json` 里 `timings.model_load_ms` 与 `total_ms` 分开报。

---

## 3. P2：工程与研究层面的欠账

| # | 问题 | 证据 | 状态 |
|---|------|------|------|
| 3.1 | 零测试、零 CI | 仓内无 tests/、无 CI 配置；`errors.log` 记录 `load_faiss_index(index_dir / "index.npy")` 因 str/Path 混用崩溃（跑一次就炸） | `已修`：`scripts/selftest.py`（stub 模式 31 项断言，秒级；真实模型模式额外验证语义链路） |
| 3.2 | benchmark 硬编码绝对路径 | `benchmark/corpus.py` 写死 `D:\claude code use\rag-research-project\...`，他人仓库无法复现 | `待办`：改 env/相对路径 + 记录 `chunks.pkl` 生成命令 |
| 3.3 | 仓库卫生 | `models/`（数百 MB 权重）、`data_raw/`、`experiments/*.npy/*.pkl` 全进版本库；无 `.gitignore`、无根 README、无 LICENSE、无 CHANGELOG；README 提到的 `tools/` 已不存在 | `部分修`：加 `.gitignore` 与根 README；LICENSE 需作者确认（第三方模型是 MIT/Apache，代码许可要作者定） |
| 3.4 | 指标饱和、无统计效力 | DuReader 三臂 `hit@5` 全 0.980，无法区分；单域 200 条，无置信区间/显著性检验；长文档只有 wx 一个域支撑"分层有益" | `待办`：报告里补 bootstrap CI，饱和指标改为主报 hit@1/nDCG@5 |
| 3.5 | 阈值 375 是单特征硬阈值 | 5 个域校准、靠平均块长；文档自己说"误判代价不对称"，但没有边界兜底 | `已缓解`（`degraded`/`corpus_type` 可见 + eval 回归建议）；`待办`：375±50 区间内 `auto` 直接走 best |
| 3.6 | 万级规模未测 | 只用 numpy 全量内积；`dense_top` 原为全排序 O(N log N) | `已修`（>k 时改 `argpartition`）；`待办`：10 万级切 FAISS/HNSW |
| 3.7 | 交付形态与开题不符 | 开题承诺 "CLI / Python 库 / HTTP API / Skill"，实际无 HTTP API | `部分修`（MCP stdio 已覆盖进程内常驻需求）；HTTP API 仍待办 |
| 3.8 | 语料类型只有两档 | 长/短二分，中间态靠"宁可判短"；中英混排、代码/表格语料未验证 | `待办` |
| 3.9 | 无隐私/安全说明 | 索引是明文 JSON + npy（含原文），易被误提交到公开仓库 | `已修`：SKILL.md「已知边界」明确提示；`.gitignore` 忽略索引目录 |
| 3.10 | 无 PDF/docx/HTML 摄入 | 只吃 `.txt/.md`；真实知识库大量是 PDF | `待办`：接 office/pdf skill 或先转文本 |

---

## 4. 已落地的改动清单（本次）

```
skill/adaptive-retrieval/
  SKILL.md                      重写：双语 description、三入口、失败模式表、冷/热延迟口径、边界
  README.md                     重写：结构、快速开始、install 一行命令
  references/harnesses.md       新增：五 harness 路径矩阵 + MCP 配置模板 + 验证/卸载
  scripts/retrieval_core.py     重写：JSON 契约、UTF-8、分数、递归、增量、schema 校验、
                                自适应 oversample、重排降级、英文分词/指令、eval 增强、模板生成
  scripts/mcp_server.py         新增：常驻 MCP server（4 个工具，--selftest/--preload/--tools）
  scripts/install.py            新增：六 target 安装器（check/uninstall/plugin-out/dry-run）
  scripts/selftest.py           新增：31 项端到端断言（stub 模式无模型依赖）
docs/评审-自适应检索Skill的不足与改进.md   本文件
benchmark/convert_to_skill_index.py  更新：写出 schema 2 meta（embedder/维度/分块参数），
                                使转换来的索引也能做维度和来源校验
.gitignore                      新增：模型、数据、索引、实验产物
README.md（根）                 新增：项目总览与评审入口
```

验证方式（都实跑过）：

| 验证 | 结果 |
|------|------|
| `AR_EMBEDDER=stub python scripts/selftest.py` | 31/31 通过（秒级） |
| `python scripts/selftest.py`（真实 bge + reranker） | 31/31 通过，102s |
| `install.py --target all --link` | 六个 target 全部 junction 成功，`--check` 全 ok |
| `install.py --check` | 依赖 ✓ / skill frontmatter ✓ / 安装状态 ✓ / 索引 ✓ / MCP 自检 ✓ |
| MCP stdio 管道（initialize + tools/call + 未知方法） | 正确返回结果与 `-32601` |
| `--mcp` 写入 codex TOML / dsh YAML / claude JSON | 结构合法，`--uninstall` 精确还原原文 |
| DeepSeek Harness 实机 | 装完即出现在会话 skill 目录，`skill` 工具可正常加载 |
| 旧索引（schema 1，无 meta.json） | 可读，`describe()` 标注 schema 1 |

## 5. 建议的下一步（按性价比排序）

1. **补 benchmark 的路径可配置化 + 一条复现命令**（3.2）——现在是"报告很完整，但别人跑不起来"。
2. **报告加 bootstrap 置信区间**，并把主指标从饱和的 hit@5 换成 hit@1 / nDCG@5（3.4）。
3. **375±50 过渡带兜底**：边界语料上 `auto` 直接走 `best`，用 1 次实验换掉"压线靠猜"（3.5）。
4. **万级压测 + FAISS 后端**（3.6）。
5. **`--template` 接 LLM 生成查询**，把"自证"闭环从"人工写 50 条"降到"跑一次"。
6. 明确 LICENSE（代码与模型分开写）与发布清单（哪些资产不进分发）。

## 6. 我没有改的部分（避免误解）

- 未更改任何实验数字与报告结论；M1–M7 报告原样保留。
- 未删除 `models/`、`data_raw/`、`experiments/` 里的二进制资产（会影响你的复现），只加了 `.gitignore`。
- 未动 `benchmark/` 的研究代码（包括 3.2 的硬编码路径），怕破坏已有报告的可追溯性。
- 未替你决定许可证与署名。
