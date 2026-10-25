---
name: adaptive-retrieval
description: "Adaptive local retrieval (自适应文本检索)。对本地 .txt/.md 文档零配置建索引，提供 fast / auto / best 三档检索，档位实现按语料类型自动适配（长文档用分层定位加一致性路由，短段落用 dense 加重排）；比固定单路向量检索更准，比全量重排更快，并自带评测。Use for local document search, retrieval, RAG retrieval layer, or benchmarking retrieval strategies on your own data. 本地文档检索、知识库问答、RAG 检索组件、检索方案对比评测时使用。"
version: 1.0.0
license: MIT
metadata:
  tags: [retrieval, rag, search, rerank, hybrid-search, evaluation]
  harnesses: [claude-code, codex, zcode, pi, deepseek-harness]
  entrypoints: [cli, mcp, python]
---

# Adaptive Retrieval

自适应检索：档位实现按语料类型自动选择（语料感知），在"精度-延迟"前沿上给出三档方案，
并自带评测，让用户在自己的数据上验证换方案到底值不值。

## 何时用 / 何时别用

**用**：本地文档定位段落、搭建 RAG 检索层、对比检索策略、需要可复现的检索评测。
**别用**：纯网页实时搜索（用 web search）；只有 1-2 篇文档（直接 read 更快）；
十万级以上向量（本实现用 numpy 全量内积，先换 FAISS）；需要 PDF/docx 解析（先转文本）。

## 三种接入形态

| 形态 | 适合 | 入口 |
|------|------|------|
| **Skill**（本文件） | 任何支持 SKILL.md 的 harness，按说明调用 CLI | `scripts/retrieval_core.py` |
| **MCP server**（推荐） | 常驻工具，消除每次 30s+ 冷启动 | `scripts/mcp_server.py` |
| **Python 库** | 嵌进自己的服务 | `from retrieval_core import Retriever` |

一次性安装到各 harness（安装器已内置各平台真实路径）：

```bash
python scripts/install.py --check                      # 先看依赖/安装/索引/自检状态
python scripts/install.py --target all --dry-run       # 预览
python scripts/install.py --target all                 # 装 skill（安全，只写 skill 目录）
python scripts/install.py --target all --mcp           # 同时注册常驻 MCP（会改 harness 配置，带备份）
python scripts/install.py --plugin-out ./dist/adr      # 生成 Claude Code / ZCode 插件包
```

各 harness 的 skill 目录、MCP 配置位置与验证方式见 `references/harnesses.md`。

## 包结构与路径锚定

随 skill 分发的目录:`scripts/`(核心 CLI/MCP/安装器)、`references/`、
`benchmark/`(研究复现管线,25 个脚本 + wx 评测集 `benchmark/data/testset_v2.json`)、
`docs/`(优化排序与评审记录)。**不随包**:`models/`(bge 权重 ~5.5GB)、
`data_raw/`(原始语料)、`experiments/`(实验输出)——这三者属于完整仓库,
定位规则(`benchmark/_paths.py`):

- skill 自身资源 → 相对本包根(`SKILL_DIR`)
- 仓库级资源 → 从脚本位置**向上搜索**仓库根(判定标记:存在 `data_raw/` 或 `models/`,
  即 `REPO_ROOT`);独立安装的 skill 找不到时退回包根,数据缺失会明确报错
- 模型 → `AR_MODEL_DIR` → 仓库根 `models/` → HuggingFace(`HF_ENDPOINT` 走镜像)

因此本文中 `benchmark/xxx.py`、`docs/xxx.md` 的相对路径**从 skill 根目录**解析即可,
无需关心 skill 被安装到哪个 harness。

## 快速开始

```bash
# 1) 建索引（递归、增量；只编码新增/变更的文档）
python scripts/retrieval_core.py index <文档目录> --index-dir ./idx

# 2) 检索（--json 是给 agent/脚本用的稳定契约）
python scripts/retrieval_core.py retrieve "查询内容" --index-dir ./idx --profile auto --json

# 3) 用自己数据评测（自动附 dense 基线做对照）
python scripts/retrieval_core.py eval --index-dir ./idx --testset 评测集.json --json

# 4) 可选：在评测集上选出适合这份语料的策略（带显著性 + 留出验证）
python scripts/retrieval_core.py tune --index-dir ./idx --testset 评测集.json --out tune.json
python scripts/retrieval_core.py retrieve "..." --index-dir ./idx --config tune.json
```

`retrieve --json` 的返回（字段稳定，可直接解析）：

```json
{"query":"...","corpus":{"corpus_type":"long","n_chunks":28,"avg_chunk_chars":432.1},
 "route":{"profile":"auto","route":"hier+rerank","agreement":false,
          "timings":{"total_ms":31.2,"model_load_ms":0.0}},
 "results":[{"rank":1,"chunk_id":"01_faq.txt::8","source":"01_faq.txt",
             "score":0.918,"stage":"rerank","text":"..."}]}
```

- `route.route` 是**实际**走的路径，用于解释"为什么这次快/慢"
- `timings.model_load_ms` 是首次调用的冷启动耗时，稳态延迟看 `total_ms - model_load_ms`
- 结果带 `score`，可配合 `--min-score` 过滤噪声；`fast` 档在长文档语料上召回偏弱，
  对 agent 默认用 `auto`

> **不要手调超参**：候选池、分数融合、首跳 RRF 这些开关的收益高度依赖语料——
> 实测同一改动在 wx 上 +0.109、在 cmedqa 上 −0.150。需要就用 `tune`，它自带
> 配对 bootstrap 与前半选择/后半复验，只在两边同向时才推荐（详见
> `docs/优化-按影响程度排序.md`）。

## 三档位与语料自适应

| 档位 | 长文档语料（平均粗块 ≥375 字符） | 短段落语料（<375 字符） |
|------|--------------------------------|------------------------|
| `fast` | 分层细块定位 → 父粗块返回 | 纯 dense 向量 |
| `auto`（默认） | 一致性路由：hier 与 BM25 top1 一致走 fast，否则升级 best | = best |
| `best` | 细块召回 + 重排序 | dense + 重排序 |
| `dense` | 单路向量（**仅作评测基线**，不属于三档契约） | 同左 |

语料类型在 `index` 时按平均 chunk 长度自动判定并写入 `meta.json`，三档 API 不变、行为自动适配。

## 答案级评测（比 chunk 级更接近"能不能答对"）

chunk 级 hit@k 只说明"块找得准"，回答不了"最终答得对"。两条补充通道：

```bash
# a) 评测集每条可选填 "answers": ["九月一日"]，eval 会额外输出 answer_recall@k
#    （检索到的上下文里是否真的含答案原文）
python scripts/retrieval_core.py eval --index-dir ./idx --testset ts.json --json

# b) 生成待打分表（含各档位上下文），由 agent/人打 answer_correct + faithful 0/1
python scripts/retrieval_core.py eval --index-dir ./idx --testset ts.json --judge-sheet judge.json
#    填完后聚合
python scripts/retrieval_core.py eval --judge-results judge.json --json
```

`--judge-sheet` 不需要额外 API：这个 skill 本来就跑在 agent 里，让 agent 读上下文打分即可。

## 策略选购：`tune`（不要手调超参）

候选池、召回分与重排分融合、首跳 BM25-RRF 这类开关**收益高度依赖语料**：
实测同一改动在 wx 上 +0.109、在 cmedqa 上 −0.150；融合在 t2 上 +0.020、在 wx 上 −0.127。
所以它们默认全关，要用就用 `tune` 在你自己的评测集上选：

```bash
python scripts/retrieval_core.py tune --index-dir ./idx --testset ts.json --out tune.json
python scripts/retrieval_core.py retrieve "..." --index-dir ./idx --config tune.json
```

`tune` 自带两道护栏：**配对 bootstrap**（2000 次，给差值的 95% 区间）与
**前半选择 / 后半复验**（两边同向才推荐，否则退回 `base`）。
实测它拦下过一个假提升：t2 全量 200 条上 fusion0.7 = 0.985 vs base 0.965，
但只用前 100 条选择时两者打平，`tune` 因此推荐保持 `base`。

可用预设：`base` / `fusion0.5` / `fusion0.7` / `fusion0.85` / `cand_rrf` /
`fast_rrf` / `multiquery` / `mq_rrf` / `rerank_ens`。
后四个在本文四域上的实测均为 0 或负收益（`rerank_ens` 在 cmedqa 上 −0.115 且延迟 ×3），
保留它们只为让你在新语料上有一条已验证的低成本试错路径。

## 先判断该优化哪一环（很重要）

在调任何参数之前，先算两个数——**这一步能省掉大部分无效工作**：

| 指标 | 含义 | 怎么算 |
|---|---|---|
| 第一跳 top5 命中率 | 召回够不够 | `fast` 档的 hit@5 |
| 候选池 recall@10 | 重排的输入是否已含正例 | 见 `benchmark/ceiling_diag.py` |
| hit@1 | 重排排得准不准 | `auto` 档 hit@1 |

三者关系决定方向：
- **候选池 ≈ 1.0 而 hit@1 低** → 召回已饱和，加 multi-query/扩池无用，去动重排器；
- **候选池明显低于第一跳** → 索引/切块有问题（或语料分类判错），先 `eval` 回归；
- **第一跳就低** → 召回不足，此时扩池、混合召回、查询改写才有意义。

本仓库四个域的实测：DuReader 候选池 **1.000** 而 hit@1 只有 0.615——
典型的"排序瓶颈"，也是为什么本仓库的 multi-query 与重排集成都测不出收益。

## 排序瓶颈时：领域微调重排器（唯一验证有效的精度提升）

如果上面的诊断显示是排序瓶颈，**不要继续改检索策略**，去微调重排器或换更强的模型。
本仓库已把这条路径做成可复现管线，并有实测结果：

| 域 | 候选池 recall@10 | 微调前 | 微调后 | Δ | P(≤0) |
|---|---|---|---|---|---|
| DuReader | 1.000 | 0.615 | **0.675** | **+0.060** | 0.025 |
| cmedqa | 0.900 | 0.600 | 0.595 | −0.005 | 0.638 |
| T2Retrieval | 0.995 | 0.965 | 0.955 | −0.010 | 0.783 |

提升只出现在训练域（其余域区间跨 0），训练/测试 query 零重叠，所以是**领域适配**而非泄漏。

```bash
# 一键：构造隔离训练集 → 微调 → 评测 → 显著性检验
python benchmark/finetune_pipeline.py --step all --domain <你的域>

# 挂上微调器（不改代码）
AR_RERANKER_DIR="experiments/_rerank_train/model" python scripts/retrieval_core.py \
  retrieve "查询" --index-dir ./idx --profile best
```

**8GB 显卡必须 `--batch 8 --max-length 256`**：实测 batch 16 的显存峰值 8.75GB 会触发
显存换页，每步从 0.16s 恶化到 10.8s。先跑 `benchmark/bench_train_speed.py` 标定。

## 显存边界：8GB 能做什么、不能做什么

**这一节全部是实测的（`benchmark/vram_inference.py` / `vram_envelope.py`），不是估算。**

**日常检索/推理：完全没问题。**

| 任务 | 参数 | 推理峰值显存 | 8GB 可行 |
|---|---|---|---|
| bge-small-zh 编码 | 24M | <0.3GB | ✅ |
| bge-reranker-base 重排（默认） | 278M | **1.26GB** | ✅ |
| 微调后的 bge-reranker-base | 278M | **1.26GB** | ✅ |
| bge-reranker-v2-m3 | 568M | **2.47GB** | ✅ |
| bge-reranker-v2-gemma | ~9B | ~18GB | ❌ |

**换更强的 v2-m3 完全可行**（推理 2.47GB），代价是延迟变成 base 的 2.6–3.2 倍。

**微调：能做的和不能做的**

| 配置 | 显存峰值 | 8GB 可行 |
|---|---|---|
| base 微调（batch 8 / maxlen 256） | 5.59GB | ✅ |
| base 微调（batch 16 / maxlen 256） | 5.97GB 但速度掉 7 倍（换页临界） | ⚠️ |
| base 微调（batch 16 / maxlen 512） | 8.53GB > 8.52GB 物理显存 | ❌ |
| **v2-m3 全参数微调** | **11.37GB**（与 batch 无关，固定开销） | ❌ |
| **v2-m3 冻结 20 层微调** | **2.71GB** | ✅ |
| 任意模型 maxlen 768 | 崩溃（device-side assert） | ❌ |

**关键结论：v2-m3 全参数微调在 8GB 上不可能，但冻结大部分层可以**——
冻结 20 层后峰值从 11.37GB 降到 2.71GB，因为优化器状态只随可训练参数增长。

```bash
# 8GB 上微调 v2-m3 的正确姿势
python benchmark/finetune_reranker.py --base-model models/bge-reranker-v2-m3 \
  --freeze-layers 12 --batch 4 --max-length 256 --lr 1e-5 --epochs 1
```

## 微调失败防护：判别力探针（必读）

**微调并不总是变好。** 实测一次 v2-m3 微调（冻结 20 层、lr=5e-5）把模型调坏了：

| 指标 | 值 |
|---|---|
| train loss | 0.5533 → 0.5096（正常下降） |
| dev pair accuracy | 0.800（看着正常） |
| 正例/负例得分 | **0.1007 / 0.1007（无判别力）** |
| 测试集 hit@1 | **0.105**（从 0.645 崩塌） |

**从训练指标完全看不出来。** 所以 `finetune_reranker.py` 内置了判别力探针：
每个 epoch 用一对固定正负例打分，`gap ≤ 0.02` 即拒绝保存该权重并报警。

即便 base 的微调（gap 0.9504 → 0.2661）也是过拟合的——只是还没垮。
**做任何重排器微调后，都要检查正负例得分间隔，不要只看 loss。**

## 从已有 RAG 语料接入（避免指标不可比）

已有系统有自己的切块，比较指标必须**同 chunk 口径**：
用 `benchmark/convert_to_skill_index.py` 把现有 chunk 与向量转换成 skill 索引，
而不是对原始文档重新切块。评测集里的 `relevant` 必须是该索引内真实存在的 `chunk_id`
（`eval` 会校验并报出不存在的 id）。

## 实测性能（chunk 级标注，与 benchmark 各臂同口径）

**wx 知识库（长文档，55 条中文查询）**

| 方案 | hit@1 | hit@5 | ndcg@5 | 稳态延迟 |
|------|-------|-------|--------|---------|
| fast（分层） | 0.491 | 0.855 | 0.684 | ~8ms |
| auto（路由） | 0.709 | 0.836 | 0.768 | ~92ms |
| best（分层+重排） | 0.709 | 0.927 | 0.805 | ~119ms |
| 基线：纯向量 | 0.418 | 0.618 | 0.526 | ~13ms |
| 基线：向量+重排 | 0.655 | 0.782 | 0.705 | ~117ms |

**DuReader（短段落，200 条真实搜索查询）**

| 方案 | hit@1 | hit@5 | ndcg@5 |
|------|-------|-------|--------|
| fast（dense） | 0.485 | 0.980 | 0.731 |
| auto（=best） | 0.615 | 0.980 | 0.798 |
| 基线：纯向量 | 0.485 | 0.980 | 0.731 |

**M7 五域阈值校准（各 200 条）**

| 语料域（平均粗块 / 分类） | fast | auto | best |
|--------------------------|------|------|------|
| cmrc2018（368.8 / short） | 0.975 | 0.985 | **0.985** |
| T2Retrieval（400.8 / long） | 0.945 | 0.970 | **0.965** |
| Cmedqa（169.0 / short） | 0.565 | 0.600 | **0.600** |

口径说明：表中延迟是**进程内稳态**（模型已加载）；CLI 冷启动约 34-50s（实测），
这正是推荐 MCP 形态的原因。duReader/cmrc2018 的 hit@5 已饱和（0.98），
这几个域上 hit@5 无法区分方案，请看 hit@1 / ndcg。

## 失败模式与排查

| 现象 | 原因 | 处理 |
|------|------|------|
| `索引不存在` | 未建索引 / 路径不对 | 先 `index`，或用 `--index-dir` 显式指定 |
| `embedder 维度 N 与索引维度 M 不一致` | 换了模型但没重建索引 | `index --force` |
| `该索引为短段落语料构建（未含细块）` | 短语料没建细块 | 用 `auto`/`best`；`fast` 会自动走 dense |
| 检索很慢（数十秒） | 每次新进程都要加载模型 | 用 MCP server（常驻）或 Python 长驻调用 |
| 结果中文乱码 | Windows 控制台 GBK | CLI 已强制 UTF-8；管道读取请按 UTF-8 解码 |
| 重排器加载失败 | 缺 `bge-reranker-base` | 自动降级为不重排并在 `route.rerank` 标注，不中断 |
| 中文文档里夹英文 | 分词/指令前缀不适配 | 英文占比 >80% 时自动切换英文分词；混合语料建议 eval 验证 |

## 已知边界

- 语料分类阈值 375 字符由五域校准（wx 428 / DuReader 315 / cmrc2018 368.8 / T2Retrieval 400.8 / Cmedqa 169）；
  误判代价不对称——短判长会启用有害的细切，压线语料宁可判短。接入新语料先用 `eval` 回归。
- 一致性路由只在长文档语料启用（train 集 90% fast-path 正确率 vs 57% 先验）；短段落语料上会掉精度，已自动规避。
- 规模：百/千 chunk 级验证；万级以上未测（`dense_top` 已用 argpartition，但仍是全量内积）。
- 中文为主；英文可用、分词与指令已切换，但未系统调优。
- 索引是明文 JSON + npy，落盘目录请勿提交到公开仓库（含原文）。
- **检索自评门控（低置信→自动升级重排）已验证不可行**：在四个域上留一域验证，
  dureader/cmedqa 的 AUC 只有 0.56–0.61，任何阈值都不改变决策；cmrc 上第一跳已 0.975 无空间。
  更深层的原因是天花板本来就低（见下条），故不实现。详见 `docs/优化-按影响程度排序.md`。
- **不存在跨域通用的检索超参**：融合/首跳 RRF 在长文档语料间结论相反（+0.020 vs −0.127）。
  需要调整请走 `tune`，不要直接改默认值。
- **瓶颈在排序，不在召回**（最有用的一条使用前提）：候选池 recall@10 实测已达 0.98–1.000
  （DuReader 精确 1.000）。含义是：
  - 继续加召回手段（multi-query / HyDE / 扩池 / 混合召回）**理论收益上限 ≤0.02**，本仓库实测为 0；
  - 如果你的语料也是 `pool_recall@10 ≈ 1` 而 `hit@1` 只有 0.6，问题在重排器，
    应去微调/更换 reranker，而不是继续调检索策略；
  - 判断方法：先算 `第一跳(top5)` 与 `候选池(top10)` 两个命中率，再看 hit@1 差多少。

## 依赖

```bash
pip install sentence-transformers rank_bm25 jieba numpy
```

`faiss-cpu` 仅 benchmark 侧使用，skill 核心不需要。模型默认找
`<repo>/models/bge-small-zh-v1.5` 与 `models/bge-reranker-base`，
可用 `AR_MODEL_DIR` 覆盖，找不到则回退 HuggingFace（国内可设 `HF_ENDPOINT=https://hf-mirror.com`）。
自测可设 `AR_EMBEDDER=stub`，用哈希向量跑通全流程而不加载任何模型。
