# adaptive-retrieval

自适应检索 skill：一套代码，五种 harness 接入（Claude Code / Codex / ZCode / Pi / DeepSeek Harness）。

一句话：没有单一检索策略对所有查询最优——本方案按查询选检索深度（一致性路由）、
按语料选档位实现（语料感知），在"更快、更准、更方便"三个维度同时取得可测量的改进。

## 结构与分发边界

```
SKILL.md                 给模型看的说明（frontmatter: name/description/version/license）
scripts/retrieval_core.py  CLI + Python 库（index/retrieve/eval/tune/stats，全命令 --json）
scripts/mcp_server.py      常驻 MCP server（stdio，零第三方依赖），消除冷启动
scripts/install.py         多 harness 安装器 + 自检 + 卸载 + 插件包生成
scripts/selftest.py        端到端自测（无模型依赖模式：AR_EMBEDDER=stub）
references/harnesses.md    各 harness 的路径、MCP 配置模板、验证与卸载
benchmark/                 ★ 研究与复现管线（随 skill 一起分发）
  ├─ run_benchmark.py        M1 三路基线跑分（dense / bm25 / dense+rerank）
  ├─ preset_bench.py         策略预设四域对比 + 天花板分析
  ├─ ceiling_diag.py         诊断瓶颈在召回还是排序
  ├─ finetune_pipeline.py    领域微调重排器一键管线（构造→微调→评测→显著性）
  ├─ finetune_reranker.py / eval_finetuned_rerank.py / bench_train_speed.py
  ├─ vram_inference.py / vram_envelope.py   8GB 显存边界实测
  ├─ build_dureader.py / build_hf_domain.py / convert_to_skill_index.py
  ├─ data/testset_v2.json    wx 域 55 条查询评测集（唯一随包数据）
  └─ _paths.py               路径锚点（SKILL_DIR / REPO_ROOT，见下）
docs/                     ★ 研究文档（随 skill 分发）
  ├─ 优化-按影响程度排序.md    各优化项实测收益与显著性
  └─ 评审-自适应检索Skill的不足与改进.md
```

**随 skill 分发**：`scripts/` `references/` `benchmark/`(全部脚本 + wx 评测集)`docs/`。
**不随包分发**(体积/机器相关,按下面方式定位):

| 资源 | 体积 | 定位方式 |
|---|---|---|
| `models/`(bge 权重) | ~5.5GB | `AR_MODEL_DIR` → 向上搜索仓库根 `models/` → HuggingFace(国内设 `HF_ENDPOINT=https://hf-mirror.com`) |
| `data_raw/`(原始语料) | 34MB+ | 仓库根;独立安装时用 `benchmark/build_dureader.py` / `build_hf_domain.py` 自建 |
| `experiments/`(实验输出) | ~5.4GB | 仓库根;微调产物用 `finetune_pipeline.py` 重建 |

**路径锚定规则**(`benchmark/_paths.py`,移动/安装后无需改代码):
- `SKILL_DIR` = benchmark 的上一级(skill 根)——`scripts/` `references/` 都从这里找;
- `REPO_ROOT` = 从脚本位置向上搜索到的仓库根(判定标记:存在 `data_raw/` 或 `models/`);
  独立安装的 skill 没有仓库根时退回 SKILL_DIR,`data_raw/` `experiments/` 相关脚本
  会因数据缺失给出明确报错——这是预期行为,不是 bug。

## 三档位（语料感知）

```
长文档语料（平均粗块 ≥375 字符）：
fast  (~8ms)   细块召回 -> 父粗块返回（分层检索）
auto  (~92ms)  hier 与 BM25 top1 一致 -> fast；否则升级 best（一致性路由）
best  (~119ms) 分层召回 + bge-reranker 重排

短段落语料（<375 字符）：
fast  (~5ms)   纯 dense 向量（分层细切在该类语料上有害，-0.07）
auto  = best   dense + bge-reranker（路由在此类语料上精度不安全，-0.045，自动规避）
best  (~150ms) dense + bge-reranker
```

## 快速开始

```bash
pip install sentence-transformers rank_bm25 jieba numpy

python scripts/install.py --check                    # 先看环境
python scripts/install.py --target all               # 装进各 harness
python scripts/install.py --target all --mcp         # 再注册常驻 MCP（推荐）

python scripts/retrieval_core.py index <文档目录> --index-dir ./idx
python scripts/retrieval_core.py retrieve "问题" --index-dir ./idx --json
python scripts/retrieval_core.py eval --index-dir ./idx --testset 评测集.json --json
python scripts/retrieval_core.py tune --index-dir ./idx --testset 评测集.json --out tune.json
```

`tune` 的作用：候选池/分数融合/首跳 RRF 这些开关**收益完全取决于语料**
（实测同一改动 wx +0.109、cmedqa −0.150），所以不设全局默认，由你在自己的评测集上选；
它自带配对 bootstrap 与前半选择/后半复验，只在两边同向时才推荐。

不装模型也能试通链路：

```bash
AR_EMBEDDER=stub python scripts/selftest.py
```

详见 [SKILL.md](SKILL.md)；各 harness 差异见 [references/harnesses.md](references/harnesses.md)。

## 研究路线图

| 里程碑 | 内容 | 报告 |
|--------|------|------|
| M1 | 评测框架 + 三路基线（dense/bm25/dense+rerank） | [experiments/m1_baseline](../../experiments/m1_baseline/M1基线报告.md) |
| M2 | RRF 混合检索 + 一致性路由（margin 路由被否决的过程记录） | [experiments/m2_router](../../experiments/m2_router/M2报告.md) |
| M3 | hybrid+rerank、分层分块、学习型路由（负结果如实记录） | [experiments/m3](../../experiments/m3/M3报告.md) |
| M4 | hier+rerank 冠军组合、分层路由、Skill 封装 | [experiments/m4](../../experiments/m4/M4报告.md) |
| M5 | 跨域验证（DuReader）：找到方法适用边界 | [experiments/m5_dureader](../../experiments/m5_dureader/M5报告.md) |
| M6 | 语料感知档位配置：分类器 + 档位自动映射 | [experiments/m6_corpus_aware](../../experiments/m6_corpus_aware/M6报告.md) |
| M7 | 自带 eval CLI + 五域阈值校准 + 近重复语料修复 | [experiments/M7报告.md](../../experiments/M7报告.md) |

各里程碑报告在**完整仓库**的 `experiments/` 下(不随 skill 包分发,约 5.4GB 二进制产物;
`results.json` 与报告文本在仓库 git 里)。独立安装的 skill 想复现实验:clone 仓库,
或按上表"定位方式"自建数据后跑 `benchmark/` 对应脚本。
开题分析:[开题分析-自适应检索Skill.md](../../开题分析-自适应检索Skill.md)(同样在仓库根)。

## 方法要点

1. **一致性路由**：稠密与 BM25 的 top1 一致时排名大概率已稳定（train 集 90% vs 57% 先验），
   可跳过昂贵的重排；不一致时升级。余弦分数间隔无区分度（负结果），一致性信号泛化成功。
2. **分层分块**：句级细块定位、父粗块返回上下文。**仅长文档语料**——短语料上细切有害（-0.07）。
3. **语料感知档位**：索引时按平均块长分类（阈值 375，五域校准），档位实现随之切换。
4. **train/test 分离**：路由选型只用 train 集；负结果都有数据支撑。

## 已知边界

见 SKILL.md「已知边界」。要强调的三条：万级以上 chunk 未测；中文为主；索引是明文落盘。
