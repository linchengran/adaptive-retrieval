# adaptive-retrieval · 自适应检索 Skill

**Adaptive local-retrieval Skill**: per-query depth routing + per-corpus tier adaptation, packaged as an installable Agent Skill for Claude Code / Codex / ZCode / Pi / DeepSeek Harness.

> 没有单一检索策略对所有查询最优——本方案**按查询选检索深度**（一致性路由）、**按语料选档位实现**（语料感知），在「更快、更准、更方便」三个维度同时取得可测量的改进，并封装成一条命令可安装的 Skill。
> 全部方法、评测过程与结论记录见 [`experiments/`](experiments/)（M1–M7 阶段报告，负结果一并保留）。

## 结果速览（五域，hit@1）

五域同方法学构建：每域 200 条真实标注查询，语料 = 正例文档 + 3 条 BM25 难负例，512 字符窗口切块；口径与过程见 [experiments/M7报告.md](experiments/M7报告.md)。

| 域 | 平均粗块字符 | hier − dense | routed − 最强臂 | routed fast 占比 | routed p50 | 最强慢路径 p50 |
|----|------------|--------------|----------------|-----------------|-----------|---------------|
| Cmedqa（短） | 169.0 | −0.045（有害） | −0.010 | 24% | 101 ms | 128 ms |
| DuReader（短） | 315.3 | −0.070（有害） | −0.045 | 48.5% | — | — |
| cmrc2018（过渡带） | 368.8 | 0.000 | ≈0（零损失） | 82% | 12 ms | — |
| T2Retrieval（长） | 400.8 | +0.005（有益） | −0.005 | 37.5% | **318 ms** | 1268 ms |
| wx（长） | 428 | +0.076（有益） | ≈0（0.709 = 最优） | 29% | — | — |

三条主要结论：

1. **语料分类阈值 375 通过五域校准，无需重调**：平均粗块字符与「分层细切是否有益」单调一致（169/315 有害 → 368.8 打平 → 400.8/428 有益），且误分代价不对称（短判长掉精度，长判短只是回退 dense+rerank），压线语料归 short 是安全侧；
2. **语料感知的档位实现有真实收益**：长语料上 hier 系支配 hybrid 系——T2 上 routed-hier 以 **318 ms 拿到 best 档的 0.965**（独占 hier+rerank 要 1268 ms，hybrid+rerank 3490 ms）；wx 上 auto 档 0.709 = best 档 0.709；
3. **一致性路由零损失是有条件的**：精度下限由慢路径决定，损失 ≈ 快路径占比 × 快慢路径命中率差（M7 §6 的机制分析；cmrc 快路径 82% 却零损失即反向印证）。

> **诚实声明**：cmrc2018 语料首轮被近重复变体污染（dense 0.110 vs 0.975 的修复前后数字不可横比），排查与修复见 M7 §7；修复前 results 存档于 `results_prefix_collision.json`。全部效度威胁见 M7 §8。

## 三分钟上手

```bash
pip install sentence-transformers rank_bm25 jieba numpy

cd skill/adaptive-retrieval
python scripts/install.py --check                 # 依赖 / 安装状态 / 索引 / MCP 自检
python scripts/install.py --target all            # 装进本机所有 harness 的 skills 目录
python scripts/install.py --target all --mcp      # 再注册常驻 MCP（推荐，消掉 30s+ 冷启动）

AR_EMBEDDER=stub python scripts/selftest.py       # 不下载模型，31 项契约自测

python scripts/retrieval_core.py index <你的文档目录> --index-dir ./idx
python scripts/retrieval_core.py retrieve "问题" --index-dir ./idx --json
python scripts/retrieval_core.py eval --index-dir ./idx --testset 评测集.json   # 自带离线评测
```

检索档位：`fast` 最快、`best` 最准、`auto` 默认（长文档语料按一致性路由决定是否升级重排）；另有 `dense` 无重排对照基线。

## 方法要点

1. **一致性路由**（长文档语料）：稠密检索与 BM25 的 top1 一致时，排名大概率已稳定
   （train 集 90% 正确率 vs 57% 先验），可跳过昂贵的重排序；不一致则升级。
   余弦分数间隔被证明无区分度（负结果），一致性信号泛化成功。
2. **分层分块**：句级细块定位、父粗块返回上下文。**仅长文档语料**——
   M5 跨域证据表明短段落语料上细切有害（-0.07），由语料分类器自动规避。
3. **语料感知档位**：索引时按平均块长分类（阈值 375，五域校准），三档实现随之切换。
4. **train/test 分离**：路由选型只用 train 集，测试集零参与。

## 目录

**skill 包（[`skill/adaptive-retrieval/`](skill/adaptive-retrieval/)，随 skill 一起分发）**：

| 路径 | 内容 |
|------|------|
| [`SKILL.md`](skill/adaptive-retrieval/SKILL.md) | **可交付的 skill**（SKILL.md + CLI + MCP server + 安装器 + 自测） |
| [`references/harnesses.md`](skill/adaptive-retrieval/references/harnesses.md) | 五个 harness 的路径矩阵、MCP 配置模板、验证与卸载 |
| [`benchmark/`](skill/adaptive-retrieval/benchmark/run_benchmark.py) | **评测框架与复现管线**（M1–M7 数字可复现；含 wx 评测集 `data/testset_v2.json`） |
| [`benchmark/domain_bench.py`](skill/adaptive-retrieval/benchmark/domain_bench.py) | 四域回归对比（改动前后跑同一个表） |
| [`docs/评审-自适应检索Skill的不足与改进.md`](skill/adaptive-retrieval/docs/评审-自适应检索Skill的不足与改进.md) | 评审：不足清单（P0/P1/P2）、证据、已修与待办 |
| [`docs/优化-按影响程度排序.md`](skill/adaptive-retrieval/docs/优化-按影响程度排序.md) | 按影响排序的一轮优化：实测过程、三条负结果、两项落地改造 |

**仓库级（大体积/机器相关，不入库）**：

| 路径 | 内容 | 体积 |
|------|------|------|
| [`experiments/`](experiments/) | M1–M7 报告与结果；负结果一并保留（报告/JSON 入库，二进制除外） | ~5.4GB（本地） |
| `models/` | bge 权重本地副本（skill 运行时向上搜索定位，或走 HF） | ~5.5GB（本地） |
| `data_raw/` | 原始语料（可用 benchmark/build_*.py 重建） | 34MB+（本地） |
| [`开题分析-自适应检索Skill.md`](开题分析-自适应检索Skill.md) | 课题动机、RQ、创新点、里程碑 | — |

benchmark 脚本的两级锚定（`benchmark/_paths.py`）：skill 自身资源按包根（`SKILL_DIR`）定位，
`data_raw/` `models/` `experiments/` 按向上搜索到的仓库根（`REPO_ROOT`）定位——
在完整仓库里开箱即用，独立安装的 skill 缺数据时给出明确报错。

## 当前状态与已知边界

- 三档位、语料自适应、增量索引、自带评测（含 dense 基线、答案级 `answer_recall` 与
  agent 打分表）、策略选购 `tune`（配对 bootstrap + 留出复验）、**重排器领域微调管线**
  均已实现并自测通过；
- **验证有效的唯一精度提升**：排序瓶颈下领域微调重排器，DuReader hit@1
  0.615→0.675（+0.060，P(≤0)=0.025，训练/测试 query 零重叠）；
- **两项已被实验否定的方向**（详见 [优化报告](skill/adaptive-retrieval/docs/优化-按影响程度排序.md)）：
  检索自评门控在四个域上零收益（AUC 0.56–0.61，跨域不可迁移）；
  multi-query / 扩池 / 混合召回在候选池 recall@10 已饱和的语料上收益上限 ≤0.02、实测为 0；
- 超参没有全局最优：同一改动可在 +0.109 与 −0.150 之间翻转，所以调整一律走 `tune`；
- 万级以上 chunk 未测；中文为主，英文可用但未调优；只吃 `.txt/.md`；
- 索引是明文 JSON + npy（含原文），`index_dir` 已在 `.gitignore` 中忽略；
- 详细边界、失败模式排查见 [SKILL.md](skill/adaptive-retrieval/SKILL.md)。

## 已知不足与路线图

- **评审清单（P0/P1/P2）**：[`skill/adaptive-retrieval/docs/评审-自适应检索Skill的不足与改进.md`](skill/adaptive-retrieval/docs/评审-自适应检索Skill的不足与改进.md)——已修/待办逐条带证据；
- **M8 方向**（[M7 报告 §9](experiments/M7报告.md)）：学习型路由多域重训、检索自评估兜底（需 LLM API）、全库规模验证（数十万段落级干扰密度）、英文域阈值泛化。

## License

[MIT](LICENSE)
