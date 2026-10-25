# experiments/_opt

一次性调参实验的产物，**不是**正式实验记录（正式记录见 experiments/m5..m7 与
docs/优化-按影响程度排序.md）。

- `check.json`：四域基线回归的完整快照（当前代码的 fast/auto/best/dense 指标与稳态延迟）
- `slowpath_tune.json`：慢路径候选池 / RRF / PRF 的对比结果

复现：

    python benchmark/domain_bench.py check      # 生成 check.json
    python benchmark/ceiling_diag.py           # 天花板分解
    python benchmark/preset_bench.py           # 预设对比（含 bootstrap）
