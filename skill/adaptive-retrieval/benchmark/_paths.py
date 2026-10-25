# -*- coding: utf-8 -*-
"""benchmark 脚本共享的路径锚点。

benchmark/ 随 skill 分发，可能处于两种环境：

- 完整仓库内：<repo>/skill/adaptive-retrieval/benchmark/
  （仓库根含 data_raw/ models/ experiments/ 等大体积资产，不随 skill 打包）
- 独立安装的 skill：<skill 安装目录>/benchmark/
  （没有仓库根；数据需用 benchmark/build_*.py 自建，模型走 HF 下载或 AR_MODEL_DIR）

锚点：

- ``SKILL_DIR``   benchmark 的上一级（skill 根：scripts/ references/ docs/ 都在这）
- ``REPO_ROOT``   向上搜索到的仓库根（判定标记：存在 data_raw/ 或 models/）；
                  找不到时退回 SKILL_DIR。data_raw / models / experiments 等
                  仓库级资源都从这里定位——独立安装时这些路径不存在，
                  相关脚本应给出清晰的构建/下载指引。
"""
from __future__ import annotations

from pathlib import Path

_HERE = Path(__file__).resolve()
SKILL_DIR = _HERE.parents[1]

REPO_ROOT = SKILL_DIR
for _base in list(_HERE.parents)[:6]:
    if (_base / "data_raw").exists() or (_base / "models").exists():
        REPO_ROOT = _base
        break
