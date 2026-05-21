from __future__ import annotations

import os
import sys
from pathlib import Path

# Wiki：默认关闭 LLM 与微博辅助，避免 CI / 无密钥环境调用外网或非确定模型。
os.environ.setdefault("SONA_WIKI_USE_LLM", "0")
os.environ.setdefault("SONA_WIKI_NEO4J_PRIORITY", "0")
os.environ.setdefault("SONA_WIKI_WEIBO_AUX", "0")
# 默认不在每次 wiki 检索前跑增量编译（避免 CI 对未编译 expert_notes 反复尝试走模型）。
os.environ.setdefault("SONA_WIKI_AUTO_COMPILE_ON_QUERY", "0")

# 专题监测单测若需内存库，在测试里 monkeypatch SONA_TOPIC_MONITOR_SQLITE=0 或传入 DummyDB。

# Ensure project root is importable when running pytest from any cwd.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

