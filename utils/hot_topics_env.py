"""热点流程（hottopics）与 Sona 统一环境：加载 .env 并映射 API Key 到 INSIGHT_/QUERY_ 变量。"""

from __future__ import annotations

import os
from typing import Optional

import yaml

from utils.env_loader import get_env_config
from utils.path import get_config_path, get_project_root

_DASHSCOPE_COMPATIBLE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"


def _set_if_absent(key: str, value: Optional[str]) -> None:
    if value and key not in os.environ:
        os.environ[key] = value


def _normalize_dashscope_base_url(url: str) -> str:
    """将已废弃的 coding.dashscope 端点统一为按量 compatible-mode。"""
    raw = str(url or "").strip()
    if not raw:
        return _DASHSCOPE_COMPATIBLE_URL
    if "coding.dashscope.aliyuncs.com" in raw:
        return _DASHSCOPE_COMPATIBLE_URL
    return raw


def _sanitize_legacy_dashscope_endpoints() -> None:
    """清理 .env 中遗留的 Coding Plan / baseurl 配置，避免热点流程误用错误端点。"""
    for key in (
        "INSIGHT_ENGINE_BASE_URL",
        "QUERY_ENGINE_BASE_URL",
        "REPORT_ENGINE_BASE_URL",
        "OPENAI_BASE_URL",
        "baseurl",
        "CODINGPLAN_BASE_URL",
    ):
        val = str(os.environ.get(key) or "").strip()
        if val and "coding.dashscope.aliyuncs.com" in val:
            os.environ[key] = _DASHSCOPE_COMPATIBLE_URL


def prepare_hot_topics_environment() -> None:
    """
    在导入或运行 tools.hottopics 之前调用。

    1. 通过 EnvConfig 加载项目根目录 .env（与主 Agent 一致）。
    2. 将 Sona 使用的变量名映射到 hottopics 内 InsightNode / ForumNode 所需的
       INSIGHT_ENGINE_*、QUERY_ENGINE_*（OpenAI 兼容接口，通义走 compatible-mode）。

    优先级（仅当对应 INSIGHT_/QUERY_ 未显式设置时填充）：
    - QWEN_APIKEY / model.yaml tools profile → DashScope compatible-mode
    - KIMI_API_KEY / KIMI_APIKEY → Moonshot
    - OPENAI_API_KEY / OPENAI_APIKEY → OpenAI 官方
    - DEEPSEEK_APIKEY → DeepSeek
    """
    env = get_env_config()
    _sanitize_legacy_dashscope_endpoints()

    kimi = os.environ.get("KIMI_API_KEY") or os.environ.get("KIMI_APIKEY")
    openai = os.environ.get("OPENAI_API_KEY") or os.environ.get("OPENAI_APIKEY")
    deepseek = os.environ.get("DEEPSEEK_APIKEY") or os.environ.get("DEEPSEEK_API_KEY")

    moonshot_base = "https://api.moonshot.cn/v1"
    moonshot_model = os.environ.get("KIMI_MODEL_NAME") or os.environ.get("KIMI_MODEL") or "moonshot-v1-8k"

    if not os.environ.get("INSIGHT_ENGINE_API_KEY"):
        model_cfg_path = get_config_path("model.yaml")
        if model_cfg_path.exists():
            try:
                with open(model_cfg_path, "r", encoding="utf-8") as f:
                    model_cfg = yaml.safe_load(f) or {}

                tools_block = model_cfg.get("tools") if isinstance(model_cfg.get("tools"), dict) else None
                if not tools_block:
                    tools_block = model_cfg.get("main") if isinstance(model_cfg.get("main"), dict) else None

                if tools_block:
                    provider = str(tools_block.get("provider") or "").lower()
                    api_key_env = str(tools_block.get("api_key_env") or "").strip()
                    api_key = env.get_api_key(api_key_env) if api_key_env else None

                    if provider in {"qwen", "openai", "deepseek", "dashscope", "kimi"} and api_key:
                        base_url = _normalize_dashscope_base_url(str(tools_block.get("base_url") or "").strip())
                        model_name = str(tools_block.get("model") or "").strip()

                        _set_if_absent("INSIGHT_ENGINE_API_KEY", api_key)

                        if provider in {"qwen", "dashscope"}:
                            _set_if_absent("INSIGHT_ENGINE_BASE_URL", base_url or _DASHSCOPE_COMPATIBLE_URL)
                        elif base_url:
                            _set_if_absent("INSIGHT_ENGINE_BASE_URL", base_url)

                        if model_name:
                            _set_if_absent("INSIGHT_ENGINE_MODEL_NAME", model_name)
            except Exception:
                pass

    qwen = (
        os.environ.get("QWEN_APIKEY")
        or os.environ.get("QWEN_API_KEY")
        or os.environ.get("DASHSCOPE_APIKEY")
        or os.environ.get("APIKEY")
    )
    qwen_model = (
        os.environ.get("QWEN_MODEL_NAME")
        or os.environ.get("QWEN_MODEL")
        or os.environ.get("INSIGHT_ENGINE_MODEL_NAME")
        or "qwen-plus"
    )
    legacy_base = _normalize_dashscope_base_url(str(os.environ.get("baseurl") or "").strip())

    if qwen:
        _set_if_absent("INSIGHT_ENGINE_API_KEY", qwen)
        _set_if_absent("INSIGHT_ENGINE_MODEL_NAME", qwen_model)
        if not str(os.environ.get("INSIGHT_ENGINE_BASE_URL") or "").strip():
            os.environ["INSIGHT_ENGINE_BASE_URL"] = legacy_base or _DASHSCOPE_COMPATIBLE_URL

    if not os.environ.get("INSIGHT_ENGINE_API_KEY"):
        if kimi:
            _set_if_absent("INSIGHT_ENGINE_API_KEY", kimi)
            _set_if_absent("INSIGHT_ENGINE_BASE_URL", os.environ.get("KIMI_BASE_URL") or moonshot_base)
            _set_if_absent("INSIGHT_ENGINE_MODEL_NAME", moonshot_model)
        elif openai:
            _set_if_absent("INSIGHT_ENGINE_API_KEY", openai)
            _set_if_absent("INSIGHT_ENGINE_BASE_URL", os.environ.get("OPENAI_BASE_URL") or "https://api.openai.com/v1")
            _set_if_absent(
                "INSIGHT_ENGINE_MODEL_NAME",
                os.environ.get("OPENAI_MODEL") or os.environ.get("INSIGHT_ENGINE_MODEL_NAME") or "gpt-4o-mini",
            )
        elif deepseek:
            _set_if_absent("INSIGHT_ENGINE_API_KEY", deepseek)
            _set_if_absent("INSIGHT_ENGINE_BASE_URL", "https://api.deepseek.com/v1")
            _set_if_absent(
                "INSIGHT_ENGINE_MODEL_NAME",
                os.environ.get("DEEPSEEK_MODEL") or "deepseek-chat",
            )

    if not os.environ.get("QUERY_ENGINE_API_KEY") and os.environ.get("INSIGHT_ENGINE_API_KEY"):
        _set_if_absent("QUERY_ENGINE_API_KEY", os.environ["INSIGHT_ENGINE_API_KEY"])
        _set_if_absent("QUERY_ENGINE_MODEL_NAME", os.environ.get("INSIGHT_ENGINE_MODEL_NAME", moonshot_model))
        if not str(os.environ.get("QUERY_ENGINE_BASE_URL") or "").strip():
            os.environ["QUERY_ENGINE_BASE_URL"] = os.environ.get("INSIGHT_ENGINE_BASE_URL", moonshot_base)

    insight_base = str(os.environ.get("INSIGHT_ENGINE_BASE_URL") or "").strip()
    if insight_base:
        os.environ["INSIGHT_ENGINE_BASE_URL"] = _normalize_dashscope_base_url(insight_base)
    query_base = str(os.environ.get("QUERY_ENGINE_BASE_URL") or "").strip()
    if query_base:
        os.environ["QUERY_ENGINE_BASE_URL"] = _normalize_dashscope_base_url(query_base)

    _sanitize_legacy_dashscope_endpoints()


def ensure_hot_topics_cwd() -> None:
    """将工作目录设为项目根，保证 output_langgraph、data_langgraph 写在仓库根目录。"""
    root = get_project_root()
    try:
        os.chdir(root)
    except OSError:
        pass
