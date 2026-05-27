"""热点环境：统一 DashScope compatible-mode，废弃 coding.dashscope 端点。"""

from __future__ import annotations

import os

import pytest

from utils.hot_topics_env import prepare_hot_topics_environment


@pytest.fixture(autouse=True)
def _clear_hot_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for key in (
        "QWEN_APIKEY",
        "CODINGPLAN_API_KEY",
        "CODINGPLAN_BASE_URL",
        "baseurl",
        "INSIGHT_ENGINE_API_KEY",
        "INSIGHT_ENGINE_BASE_URL",
        "INSIGHT_ENGINE_MODEL_NAME",
        "QUERY_ENGINE_BASE_URL",
    ):
        monkeypatch.delenv(key, raising=False)


def test_qwen_apikey_uses_compatible_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("QWEN_APIKEY", "sk-test-qwen")
    prepare_hot_topics_environment()
    assert os.environ.get("INSIGHT_ENGINE_BASE_URL") == "https://dashscope.aliyuncs.com/compatible-mode/v1"


def test_legacy_baseurl_coding_is_remapped(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("QWEN_APIKEY", "sk-test-qwen")
    monkeypatch.setenv("baseurl", "https://coding.dashscope.aliyuncs.com/v1")
    prepare_hot_topics_environment()
    assert os.environ.get("INSIGHT_ENGINE_BASE_URL") == "https://dashscope.aliyuncs.com/compatible-mode/v1"
    assert os.environ.get("baseurl") == "https://dashscope.aliyuncs.com/compatible-mode/v1"


def test_explicit_insight_coding_url_is_remapped(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("QWEN_APIKEY", "sk-test-qwen")
    monkeypatch.setenv("INSIGHT_ENGINE_BASE_URL", "https://coding.dashscope.aliyuncs.com/v1")
    prepare_hot_topics_environment()
    assert os.environ.get("INSIGHT_ENGINE_BASE_URL") == "https://dashscope.aliyuncs.com/compatible-mode/v1"


def test_codingplan_key_does_not_force_coding_endpoint(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CODINGPLAN_API_KEY", "sk-test-coding")
    monkeypatch.setenv("CODINGPLAN_BASE_URL", "https://coding.dashscope.aliyuncs.com/v1")
    monkeypatch.setenv("QWEN_APIKEY", "sk-test-qwen")
    prepare_hot_topics_environment()
    assert "coding.dashscope" not in str(os.environ.get("INSIGHT_ENGINE_BASE_URL", ""))
