"""情感阶段策略：默认强制 LLM，CSV 兜底需显式开启。"""

from __future__ import annotations

import os
from unittest.mock import MagicMock

import pytest

from workflow.runner import run_sentiment_stage


def _invoke_ok(_tool: object, _payload: dict, **kwargs: object) -> dict:
    return {
        "error": "",
        "statistics": {
            "total": 200,
            "positive_count": 0,
            "negative_count": 200,
            "neutral_count": 0,
            "sentiment_source": "llm_scoring",
            "llm_coverage": 1.0,
        },
        "positive_summary": ["a"],
        "negative_summary": ["b"],
        "result_file_path": "/tmp/sentiment.json",
    }


def _invoke_fail(_tool: object, _payload: dict, **kwargs: object) -> dict:
    return {"error": "analysis_sentiment 超时（>300s）", "result_file_path": ""}


def _csv_fallback(path: str) -> dict:
    return {
        "error": "",
        "statistics": {
            "total": 100,
            "sentiment_source": "existing_column_fallback",
        },
        "positive_summary": [],
        "negative_summary": [],
        "result_file_path": "",
    }


@pytest.fixture
def _base_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("SONA_SENTIMENT_ALLOW_COLUMN_FALLBACK", raising=False)
    monkeypatch.delenv("SONA_SENTIMENT_FORCE_LLM", raising=False)
    monkeypatch.setenv("SONA_SENTIMENT_PREFER_EXISTING", "0")


def test_quality_guard_does_not_replace_llm_when_csv_fallback_disabled(
    monkeypatch: pytest.MonkeyPatch, _base_env: None
) -> None:
    out, _ = run_sentiment_stage(
        user_query="测试",
        search_plan={"eventIntroduction": "简介"},
        save_path="/tmp/data.csv",
        debug=False,
        sentiment_timeout_sec=120,
        analysis_sentiment_tool=MagicMock(),
        invoke_tool_with_timeout=_invoke_ok,
        fallback_from_csv=_csv_fallback,
        append_log=lambda **_: None,
    )
    st = out.get("statistics") or {}
    assert st.get("sentiment_source") == "llm_scoring"


def test_failure_does_not_csv_fallback_by_default(
    monkeypatch: pytest.MonkeyPatch, _base_env: None
) -> None:
    out, _ = run_sentiment_stage(
        user_query="测试",
        search_plan={"eventIntroduction": "简介"},
        save_path="/tmp/data.csv",
        debug=False,
        sentiment_timeout_sec=120,
        analysis_sentiment_tool=MagicMock(),
        invoke_tool_with_timeout=_invoke_fail,
        fallback_from_csv=_csv_fallback,
        append_log=lambda **_: None,
    )
    assert "超时" in str(out.get("error", ""))
    st = out.get("statistics") or {}
    assert st.get("sentiment_source") != "existing_column_fallback"


def test_failure_uses_csv_fallback_when_explicitly_allowed(
    monkeypatch: pytest.MonkeyPatch, _base_env: None
) -> None:
    monkeypatch.setenv("SONA_SENTIMENT_ALLOW_COLUMN_FALLBACK", "1")
    monkeypatch.setenv("SONA_SENTIMENT_FORCE_LLM", "0")
    out, _ = run_sentiment_stage(
        user_query="测试",
        search_plan={"eventIntroduction": "简介"},
        save_path="/tmp/data.csv",
        debug=False,
        sentiment_timeout_sec=120,
        analysis_sentiment_tool=MagicMock(),
        invoke_tool_with_timeout=_invoke_fail,
        fallback_from_csv=_csv_fallback,
        append_log=lambda **_: None,
    )
    st = out.get("statistics") or {}
    assert st.get("sentiment_source") == "existing_column_fallback"


def test_prefer_existing_column_not_set_when_force_llm(
    monkeypatch: pytest.MonkeyPatch, _base_env: None
) -> None:
    captured: list[dict] = []

    def _capture(_tool: object, payload: dict, **kwargs: object) -> dict:
        captured.append(dict(payload))
        return _invoke_ok(_tool, payload, **kwargs)

    monkeypatch.setenv("SONA_BUDGET_SENTIMENT_TOKEN_BUDGET", "10")
    run_sentiment_stage(
        user_query="测试",
        search_plan={"eventIntroduction": "x" * 5000},
        save_path="/tmp/data.csv",
        debug=False,
        sentiment_timeout_sec=120,
        analysis_sentiment_tool=MagicMock(),
        invoke_tool_with_timeout=_capture,
        fallback_from_csv=_csv_fallback,
        append_log=lambda **_: None,
    )
    assert captured
    assert captured[0].get("preferExistingSentimentColumn") is False
