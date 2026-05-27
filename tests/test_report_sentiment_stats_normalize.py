"""报告模板应兼容 pipeline CSV 情感兜底 statistics 结构。"""

from __future__ import annotations

from tools.report_html_template import (
    _normalize_sentiment_statistics,
    build_report_config_from_json_files,
    build_report_data_from_json_files,
    _overall_attitude_label,
)


def test_normalize_pipeline_csv_fallback_statistics() -> None:
    stats = {
        "total": 1671,
        "positive": {"count": 398, "pct": 23.82},
        "negative": {"count": 83, "pct": 4.97},
        "neutral": {"count": 1190, "pct": 71.21},
        "sentiment_source": "existing_column_fallback",
    }
    norm = _normalize_sentiment_statistics(stats)
    assert norm["positive_count"] == 398
    assert norm["negative_count"] == 83
    assert norm["neutral_count"] == 1190
    assert abs(norm["positive_ratio"] - 0.2382) < 0.001
    assert _overall_attitude_label(stats) == "中性（71.2%）"


def test_build_report_charts_from_fallback_json() -> None:
    payload = {
        "filename": "sentiment_analysis_fallback_20260521_152751.json",
        "content": {
            "statistics": {
                "total": 100,
                "positive": {"count": 30, "pct": 30.0},
                "negative": {"count": 20, "pct": 20.0},
                "neutral": {"count": 50, "pct": 50.0},
                "sentiment_source": "existing_column_fallback",
            }
        },
    }
    cfg = build_report_config_from_json_files([payload])
    assert len(cfg["sentiment"]) == 3
    values = {x["name"]: x["value"] for x in cfg["sentiment"]}
    assert values["正面"] == 30
    assert values["负面"] == 20
    assert values["中立"] == 50

    data = build_report_data_from_json_files([payload])
    coarse = data["charts"]["sentiment"]
    assert sum(x["value"] for x in coarse) == 100
    assert coarse[0]["name"] != "中立" or len(coarse) > 1
