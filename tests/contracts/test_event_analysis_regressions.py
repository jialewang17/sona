from __future__ import annotations

import os
import json
from pathlib import Path


def test_event_strict_search_words_levels(monkeypatch) -> None:
    from workflow import event_analysis_pipeline as p

    monkeypatch.setenv("SONA_EVENT_QUERY_STRICT_MODE", "true")
    user_query = "12306回应家长和孩子相隔14个车厢事件"
    base_words = ["12306回应家长和孩子相隔14个车厢事件", "高铁", "回应"]

    w1, level1 = p._pick_search_words_for_round(base_words=base_words, user_query=user_query, round_idx=1)
    assert level1 == "core"
    assert "铁路12306" in w1
    assert any("相隔" in x and "车厢" in x for x in w1)

    w2, level2 = p._pick_search_words_for_round(base_words=base_words, user_query=user_query, round_idx=2)
    assert level2 in {"extended", "core"}
    assert len(w2) >= len(w1)

    w3, level3 = p._pick_search_words_for_round(base_words=base_words, user_query=user_query, round_idx=3)
    assert level3 == "broad"
    assert len(w3) >= len(w2)


def test_topic_relevance_composite_has_phrase_signal() -> None:
    from workflow import event_analysis_pipeline as p

    relevance = p._topic_relevance_metrics(
        user_query="12306回应家长孩子相隔14车厢",
        search_words=["铁路12306", "家长孩子相隔14车厢"],
        top_keywords=["铁路12306", "相隔14车厢", "回应", "安置", "乘务员"],
    )
    assert relevance["coverage"] >= 0.0
    assert relevance["coverage_phrase"] > 0.0
    assert relevance["composite"] > 0.0
    assert any("12306" in x for x in relevance.get("phrase_hits", []))


def test_topic_relevance_flags_generic_keyword_pollution() -> None:
    from workflow import event_analysis_pipeline as p

    relevance = p._topic_relevance_metrics(
        user_query="广州长隆大熊猫家和婷仔健康状况",
        search_words=["广州长隆", "大熊猫家和", "大熊猫婷仔", "健康状况"],
        top_keywords=[
            "中国",
            "国家",
            "发展",
            "工作",
            "生活",
            "社会",
            "全球",
            "世界",
            "家和万事兴",
            "市场",
            "大熊猫",
            "长隆",
        ],
    )
    assert relevance["generic_pollution_suspected"] is True
    assert relevance["generic_top_ratio"] >= 0.5
    assert "家和万事兴" in relevance["generic_top_terms"]


def test_save_run_embedding_matrix(tmp_path: Path) -> None:
    import numpy as np

    from tools.data_bertopic_qwen import _save_run_embedding_matrix

    vecs = np.random.randn(3, 4).astype(np.float32)
    npy_path, meta_path = _save_run_embedding_matrix(tmp_path, vecs, texts=["a", "b", "c"])
    assert npy_path.exists()
    assert meta_path.exists()
    loaded = np.load(npy_path)
    assert loaded.shape == (3, 4)


def test_normalize_topic_label_max_ten_chars() -> None:
    from tools.data_bertopic_qwen import MAX_TOPIC_LABEL_LEN, _normalize_topic_label

    long_name = "李晨白鹿嘉兴录制同框及粉丝共情互动"
    short = _normalize_topic_label(long_name, ["李晨", "白鹿", "沙溢", "奔跑"])
    assert len(short) <= MAX_TOPIC_LABEL_LEN
    assert "李晨" in short

    micro = _normalize_topic_label(
        "白鹿微博微指断层第一及平台热度联动",
        ["奔跑", "白鹿", "断层", "微指", "第一"],
    )
    assert len(micro) <= MAX_TOPIC_LABEL_LEN


def test_generic_topic_label_detection() -> None:
    from tools.data_bertopic_qwen import (
        _build_event_naming_hints,
        _collect_cluster_keyword_hints,
        _is_generic_topic_label,
    )

    assert _is_generic_topic_label("舆情响应与情绪极化") is True
    assert _is_generic_topic_label("核心录制与同框实证") is True
    assert _is_generic_topic_label("奔跑吧白鹿造型争议") is False
    assert _is_generic_topic_label("李晨女艺人同框讨论") is False

    input_data = {
        "主题信息": {
            "主题0": {"文档数": 100, "关键词": ["李晨", "白鹿", "奔跑"]},
            "主题1": {"文档数": 50, "关键词": ["微指", "话题"]},
        }
    }
    hints = _build_event_naming_hints("奔跑吧录制引发同框讨论", input_data)
    assert "李晨" in hints or "奔跑" in hints
    assert _collect_cluster_keyword_hints(input_data)


def test_partition_merge_plan_counts_irrelevant_bucket() -> None:
    from tools.data_bertopic_qwen import (
        MIN_TOPIC_COUNT,
        _partition_merge_plan_for_pipeline,
        _projected_topic_bucket_count,
    )

    valid = {f"主题{i}" for i in range(10)}
    plan = [
        {
            "主题命名": "议题A",
            "原始主题集合": ["主题0", "主题1"],
            "是否无关主题": False,
        },
        {
            "主题命名": "议题B",
            "原始主题集合": ["主题2", "主题3"],
            "是否无关主题": False,
        },
        {
            "主题命名": "无关议题与噪声内容",
            "原始主题集合": ["主题4", "主题5", "主题6"],
            "是否无关主题": True,
        },
        {
            "主题命名": "校验占位（已弃用）",
            "原始主题集合": [],
            "是否无关主题": False,
        },
    ]
    relevant, deferred = _partition_merge_plan_for_pipeline(plan, valid)
    assert len(relevant) == 2
    assert deferred == {"主题4", "主题5", "主题6"}
    assert _projected_topic_bucket_count(relevant, deferred) >= MIN_TOPIC_COUNT


def test_report_topic_bertopic_reads_fallback_and_latest(tmp_path: Path) -> None:
    from tools.report_html_template import (
        _find_topic_bertopic_json,
        build_report_data_from_json_files,
        ensure_topic_bertopic_canonical_for_report,
    )

    topics = [
        {"label": "议题A", "doc_count": 30, "keywords": ["k1", "k2"]},
        {"label": "议题B", "doc_count": 20, "keywords": ["x"]},
    ]
    fallback = {
        "kind": "topic_bertopic",
        "topics": topics,
        "statistics": {"topic_count": 2},
    }
    fallback_path = tmp_path / "topic_bertopic_analysis_fallback_20260529_120641.json"
    fallback_path.write_text(json.dumps(fallback, ensure_ascii=False), encoding="utf-8")

    json_files = [
        {"filename": fallback_path.name, "content": fallback},
        {"filename": "keyword_stats.json", "content": {"keywords": []}},
    ]
    found = _find_topic_bertopic_json(json_files)
    assert found is not None
    assert len(found.get("topics", [])) == 2

    enriched = ensure_topic_bertopic_canonical_for_report(str(tmp_path), json_files)
    latest = next(x for x in enriched if x["filename"] == "topic_bertopic_latest.json")
    assert latest["content"]["topics"][0]["label"] == "议题A"
    assert (tmp_path / "topic_bertopic_latest.json").exists()

    report_data = build_report_data_from_json_files(enriched)
    cluster = report_data["charts"]["topicCluster"]
    assert cluster["available"] is True
    assert len(cluster["docShare"]) == 2


def test_invoke_tool_timeout_propagates_task_id(monkeypatch) -> None:
    from utils.task_context import get_task_id, set_task_id
    from workflow.event_analysis_pipeline import _invoke_tool_to_json_with_timeout

    captured: list[str | None] = []

    class _FakeTool:
        def invoke(self, payload: dict) -> str:
            captured.append(get_task_id())
            return json.dumps({"ok": True, "result_file_path": ""})

    set_task_id("task-propagation-test")
    out = _invoke_tool_to_json_with_timeout(
        _FakeTool(),
        {},
        timeout_sec=30,
        tool_name="fake_tool",
    )
    assert out.get("ok") is True
    assert captured == ["task-propagation-test"]


def test_report_template_strips_collaboration_meta_language() -> None:
    from tools import report_html_template as r

    text = "根据用户补充研判：需要尽快发布完整通报。过程文件显示，微博超话讨论仍在升温。"
    polished = r._polish_report_prose(text)

    assert "用户补充" not in polished
    assert "过程文件" not in polished
    assert "需要尽快发布完整通报" in polished
    assert "监测数据" in polished


def test_morandi_report_template_has_pdf_and_image_export_controls() -> None:
    template = (
        Path(__file__).resolve().parents[2]
        / "prompt"
        / "report_html_morandi_template.html"
    ).read_text(encoding="utf-8")

    assert "html2canvas.min.js" in template
    assert "jspdf.umd.min.js" in template
    assert "exportReportPDF(event)" in template
    assert "exportReportImage(event)" in template
    assert "report-export-toolbar" in template
    assert "@media print" in template
    assert "setChartToolboxVisible(false)" in template


def test_runtime_harness_scores_generic_keyword_pollution(tmp_path: Path) -> None:
    from workflow.runtime_harness import RuntimeHarness

    harness = RuntimeHarness(task_id="case", process_dir=tmp_path, user_query="广州长隆大熊猫家和婷仔健康状况")
    harness.record(
        "topic_relevance_quality",
        {
            "coverage": 0.3,
            "composite": 0.3,
            "min_coverage": 0.08,
            "overlap_count": 2,
            "generic_top_ratio": 0.55,
            "generic_pollution_suspected": True,
        },
    )
    check = harness._score_topic_relevance()
    assert check["status"] == "warning"
    assert check["reason"] == "topic_keywords_generic_pollution"


def test_count_channels_from_csv_platform_column(tmp_path: Path) -> None:
    from workflow import event_analysis_pipeline as p

    csv_path = tmp_path / "sample.csv"
    csv_path.write_text(
        "平台,内容\n微博,a\n微博,b\n小红书,c\n",
        encoding="utf-8",
    )
    counts = p._count_channels_from_csv(str(csv_path))
    assert counts.get("微博") == 2
    assert counts.get("小红书") == 1


def test_timeline_event_relevance_filters_unrelated_rows() -> None:
    # Keep this test pure (no model calls): validate the relevance pre-filter
    import importlib

    tl = importlib.import_module("tools.analysis_timeline")

    rows = [
        {"内容": "粤超足球比赛今晚开赛，门票热卖", "发布时间": "2026-04-23 10:00:00"},
        {"内容": "12306回应：家长孩子相隔14车厢将优化安排", "发布时间": "2026-04-23 11:00:00"},
        {"内容": "铁路12306客服：会协助调换座位", "发布时间": "2026-04-23 12:00:00"},
    ]
    filtered = tl._filter_by_event_relevance(rows, "内容", "12306 家长孩子 相隔14车厢", min_hits=1)
    assert len(filtered) == 2
    assert all("12306" in r["内容"] for r in filtered)


def test_golden_case_12306_report_harness_passed() -> None:
    case_dir = (
        Path(__file__).resolve().parents[2]
        / "eval_results"
        / "golden_cases"
        / "event_analysis_12306_14cars_20260426"
    )
    scorecard_path = case_dir / "runtime_harness_scorecard.json"
    report_meta_path = case_dir / "report_meta.json"
    report_html_path = case_dir / "report.html"

    assert scorecard_path.exists()
    assert report_meta_path.exists()
    assert report_html_path.exists()

    scorecard = json.loads(scorecard_path.read_text(encoding="utf-8"))
    assert scorecard.get("status") == "passed"
    checks = scorecard.get("checks") if isinstance(scorecard.get("checks"), list) else []
    assert any(c.get("name") == "topic_relevance_quality" and c.get("status") == "pass" for c in checks)
    assert any(c.get("name") == "reference_recall_quality" and c.get("status") == "pass" for c in checks)

    meta = json.loads(report_meta_path.read_text(encoding="utf-8"))
    assert meta.get("has_summary") is True
    assert meta.get("has_timeline") is True
    assert meta.get("has_recommendations") is True
    frameworks = meta.get("theory_frameworks") if isinstance(meta.get("theory_frameworks"), list) else []
    assert "议程设置" in frameworks


def test_golden_case_disney_channel_mapping_and_volume_series() -> None:
    from tools import report_html_template as r

    case_dir = (
        Path(__file__).resolve().parents[2]
        / "eval_results"
        / "golden_cases"
        / "event_analysis_disney_smoking_20260427"
    )
    channel_obj = json.loads((case_dir / "process_channel_distribution.json").read_text(encoding="utf-8"))
    pie = r._build_channel_pie_data(channel_obj)
    assert pie
    assert pie[0]["name"] == "微博"
    assert all(x["name"] != "total_count" for x in pie)

    vol_obj = json.loads((case_dir / "process_volume_stats.json").read_text(encoding="utf-8"))
    dates, post_counts, heat_norm, _raw = r._extract_volume_series(vol_obj)
    assert len(dates) == len(post_counts) == len(heat_norm)
    assert max(post_counts) >= 1000  # 2026-04-25 单日破千
    assert 0 <= max(heat_norm) <= 100  # 热度已标准化
