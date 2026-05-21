"""专题监测 SQLite 本地存储契约测试。"""

from __future__ import annotations

from pathlib import Path

import pytest

from workflow.topic_monitor_sqlite import TopicMonitorSqliteStore
from workflow.topic_monitoring_pipeline import TopicMonitoringPipeline


def test_sqlite_store_roundtrip(tmp_path: Path) -> None:
    db_path = tmp_path / "mon.db"
    store = TopicMonitorSqliteStore(db_path)
    topic = store.create_monitor_topic(name="测试专题", domain="交通", description="d", owner="t")
    tid = str(topic["id"])
    store.add_topic_keywords(
        tid,
        [{"keyword": "事故", "keyword_type": "include", "weight": 1.0}],
    )
    store.bulk_collect_posts(
        tid,
        [
            {
                "post_id": "p1",
                "post_url": "https://example.com/1",
                "platform": "微博",
                "author": "a",
                "title": "t",
                "content": "c",
                "likes": 1,
                "comments": 2,
                "shares": 0,
                "sentiment": "neutral",
                "tags": ["x"],
                "metadata": {"k": 1},
            }
        ],
    )
    posts = store.get_collected_posts(tid, limit=10)
    assert len(posts) == 1
    assert posts[0]["tags"] == ["x"]
    assert isinstance(posts[0]["metadata"], dict)
    snap = store.create_snapshot(
        tid,
        {
            "post_count": 1,
            "engagement_sum": 3,
            "avg_sentiment": 0.0,
            "top_keywords": ["事故"],
            "volume_trend": "stable",
            "summary": "ok",
        },
    )
    assert snap.get("post_count") == 1
    assert store.get_latest_snapshot(tid) is not None


def test_pipeline_uses_sqlite_by_default(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("SONA_TOPIC_MONITOR_SQLITE", str(tmp_path / "p.db"))
    p = TopicMonitoringPipeline()
    t = p.create_topic(name="SQLite 专题", domain="综合舆情", keywords=["测试"], description="")
    tid = str(t["id"])
    p.scan_topic(tid, [])
    st = p.get_topic_status(tid)
    assert not st.get("error")
    assert st.get("topic", {}).get("name") == "SQLite 专题"
