"""专题监测本地 SQLite 存储（默认持久化，不依赖 Supabase/Postgres）。

环境变量 ``SONA_TOPIC_MONITOR_SQLITE``（可选）：

- 未设置或 ``1`` / ``true``：``{项目根}/data/topic_monitor_local.db``；
- 其它非空路径：自定义 ``.db`` 文件（相对路径相对项目根）；
- ``0`` / ``false``：仅当 ``TopicMonitoringPipeline(use_external_db=False)`` 时使用内存库。
"""

from __future__ import annotations

import json
import os
import sqlite3
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv

_project_root = Path(__file__).resolve().parents[1]
_env_path = _project_root / ".env"
if _env_path.is_file():
    load_dotenv(_env_path, override=True)
else:
    load_dotenv()


def resolve_topic_monitor_sqlite_path(project_root: Optional[Path] = None) -> Optional[Path]:
    """
    解析本地库路径；返回 ``None`` 表示调用方应使用内存库（仅演示/单测显式关闭时）。
    """
    root = (project_root or _project_root).resolve()
    raw = str(os.environ.get("SONA_TOPIC_MONITOR_SQLITE", "") or "").strip()
    if raw.lower() in ("0", "false", "no", "n", "off"):
        return None
    if not raw or raw.lower() in ("1", "true", "yes", "on", "y"):
        return root / "data" / "topic_monitor_local.db"
    p = Path(raw).expanduser()
    return p if p.is_absolute() else (root / p).resolve()


def topic_monitor_sqlite_enabled() -> bool:
    """默认启用本地 SQLite（除非环境变量显式设为 0）。"""
    return resolve_topic_monitor_sqlite_path() is not None


def _now_iso() -> str:
    return datetime.utcnow().replace(microsecond=0).isoformat()


def _json_dumps(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, default=str)


def _json_loads(s: Any, default: Any) -> Any:
    if s is None or s == "":
        return default
    if isinstance(s, (dict, list)):
        return s
    try:
        return json.loads(str(s))
    except Exception:
        return default


class TopicMonitorSqliteStore:
    """与 ``SupabaseDB`` 专题监测相关方法对齐的 SQLite 实现。"""

    def __init__(self, db_path: Path) -> None:
        self.db_path = Path(db_path).resolve()
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.is_postgres = False
        self.is_supabase = False
        self._ensure_schema()

    @classmethod
    def from_env(cls, project_root: Optional[Path] = None) -> "TopicMonitorSqliteStore":
        db_path = resolve_topic_monitor_sqlite_path(project_root)
        if db_path is None:
            raise RuntimeError("未配置本地 SQLite 路径（SONA_TOPIC_MONITOR_SQLITE=0）")
        return cls(db_path)

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path), timeout=30)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        return conn

    def _ensure_schema(self) -> None:
        ddl = """
        CREATE TABLE IF NOT EXISTS monitor_topics (
            id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            domain TEXT NOT NULL,
            description TEXT DEFAULT '',
            owner TEXT DEFAULT 'system',
            is_active INTEGER NOT NULL DEFAULT 1,
            config TEXT NOT NULL DEFAULT '{}',
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS topic_keywords (
            id TEXT PRIMARY KEY,
            topic_id TEXT NOT NULL REFERENCES monitor_topics(id) ON DELETE CASCADE,
            keyword TEXT NOT NULL,
            keyword_type TEXT DEFAULT 'include',
            weight REAL DEFAULT 1.0,
            created_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS collected_posts (
            id TEXT PRIMARY KEY,
            topic_id TEXT NOT NULL REFERENCES monitor_topics(id) ON DELETE CASCADE,
            post_id TEXT,
            post_url TEXT,
            platform TEXT,
            author TEXT,
            title TEXT,
            content TEXT,
            likes INTEGER DEFAULT 0,
            comments INTEGER DEFAULT 0,
            shares INTEGER DEFAULT 0,
            sentiment TEXT,
            tags TEXT,
            metadata TEXT,
            collected_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS topic_snapshots (
            id TEXT PRIMARY KEY,
            topic_id TEXT NOT NULL REFERENCES monitor_topics(id) ON DELETE CASCADE,
            post_count INTEGER DEFAULT 0,
            engagement_sum INTEGER DEFAULT 0,
            avg_sentiment REAL,
            top_keywords TEXT,
            volume_trend TEXT,
            summary TEXT,
            created_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS alerts (
            id TEXT PRIMARY KEY,
            topic_id TEXT NOT NULL REFERENCES monitor_topics(id) ON DELETE CASCADE,
            alert_type TEXT NOT NULL,
            title TEXT NOT NULL,
            message TEXT,
            severity TEXT DEFAULT 'info',
            metadata TEXT,
            is_resolved INTEGER NOT NULL DEFAULT 0,
            resolved_at TEXT,
            created_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS case_links (
            id TEXT PRIMARY KEY,
            topic_id TEXT NOT NULL REFERENCES monitor_topics(id) ON DELETE CASCADE,
            case_title TEXT NOT NULL,
            case_domain TEXT,
            case_url TEXT,
            relevance_score REAL DEFAULT 1.0,
            evidence TEXT,
            linked_at TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_topic_keywords_topic_id ON topic_keywords(topic_id);
        CREATE INDEX IF NOT EXISTS idx_collected_posts_topic_id ON collected_posts(topic_id);
        CREATE INDEX IF NOT EXISTS idx_collected_posts_collected_at ON collected_posts(collected_at DESC);
        CREATE INDEX IF NOT EXISTS idx_topic_snapshots_topic_id ON topic_snapshots(topic_id);
        CREATE INDEX IF NOT EXISTS idx_alerts_topic_id ON alerts(topic_id);
        CREATE INDEX IF NOT EXISTS idx_case_links_topic_id ON case_links(topic_id);
        """
        with self._connect() as conn:
            conn.executescript(ddl)

    def _row_topic(self, row: sqlite3.Row) -> Dict[str, Any]:
        d = dict(row)
        d["config"] = _json_loads(d.get("config"), {})
        d["is_active"] = bool(d.get("is_active"))
        return d

    def _row_post(self, row: sqlite3.Row) -> Dict[str, Any]:
        d = dict(row)
        tags = _json_loads(d.get("tags"), [])
        d["tags"] = tags if isinstance(tags, list) else []
        meta = _json_loads(d.get("metadata"), {})
        d["metadata"] = meta if isinstance(meta, dict) else {}
        return d

    def _row_snapshot(self, row: sqlite3.Row) -> Dict[str, Any]:
        d = dict(row)
        tk = _json_loads(d.get("top_keywords"), [])
        d["top_keywords"] = tk if isinstance(tk, list) else []
        return d

    def _row_alert(self, row: sqlite3.Row) -> Dict[str, Any]:
        d = dict(row)
        d["metadata"] = _json_loads(d.get("metadata"), {})
        d["is_resolved"] = bool(d.get("is_resolved"))
        return d

    # --- monitor_topics ---

    def create_monitor_topic(
        self,
        name: str,
        domain: str,
        description: str = "",
        owner: str = "system",
    ) -> Dict[str, Any]:
        tid = str(uuid.uuid4())
        now = _now_iso()
        cfg = _json_dumps({})
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO monitor_topics (id, name, domain, description, owner, is_active, config, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, 1, ?, ?, ?)
                """,
                (tid, name, domain, description or "", owner, cfg, now, now),
            )
            row = conn.execute("SELECT * FROM monitor_topics WHERE id = ?", (tid,)).fetchone()
        assert row is not None
        return self._row_topic(row)

    def get_topic_by_id(self, topic_id: str) -> Optional[Dict[str, Any]]:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM monitor_topics WHERE id = ?", (str(topic_id),)).fetchone()
        return self._row_topic(row) if row else None

    def list_monitor_topics(
        self,
        is_active: Optional[bool] = None,
        domain: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        sql = "SELECT * FROM monitor_topics WHERE 1=1"
        params: List[Any] = []
        if is_active is not None:
            sql += " AND is_active = ?"
            params.append(1 if is_active else 0)
        if domain:
            sql += " AND domain = ?"
            params.append(domain)
        sql += " ORDER BY created_at DESC"
        with self._connect() as conn:
            rows = conn.execute(sql, params).fetchall()
        return [self._row_topic(r) for r in rows]

    def update_monitor_topic(self, topic_id: str, updates: Dict[str, Any]) -> Dict[str, Any]:
        updates = dict(updates)
        updates["updated_at"] = _now_iso()
        if "config" in updates and isinstance(updates["config"], dict):
            updates["config"] = _json_dumps(updates["config"])
        if "is_active" in updates:
            updates["is_active"] = 1 if bool(updates["is_active"]) else 0
        keys = [k for k in updates if k != "id"]
        if not keys:
            row = self.get_topic_by_id(topic_id)
            return row or {}
        sets = ", ".join(f"{k} = ?" for k in keys)
        vals = [updates[k] for k in keys] + [str(topic_id)]
        with self._connect() as conn:
            conn.execute(f"UPDATE monitor_topics SET {sets} WHERE id = ?", vals)
            row = conn.execute("SELECT * FROM monitor_topics WHERE id = ?", (str(topic_id),)).fetchone()
        assert row is not None
        return self._row_topic(row)

    # --- topic_keywords ---

    def add_topic_keywords(self, topic_id: str, keywords: List[Dict[str, str]]) -> List[Dict[str, Any]]:
        now = _now_iso()
        out: List[Dict[str, Any]] = []
        with self._connect() as conn:
            for kw in keywords:
                kid = str(uuid.uuid4())
                conn.execute(
                    """
                    INSERT INTO topic_keywords (id, topic_id, keyword, keyword_type, weight, created_at)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (
                        kid,
                        str(topic_id),
                        str(kw.get("keyword") or "").strip(),
                        str(kw.get("keyword_type") or "include"),
                        float(kw.get("weight") or 1.0),
                        now,
                    ),
                )
                row = conn.execute("SELECT * FROM topic_keywords WHERE id = ?", (kid,)).fetchone()
                if row:
                    out.append(dict(row))
        return out

    def get_topic_keywords(self, topic_id: str) -> List[Dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM topic_keywords WHERE topic_id = ? ORDER BY created_at DESC",
                (str(topic_id),),
            ).fetchall()
        return [dict(r) for r in rows]

    def delete_topic_keyword(self, keyword_id: str) -> None:
        with self._connect() as conn:
            conn.execute("DELETE FROM topic_keywords WHERE id = ?", (str(keyword_id),))

    # --- collected_posts ---

    def collect_post(self, topic_id: str, post_data: Dict[str, Any]) -> Dict[str, Any]:
        pid = str(uuid.uuid4())
        now = _now_iso()
        tags = post_data.get("tags")
        tags_s = _json_dumps(tags if isinstance(tags, list) else [])
        meta = post_data.get("metadata")
        meta_s = _json_dumps(meta if isinstance(meta, dict) else {})
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO collected_posts (
                    id, topic_id, post_id, post_url, platform, author, title, content,
                    likes, comments, shares, sentiment, tags, metadata, collected_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    pid,
                    str(topic_id),
                    post_data.get("post_id"),
                    post_data.get("post_url"),
                    post_data.get("platform"),
                    post_data.get("author"),
                    post_data.get("title"),
                    post_data.get("content"),
                    int(post_data.get("likes") or 0),
                    int(post_data.get("comments") or 0),
                    int(post_data.get("shares") or 0),
                    post_data.get("sentiment"),
                    tags_s,
                    meta_s,
                    now,
                ),
            )
            row = conn.execute("SELECT * FROM collected_posts WHERE id = ?", (pid,)).fetchone()
        assert row is not None
        return self._row_post(row)

    def bulk_collect_posts(self, topic_id: str, posts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        return [self.collect_post(topic_id, p) for p in posts]

    def get_collected_posts(
        self,
        topic_id: str,
        limit: int = 100,
        since: Optional[datetime] = None,
    ) -> List[Dict[str, Any]]:
        sql = "SELECT * FROM collected_posts WHERE topic_id = ?"
        params: List[Any] = [str(topic_id)]
        if since is not None:
            since_iso = since.replace(tzinfo=None).isoformat() if since.tzinfo else since.isoformat()
            sql += " AND collected_at >= ?"
            params.append(since_iso)
        sql += " ORDER BY collected_at DESC LIMIT ?"
        params.append(int(limit))
        with self._connect() as conn:
            rows = conn.execute(sql, params).fetchall()
        return [self._row_post(r) for r in rows]

    # --- snapshots ---

    def create_snapshot(self, topic_id: str, snapshot_data: Dict[str, Any]) -> Dict[str, Any]:
        sid = str(uuid.uuid4())
        now = _now_iso()
        tk = snapshot_data.get("top_keywords")
        tk_s = _json_dumps(tk if isinstance(tk, list) else [])
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO topic_snapshots (
                    id, topic_id, post_count, engagement_sum, avg_sentiment, top_keywords,
                    volume_trend, summary, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    sid,
                    str(topic_id),
                    int(snapshot_data.get("post_count") or 0),
                    int(snapshot_data.get("engagement_sum") or 0),
                    float(snapshot_data.get("avg_sentiment") or 0.0),
                    tk_s,
                    str(snapshot_data.get("volume_trend") or "stable"),
                    str(snapshot_data.get("summary") or ""),
                    now,
                ),
            )
            row = conn.execute("SELECT * FROM topic_snapshots WHERE id = ?", (sid,)).fetchone()
        assert row is not None
        return self._row_snapshot(row)

    def get_latest_snapshot(self, topic_id: str) -> Optional[Dict[str, Any]]:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM topic_snapshots WHERE topic_id = ? ORDER BY created_at DESC LIMIT 1",
                (str(topic_id),),
            ).fetchone()
        return self._row_snapshot(row) if row else None

    def get_snapshots(self, topic_id: str, limit: int = 10) -> List[Dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM topic_snapshots WHERE topic_id = ? ORDER BY created_at DESC LIMIT ?",
                (str(topic_id), int(limit)),
            ).fetchall()
        return [self._row_snapshot(r) for r in rows]

    # --- alerts ---

    def create_alert(
        self,
        topic_id: str,
        alert_type: str,
        title: str,
        message: str,
        severity: str = "info",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        aid = str(uuid.uuid4())
        now = _now_iso()
        meta_s = _json_dumps(metadata or {})
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO alerts (
                    id, topic_id, alert_type, title, message, severity, metadata,
                    is_resolved, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, 0, ?)
                """,
                (aid, str(topic_id), alert_type, title, message, severity, meta_s, now),
            )
            row = conn.execute("SELECT * FROM alerts WHERE id = ?", (aid,)).fetchone()
        assert row is not None
        return self._row_alert(row)

    def list_alerts(
        self,
        topic_id: Optional[str] = None,
        is_resolved: Optional[bool] = None,
        severity: Optional[str] = None,
        limit: int = 50,
    ) -> List[Dict[str, Any]]:
        sql = "SELECT * FROM alerts WHERE 1=1"
        params: List[Any] = []
        if topic_id:
            sql += " AND topic_id = ?"
            params.append(str(topic_id))
        if is_resolved is not None:
            sql += " AND is_resolved = ?"
            params.append(1 if is_resolved else 0)
        if severity:
            sql += " AND severity = ?"
            params.append(severity)
        sql += " ORDER BY created_at DESC LIMIT ?"
        params.append(int(limit))
        with self._connect() as conn:
            rows = conn.execute(sql, params).fetchall()
        return [self._row_alert(r) for r in rows]

    def resolve_alert(self, alert_id: str) -> Dict[str, Any]:
        now = _now_iso()
        with self._connect() as conn:
            conn.execute(
                "UPDATE alerts SET is_resolved = 1, resolved_at = ? WHERE id = ?",
                (now, str(alert_id)),
            )
            row = conn.execute("SELECT * FROM alerts WHERE id = ?", (str(alert_id),)).fetchone()
        assert row is not None
        return self._row_alert(row)

    # --- case_links ---

    def link_case(
        self,
        topic_id: str,
        case_title: str,
        case_domain: str,
        case_url: str,
        relevance_score: float = 1.0,
        evidence: str = "",
    ) -> Dict[str, Any]:
        lid = str(uuid.uuid4())
        now = _now_iso()
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO case_links (
                    id, topic_id, case_title, case_domain, case_url, relevance_score, evidence, linked_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (lid, str(topic_id), case_title, case_domain or "", case_url or "", relevance_score, evidence or "", now),
            )
            row = conn.execute("SELECT * FROM case_links WHERE id = ?", (lid,)).fetchone()
        assert row is not None
        return dict(row)

    def get_linked_cases(self, topic_id: str, min_score: float = 0.5) -> List[Dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM case_links WHERE topic_id = ? AND relevance_score >= ? ORDER BY relevance_score DESC",
                (str(topic_id), float(min_score)),
            ).fetchall()
        return [dict(r) for r in rows]

    def _timestamp_for_write(self) -> Any:
        return datetime.utcnow()
