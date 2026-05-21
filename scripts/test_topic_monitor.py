#!/usr/bin/env python3
"""专题监测（/monitor）端到端烟测脚本。

默认 ``--mode quick``：独立临时 SQLite + 模拟帖子，不依赖网察，数十秒内完成。

用法::

    cd D:\\sona-master
    python scripts/test_topic_monitor.py
    python scripts/test_topic_monitor.py --mode quick
    python scripts/test_topic_monitor.py --mode local
    python scripts/test_topic_monitor.py --mode netinsight
    python scripts/test_topic_monitor.py --mode netinsight --topic-id <已有专题UUID>
    python scripts/test_topic_monitor.py --check-netinsight-only

仅导出本地库为 JSON（不写临时库、不跑烟测）::

    python scripts/test_topic_monitor.py --export-json data/my_topic.json --topic-id <uuid>
    python scripts/test_topic_monitor.py --export-all --export-dir data/topic_monitor_exports

烟测成功后**顺带**写出 JSON（勿写 ``<topic_id>``，PowerShell 会当成重定向；仅 ``local`` / ``netinsight``）::

    python scripts/test_topic_monitor.py --mode netinsight --name "电影市场" --keywords "电影,票房" --also-export-json data/电影市场.json

环境（``--mode netinsight`` 时需 .env 中配置）::

    SONA_TOPIC_MONITOR_USE_OPINION_NETINSIGHT=1
    SONA_OPINION_SYSTEM_ROOT=D:/netinsight
    NETINSIGHT_USER=...
    NETINSIGHT_PASS=...

可选缩小拉取规模（烟测推荐）::

    SONA_MONITOR_NETINSIGHT_ROW_CAP=200
    SONA_TOPIC_MONITOR_NETINSIGHT_WINDOW_HOURS=6

本地库为 SQLite（``data/topic_monitor_local.db``）；JSON 预览见上方 ``--export-json`` / ``--export-all``。
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from workflow.topic_monitor_sqlite import (  # noqa: E402
    TopicMonitorSqliteStore,
    resolve_topic_monitor_sqlite_path,
    topic_monitor_sqlite_enabled,
)
from workflow.topic_monitoring_pipeline import TopicMonitoringPipeline  # noqa: E402
from workflow.topic_monitoring_workflow import build_default_topic_config  # noqa: E402

SearchFunc = Callable[[List[str], str, int], List[Dict[str, Any]]]


def _utc_tag() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def _mock_search_func(keyword_list: List[str], topic_id: str, cycle_idx: int) -> List[Dict[str, Any]]:
    """模拟网察返回，用于不连网的快速烟测。"""
    base = f"smoke-{topic_id[:8]}-{cycle_idx}"
    posts: List[Dict[str, Any]] = []
    for i, kw in enumerate(keyword_list[:3] or ["测试"]):
        posts.append(
            {
                "id": f"{base}-{i}",
                "url": f"https://example.com/post/{base}/{i}",
                "platform": "微博" if i % 2 == 0 else "新闻网站",
                "author": "smoke_bot",
                "title": f"[烟测] {kw} 相关帖 {i}",
                "content": f"这是专题监测烟测帖子，关键词={kw}，cycle={cycle_idx}。",
                "likes": 120 + i * 10,
                "comments": 30 + i,
                "shares": 5,
                "sentiment": "neutral",
                "published_at": datetime.now(timezone.utc).replace(tzinfo=None).isoformat(timespec="seconds"),
            }
        )
    return posts


def _print_step(title: str) -> None:
    print(f"\n=== {title} ===", flush=True)


def _bundle_topic_for_json(
    store: TopicMonitorSqliteStore,
    topic_id: str,
    *,
    post_limit: int,
    snapshot_limit: int,
    alert_limit: int,
) -> Dict[str, Any]:
    topic = store.get_topic_by_id(topic_id)
    if not topic:
        raise ValueError(f"专题不存在: {topic_id}")
    return {
        "exported_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "topic": topic,
        "keywords": store.get_topic_keywords(topic_id),
        "posts": store.get_collected_posts(topic_id, limit=post_limit),
        "posts_note": (
            f"最多 {post_limit} 条（collected_at 倒序）；调大 --export-post-limit。"
        ),
        "snapshots": store.get_snapshots(topic_id, limit=snapshot_limit),
        "alerts": store.list_alerts(topic_id=str(topic_id), limit=alert_limit),
        "linked_cases": store.get_linked_cases(topic_id=str(topic_id), min_score=0.0),
    }


def _write_json_file(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )


def run_export_json(
    *,
    topic_id: str,
    out_path: Path,
    post_limit: int,
    snapshot_limit: int,
    alert_limit: int,
) -> Dict[str, Any]:
    store = TopicMonitorSqliteStore.from_env()
    bundle = _bundle_topic_for_json(
        store,
        topic_id,
        post_limit=post_limit,
        snapshot_limit=snapshot_limit,
        alert_limit=alert_limit,
    )
    _write_json_file(out_path, bundle)
    return {"ok": True, "out": str(out_path.resolve()), "database": str(store.db_path)}


def run_export_all_json(
    *,
    out_dir: Path,
    post_limit: int,
    snapshot_limit: int,
    alert_limit: int,
) -> Dict[str, Any]:
    store = TopicMonitorSqliteStore.from_env()
    topics = store.list_monitor_topics()
    written: List[str] = []
    out_dir.mkdir(parents=True, exist_ok=True)
    for t in topics:
        tid = str(t.get("id") or "")
        if not tid:
            continue
        bundle = _bundle_topic_for_json(
            store,
            tid,
            post_limit=post_limit,
            snapshot_limit=snapshot_limit,
            alert_limit=alert_limit,
        )
        path = out_dir / f"topic_{tid}.json"
        _write_json_file(path, bundle)
        written.append(str(path.resolve()))
    index = {
        "exported_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "database": str(store.db_path),
        "count": len(written),
        "files": written,
    }
    _write_json_file(out_dir / "index.json", index)
    return {"ok": True, "out_dir": str(out_dir.resolve()), "count": len(written)}


def _check_netinsight_ready() -> Dict[str, Any]:
    from workflow.topic_netinsight_adapter import (
        load_opinion_netinsight_client,
        opinion_system_root,
        topic_monitor_use_opinion_netinsight,
    )

    root = opinion_system_root()
    flat = root / "client.py"
    pkg = root / "backend" / "src" / "netinsight" / "client.py"
    user = str(os.environ.get("NETINSIGHT_USER") or os.environ.get("NETINSIGHT_USERNAME") or "").strip()
    password = str(os.environ.get("NETINSIGHT_PASS") or os.environ.get("NETINSIGHT_PASSWORD") or "").strip()
    out: Dict[str, Any] = {
        "use_flag": topic_monitor_use_opinion_netinsight(),
        "opinion_root": str(root),
        "client_flat": flat.is_file(),
        "client_pkg": pkg.is_file(),
        "has_credentials": bool(user and password),
    }
    try:
        load_opinion_netinsight_client()
        out["client_load"] = "ok"
    except Exception as exc:  # noqa: BLE001
        out["client_load"] = f"error: {exc}"
    out["ok"] = (
        out["use_flag"]
        and out["has_credentials"]
        and (out["client_flat"] or out["client_pkg"])
        and out["client_load"] == "ok"
    )
    return out


def _resolve_search_func(
    pipeline: TopicMonitoringPipeline,
    *,
    mode: str,
) -> tuple[Optional[SearchFunc], str]:
    if mode == "quick":
        return _mock_search_func, "mock"

    from workflow.topic_netinsight_adapter import (
        build_opinion_netinsight_search_func,
        topic_monitor_use_opinion_netinsight,
    )

    if mode == "local":
        return _mock_search_func, "mock"

    if not topic_monitor_use_opinion_netinsight():
        return None, "none"

    try:
        return build_opinion_netinsight_search_func(pipeline), "netinsight"
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"构建 NetInsight search_func 失败: {exc}") from exc


def _run_flow(
    pipeline: TopicMonitoringPipeline,
    *,
    mode: str,
    topic_id: Optional[str],
    name: str,
    domain: str,
    keywords: List[str],
    report_dir: Path,
    skip_create: bool,
) -> Dict[str, Any]:
    result: Dict[str, Any] = {"mode": mode, "steps": []}

    tid = (topic_id or "").strip()
    if not tid:
        if skip_create:
            raise ValueError("未指定 --topic-id 且未创建新专题")
        _print_step("1/5 创建专题")
        topic = pipeline.create_topic(
            name=name,
            domain=domain,
            keywords=keywords,
            description=f"scripts/test_topic_monitor.py 烟测 {_utc_tag()}",
        )
        tid = str(topic.get("id") or "")
        cfg = build_default_topic_config()
        if mode == "netinsight":
            cfg["netinsight_max_rows_hint"] = min(
                int(cfg.get("netinsight_max_rows_hint") or 10_000),
                int(os.environ.get("SONA_MONITOR_NETINSIGHT_ROW_CAP", "500") or 500),
            )
            cfg["netinsight_pull_window_hours"] = float(
                os.environ.get("SONA_TOPIC_MONITOR_NETINSIGHT_WINDOW_HOURS", "6") or 6
            )
        pipeline.patch_topic_config(tid, cfg)
        result["steps"].append({"create": {"topic_id": tid, "name": name, "keywords": keywords}})
        print(json.dumps(result["steps"][-1], ensure_ascii=False, indent=2))
    else:
        topic = pipeline.db.get_topic_by_id(tid)
        if not topic:
            raise ValueError(f"专题不存在: {tid}")
        result["steps"].append({"reuse_topic": {"topic_id": tid, "name": topic.get("name")}})
        print(json.dumps(result["steps"][-1], ensure_ascii=False, indent=2))

    _print_step("2/5 监测周期（拉帖 + 快照）")
    search_func, search_label = _resolve_search_func(pipeline, mode=mode)
    if mode == "netinsight" and search_label == "none":
        print(
            "[warn] SONA_TOPIC_MONITOR_USE_OPINION_NETINSIGHT 未开启，本轮不拉网帖。"
            "可在 .env 设为 1 后重试。",
            flush=True,
        )
    elif search_label == "netinsight":
        print("[info] 使用 NetInsight 拉帖（可能需数分钟）…", flush=True)

    cycle = pipeline.run_monitoring_cycle([tid], search_func=search_func)
    snap = {}
    for item in cycle.get("results") or []:
        if str(item.get("topic_id") or "") == tid:
            snap = item.get("snapshot") if isinstance(item.get("snapshot"), dict) else {}
            break
    post_count = int(snap.get("post_count") or 0)
    result["steps"].append(
        {
            "cycle": {
                "search": search_label,
                "post_count": post_count,
                "summary": snap.get("summary"),
                "alerts": len((cycle.get("results") or [{}])[0].get("alerts") or []),
            }
        }
    )
    print(json.dumps(result["steps"][-1], ensure_ascii=False, indent=2))

    _print_step("3/5 专题状态")
    status = pipeline.get_topic_status(tid)
    if status.get("error"):
        raise RuntimeError(str(status["error"]))
    stored_posts = len(pipeline.db.get_collected_posts(tid, limit=10_000))
    result["steps"].append(
        {
            "status": {
                "topic_id": tid,
                "name": (status.get("topic") or {}).get("name"),
                "latest_snapshot": status.get("latest_snapshot"),
                "active_alerts": len(status.get("active_alerts") or []),
                "collected_posts_in_db": stored_posts,
            }
        }
    )
    print(json.dumps(result["steps"][-1], ensure_ascii=False, indent=2))

    _print_step("4/5 生成日报")
    report_dir.mkdir(parents=True, exist_ok=True)
    report = pipeline.generate_periodic_report(tid, period="daily", output_dir=report_dir)
    report_path = Path(str(report.get("report_path") or ""))
    result["steps"].append(
        {
            "report": {
                "period": "daily",
                "report_path": str(report_path),
                "exists": report_path.is_file(),
                "size_bytes": report_path.stat().st_size if report_path.is_file() else 0,
            }
        }
    )
    print(json.dumps(result["steps"][-1], ensure_ascii=False, indent=2))

    _print_step("5/5 汇总")
    ok = post_count > 0 or search_label == "none"
    if mode in ("quick", "local"):
        ok = stored_posts > 0 and post_count > 0
    result["topic_id"] = tid
    result["ok"] = ok
    result["hint"] = (
        f"专题 ID: {tid}；查看状态: sona monitor status {tid}；"
        f"报告目录: {report_dir}"
    )
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="专题监测 /monitor 端到端烟测")
    parser.add_argument(
        "--mode",
        choices=("quick", "local", "netinsight"),
        default="quick",
        help="quick=临时库+模拟帖；local=项目 data/topic_monitor_local.db+模拟帖；netinsight=真实网察",
    )
    parser.add_argument("--topic-id", default="", help="跳过创建，仅对已有专题跑 cycle/status/report")
    parser.add_argument("--name", default="监测烟测专题", help="新建专题名称")
    parser.add_argument("--domain", default="交通", help="新建专题领域")
    parser.add_argument(
        "--keywords",
        default="高铁,交通事故",
        help="逗号分隔关键词（新建时）",
    )
    parser.add_argument(
        "--report-dir",
        default="",
        help="报告输出目录，默认 topic_monitoring_reports/smoke_<tag>",
    )
    parser.add_argument(
        "--keep-db",
        action="store_true",
        help="quick 模式保留临时库路径（默认测完删除）",
    )
    parser.add_argument(
        "--check-netinsight-only",
        action="store_true",
        help="仅检查 NetInsight 环境与客户端能否加载",
    )
    parser.add_argument(
        "--export-json",
        metavar="OUT",
        default="",
        help="仅导出：将 --topic-id 对应专题写入 OUT（UTF-8 JSON）；不运行烟测",
    )
    parser.add_argument(
        "--export-all",
        action="store_true",
        help="仅导出：全部专题写入 --export-dir；不运行烟测",
    )
    parser.add_argument(
        "--export-dir",
        default="",
        help="配合 --export-all，默认 data/topic_monitor_exports/<UTC时间戳>",
    )
    parser.add_argument("--export-post-limit", type=int, default=10_000)
    parser.add_argument("--export-snapshot-limit", type=int, default=50)
    parser.add_argument("--export-alert-limit", type=int, default=200)
    parser.add_argument(
        "--also-export-json",
        metavar="OUT",
        default="",
        help="烟测成功后额外导出该轮 topic 到 OUT（UTF-8 JSON）；仅 --mode local/netinsight 有效；quick 模式会跳过",
    )
    args = parser.parse_args()

    if args.check_netinsight_only:
        _print_step("NetInsight 环境检查")
        info = _check_netinsight_ready()
        print(json.dumps(info, ensure_ascii=False, indent=2))
        return 0 if info.get("ok") else 2

    also_json = str(getattr(args, "also_export_json", "") or "").strip()
    if also_json and (str(args.export_json or "").strip() or args.export_all):
        print(json.dumps({"ok": False, "error": "--also-export-json 不能与 --export-json / --export-all 同用"}, ensure_ascii=False))
        return 2

    ex_out = str(args.export_json or "").strip()
    if ex_out or args.export_all:
        if ex_out and args.export_all:
            print(json.dumps({"ok": False, "error": "不能同时使用 --export-json 与 --export-all"}, ensure_ascii=False))
            return 2
        if not topic_monitor_sqlite_enabled():
            print("[error] 本地 SQLite 未启用（SONA_TOPIC_MONITOR_SQLITE=0）", flush=True)
            return 2
        try:
            if ex_out:
                tid = str(args.topic_id or "").strip()
                if not tid:
                    store = TopicMonitorSqliteStore.from_env()
                    topics = store.list_monitor_topics()
                    print(
                        json.dumps(
                            {
                                "ok": False,
                                "error": "--export-json 需要 --topic-id",
                                "database": str(store.db_path),
                                "topics": [
                                    {"id": str(x.get("id")), "name": x.get("name"), "domain": x.get("domain")}
                                    for x in topics
                                ],
                            },
                            ensure_ascii=False,
                            indent=2,
                        ),
                        flush=True,
                    )
                    return 2
                r = run_export_json(
                    topic_id=tid,
                    out_path=_ROOT / ex_out if not Path(ex_out).is_absolute() else Path(ex_out),
                    post_limit=args.export_post_limit,
                    snapshot_limit=args.export_snapshot_limit,
                    alert_limit=args.export_alert_limit,
                )
                print(json.dumps(r, ensure_ascii=False, indent=2))
                return 0

            tag = _utc_tag()
            ed = str(args.export_dir or "").strip()
            if ed:
                p = Path(ed)
                out_dir = p.resolve() if p.is_absolute() else (_ROOT / p).resolve()
            else:
                out_dir = (_ROOT / "data" / "topic_monitor_exports" / tag).resolve()
            r = run_export_all_json(
                out_dir=out_dir,
                post_limit=args.export_post_limit,
                snapshot_limit=args.export_snapshot_limit,
                alert_limit=args.export_alert_limit,
            )
            print(json.dumps(r, ensure_ascii=False, indent=2))
            return 0
        except Exception as exc:  # noqa: BLE001
            print(json.dumps({"ok": False, "error": str(exc)}, ensure_ascii=False), flush=True)
            return 1

    keywords = [k.strip() for k in str(args.keywords).split(",") if k.strip()]
    tag = _utc_tag()
    report_dir = Path(args.report_dir) if args.report_dir else _ROOT / "topic_monitoring_reports" / f"smoke_{tag}"

    temp_db: Optional[Path] = None
    pipeline: TopicMonitoringPipeline

    try:
        if args.mode == "quick":
            temp_db = Path(tempfile.gettempdir()) / f"sona_monitor_smoke_{tag}_{uuid.uuid4().hex[:8]}.db"
            store = TopicMonitorSqliteStore(temp_db)
            pipeline = TopicMonitoringPipeline(db=store)
            print(f"[info] 临时 SQLite: {temp_db}", flush=True)
        elif args.mode == "local":
            if not topic_monitor_sqlite_enabled():
                print("[error] 本地 SQLite 已关闭（SONA_TOPIC_MONITOR_SQLITE=0）", flush=True)
                return 2
            db_path = resolve_topic_monitor_sqlite_path()
            print(f"[info] 使用本地库: {db_path}", flush=True)
            pipeline = TopicMonitoringPipeline()
        else:
            if not topic_monitor_sqlite_enabled():
                print("[error] netinsight 模式需要本地 SQLite 持久化", flush=True)
                return 2
            db_path = resolve_topic_monitor_sqlite_path()
            print(f"[info] 使用本地库: {db_path}", flush=True)
            ni = _check_netinsight_ready()
            print(json.dumps({"netinsight_precheck": ni}, ensure_ascii=False, indent=2))
            if not ni.get("ok"):
                print("[error] NetInsight 未就绪，见上方 netinsight_precheck", flush=True)
                return 2
            pipeline = TopicMonitoringPipeline()

        summary = _run_flow(
            pipeline,
            mode=args.mode,
            topic_id=args.topic_id or None,
            name=args.name,
            domain=args.domain,
            keywords=keywords,
            report_dir=report_dir,
            skip_create=bool(args.topic_id),
        )
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        if summary.get("ok"):
            print(f"\n[OK] {summary.get('hint')}", flush=True)
            if also_json:
                if args.mode == "quick":
                    print(
                        "[warn] --also-export-json 在 quick 模式下已跳过（临时库与项目库路径不一致）。"
                        "请改用 --mode local 或 netinsight。",
                        flush=True,
                    )
                elif topic_monitor_sqlite_enabled():
                    tid_done = str(summary.get("topic_id") or "").strip()
                    if tid_done:
                        try:
                            out_p = Path(also_json)
                            out_p = out_p if out_p.is_absolute() else (_ROOT / out_p).resolve()
                            extra = run_export_json(
                                topic_id=tid_done,
                                out_path=out_p,
                                post_limit=args.export_post_limit,
                                snapshot_limit=args.export_snapshot_limit,
                                alert_limit=args.export_alert_limit,
                            )
                            print(json.dumps({"also_export_json": extra}, ensure_ascii=False, indent=2), flush=True)
                        except Exception as exc:  # noqa: BLE001
                            print(json.dumps({"also_export_json_error": str(exc)}, ensure_ascii=False), flush=True)
            return 0
        print("\n[FAIL] 未产生有效帖子或快照，请检查上方步骤输出", flush=True)
        return 1
    except Exception as exc:  # noqa: BLE001
        print(json.dumps({"ok": False, "error": str(exc)}, ensure_ascii=False, indent=2), flush=True)
        return 1
    finally:
        if temp_db is not None and not args.keep_db and temp_db.is_file():
            try:
                temp_db.unlink()
                print(f"[info] 已删除临时库 {temp_db}", flush=True)
            except OSError as exc:
                print(f"[warn] 无法删除临时库 {temp_db}: {exc}", flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
