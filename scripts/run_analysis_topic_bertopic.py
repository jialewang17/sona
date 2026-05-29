"""测试脚本：调用 analysis_topic_bertopic 工具做 BERTopic + Qwen 主题聚类。"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import List, Optional

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from tools.data_bertopic_qwen import (
    RECLUSTER_PROMPT_YAML,
    _recluster_prompt_yaml_path,
    analysis_topic_bertopic,
)
from utils.path import ensure_task_dirs
from utils.task_context import set_task_id

DEFAULT_DATA_PATH = "sandbox/测试/过程文件/测试.csv"


def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="运行 analysis_topic_bertopic：CSV 主题聚类 + 大模型议题合并。",
    )
    parser.add_argument("--data", default=DEFAULT_DATA_PATH, help="采集后的 CSV 文件路径。")
    parser.add_argument("--task-id", default="测试", help="任务 ID（结果写入 sandbox/<task_id>/过程文件）。")
    parser.add_argument(
        "--event-intro",
        default="舆情热点事件测试：关注舆论焦点、争议议题与传播脉络。",
        help="事件背景（传入大模型做主题命名与合并）。",
    )
    parser.add_argument(
        "--content-columns",
        default="",
        help="强制内容列，逗号分隔，如：内容,摘要。留空则自动识别。",
    )
    parser.add_argument(
        "--domain-topic",
        default="sona_test",
        help="领域标识（用于可选 YAML 提示词回退路径）。",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=0,
        help="覆盖 SONA_TOPIC_BERTOPIC_MAX_ROWS；0 表示使用环境变量或默认 3000。",
    )
    return parser.parse_args(argv)


def _read_headers(csv_path: Path) -> List[str]:
    encodings = ["utf-8-sig", "utf-8", "gb18030", "gbk"]
    for enc in encodings:
        try:
            with csv_path.open("r", encoding=enc, errors="strict") as f:
                import csv as _csv

                reader = _csv.reader(f)
                header = next(reader, [])
                return [str(h) for h in header]
        except Exception:
            continue
    try:
        with csv_path.open("r", encoding="utf-8-sig", errors="replace") as f:
            import csv as _csv

            reader = _csv.reader(f)
            header = next(reader, [])
            return [str(h) for h in header]
    except Exception:
        return []


def _guess_content_columns(headers: List[str]) -> List[str]:
    normalized = [h.strip() for h in headers]
    if "内容" in normalized:
        return ["内容"]
    lower_map = {h.lower(): h for h in headers}
    for cand in ("content", "contents", "正文", "摘要", "ocr", "segment"):
        if cand in lower_map:
            return [lower_map[cand]]
    return []


def _print_topics(topics: list, *, limit: int = 12) -> None:
    if not topics:
        print("  - 议题列表: 无")
        return
    print(f"  - 议题列表（前 {min(limit, len(topics))} 个）:")
    for i, item in enumerate(topics[:limit], 1):
        if not isinstance(item, dict):
            continue
        label = item.get("label", "")
        doc_count = item.get("doc_count", 0)
        keywords = item.get("keywords") or []
        kw_preview = "、".join(str(k) for k in keywords[:8])
        desc = str(item.get("description", "") or "")[:80]
        print(f"    {i}. [{label}] 文档数={doc_count}")
        if kw_preview:
            print(f"       关键词: {kw_preview}")
        if desc:
            print(f"       描述: {desc}{'...' if len(str(item.get('description', ''))) > 80 else ''}")


def _print_statistics(statistics: dict) -> None:
    if not statistics:
        print("  - 统计信息: 无")
        return
    keys = (
        "input_rows",
        "valid_text_rows",
        "clustered_unique_texts",
        "duplicate_dropped",
        "topic_count",
        "relevant_topic_count",
        "irrelevant_topic_count",
        "content_columns",
        "truncated",
        "used_rows",
        "total_rows",
    )
    print("  - 统计信息:")
    for key in keys:
        if key in statistics:
            print(f"      {key}: {statistics[key]}")


def _print_artifacts(artifacts: dict) -> None:
    if not isinstance(artifacts, dict) or not artifacts:
        print("  - 产物目录: 无")
        return
    print("  - 产物路径:")
    for label, path in (
        ("输出目录", artifacts.get("output_dir")),
        ("再聚类 JSON", artifacts.get("recluster_json")),
        ("主题关键词 JSON", artifacts.get("recluster_keywords_json")),
        ("文档归属 CSV", artifacts.get("doc_topic_csv")),
        ("向量矩阵 NPY", artifacts.get("embedding_npy")),
        ("向量矩阵元数据", artifacts.get("embedding_meta")),
        ("过程目录向量(latest)", artifacts.get("topic_bertopic_embeddings_latest_npy")),
        ("向量缓存 NPY", artifacts.get("embedding_cache_npy")),
        ("向量缓存 JSON", artifacts.get("embedding_cache_json")),
    ):
        if not path:
            continue
        p = Path(str(path))
        status = "[OK]" if p.exists() else "[WARN] 未找到"
        print(f"      {label}: {path} {status}")


def main(argv: Optional[List[str]] = None) -> None:
    args = _parse_args(argv)

    if args.max_rows > 0:
        os.environ["SONA_TOPIC_BERTOPIC_MAX_ROWS"] = str(args.max_rows)

    task_id = str(args.task_id).strip() or "测试"
    ensure_task_dirs(task_id)
    set_task_id(task_id)

    try:
        print("=" * 80)
        print("analysis_topic_bertopic 工具测试（BERTopic + Qwen 议题合并）")
        print("=" * 80)
        prompt_path = _recluster_prompt_yaml_path()
        print(f"[提示词] 大模型再聚类阶段统一读取: {prompt_path}")
        if prompt_path.is_file():
            print(f"[提示词] [OK] 文件存在 ({RECLUSTER_PROMPT_YAML})")
        else:
            print(f"[提示词] [WARN] 文件不存在，再聚类阶段将失败: {prompt_path}")
        print(
            f"[配置] SONA_TOPIC_BERTOPIC_MAX_ROWS="
            f"{os.environ.get('SONA_TOPIC_BERTOPIC_MAX_ROWS', '3000')}"
        )

        data_file = Path(args.data)
        if not data_file.is_absolute():
            data_file = project_root / data_file

        print(f"事件介绍: {args.event_intro}")
        print(f"数据文件: {data_file}")
        print(f"任务 ID: {task_id}")
        print(f"领域标识: {args.domain_topic}")
        print("-" * 80)

        if not data_file.exists():
            print(f"[WARN] 数据文件不存在: {data_file}")
            print("   请指定 --data 指向有效的采集 CSV")
            return

        headers = _read_headers(data_file)
        if headers:
            print(f"[表头预览] {headers}")
        else:
            print("[表头预览] 读取失败")

        content_columns: List[str] = []
        if args.content_columns.strip():
            content_columns = [c.strip() for c in args.content_columns.split(",") if c.strip()]
        else:
            content_columns = _guess_content_columns(headers)

        if content_columns:
            print(f"[内容列] 使用: {content_columns}")
        else:
            print("[内容列] 未命中常见列名，交由工具自动识别")

        invoke_params = {
            "eventIntroduction": args.event_intro,
            "dataFilePath": str(data_file),
            "domainTopic": args.domain_topic,
        }
        if content_columns:
            invoke_params["contentColumns"] = content_columns

        try:
            raw = analysis_topic_bertopic.invoke(invoke_params)
        except Exception as exc:
            print(f"[ERR] 调用失败: {exc}")
            import traceback

            traceback.print_exc()
            return

        if not isinstance(raw, str):
            print("[WARN] 返回非字符串:")
            print(raw)
            return

        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            print("[WARN] 返回结果不是有效 JSON:")
            print(raw)
            return

        if parsed.get("error"):
            print(f"[ERR] {parsed['error']}")
            if parsed.get("save_error"):
                print(f"       保存: {parsed['save_error']}")
            return

        print("\n[OK] 分析完成")
        print("\n摘要:")
        if parsed.get("message"):
            print(f"  {parsed['message']}")

        statistics = parsed.get("statistics") if isinstance(parsed.get("statistics"), dict) else {}
        _print_statistics(statistics)

        topics = parsed.get("topics") if isinstance(parsed.get("topics"), list) else []
        _print_topics(topics)

        artifacts = parsed.get("artifacts") if isinstance(parsed.get("artifacts"), dict) else {}
        _print_artifacts(artifacts)

        result_file_path = str(parsed.get("result_file_path") or "").strip()
        if result_file_path:
            result_file = Path(result_file_path)
            print(f"\n  - 结果文件: {result_file_path}")
            if result_file.exists():
                print(f"  [OK] 已保存，大小 {result_file.stat().st_size} 字节")
            else:
                print("  [WARN] 路径已返回但文件不存在")
        else:
            print("\n  [WARN] 未返回 result_file_path")

        print("\n" + "=" * 80)
        print("[OK] 测试结束")
    finally:
        set_task_id(None)


if __name__ == "__main__":
    main()
