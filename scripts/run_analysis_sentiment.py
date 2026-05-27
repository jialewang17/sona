"""测试脚本：调用 analysis_sentiment 工具分析情感倾向（0～10 分 + 六类细粒度情绪）。"""

from __future__ import annotations

import json
import sys
from pathlib import Path
import os

# 添加项目根目录到 Python 路径
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from tools.analysis_sentiment import FINE_EMOTIONS, analysis_sentiment
from utils.path import ensure_task_dirs
from utils.task_context import set_task_id


def _print_emotion_block(statistics: dict) -> None:
    """打印 statistics 中的细粒度情绪分布（与工具返回字段对齐）。"""
    counts = statistics.get("emotion_counts")
    ratios = statistics.get("emotion_ratios")
    if not isinstance(counts, dict):
        print("  - 细粒度情绪: statistics 中无 emotion_counts（可能为旧版工具）")
        return
    print("  - 细粒度情绪计数 emotion_counts:")
    for name in list(FINE_EMOTIONS) + ["其他"]:
        c = counts.get(name, 0)
        r = (ratios or {}).get(name) if isinstance(ratios, dict) else None
        if r is not None:
            print(f"      {name}: {c} ({float(r) * 100:.2f}%)")
        else:
            print(f"      {name}: {c}")


def _preview_row_scores_from_file(result_file: Path, *, limit: int = 5) -> None:
    """从完整结果 JSON 中抽样打印 row_scores（含 emotion）。"""
    try:
        text = result_file.read_text(encoding="utf-8", errors="replace")
        data = json.loads(text)
    except Exception as e:
        print(f"  [WARN] 无法读取完整结果用于 row_scores 预览: {e}")
        return
    rows = data.get("row_scores")
    if not isinstance(rows, list) or not rows:
        print("  - row_scores: 无或格式异常")
        return
    print(f"  - row_scores 抽样（前 {limit} 条，含 emotion）:")
    shown = 0
    for item in rows:
        if shown >= limit:
            break
        if not isinstance(item, dict):
            continue
        idx = item.get("row_index")
        score = item.get("score")
        emo = item.get("emotion")
        preview = str(item.get("text_preview") or "")[:60]
        print(f"      row={idx} score={score} emotion={emo} preview={preview!r}")
        shown += 1
    fine_set = set(FINE_EMOTIONS)
    bad = 0
    for item in rows:
        if not isinstance(item, dict):
            continue
        s, emo = item.get("score"), item.get("emotion")
        if s is None:
            continue
        if emo not in fine_set:
            bad += 1
    if bad:
        print(f"  [WARN] row_scores 中有 {bad} 条 score 非空但 emotion 不在六字集合内")
    else:
        print("  [OK] row_scores 中已打分行的 emotion 字段均在预期集合内")


def main() -> None:
    """主函数：执行情感倾向分析测试。"""
    # 编码探测辅助：用常见编码读取表头，便于决定 contentColumns
    def _read_headers(csv_path: Path) -> list[str]:
        encodings = ["utf-8-sig", "utf-8", "gb18030", "gbk"]
        for enc in encodings:
            try:
                with csv_path.open("r", encoding=enc, errors="strict") as f:
                    import csv as _csv  # 局部导入，避免脚本顶层依赖
                    reader = _csv.reader(f)
                    header = next(reader, [])
                    return [str(h) for h in header]
            except Exception:
                continue
        # 最差回退（可能有替换字符）
        try:
            with csv_path.open("r", encoding="utf-8-sig", errors="replace") as f:
                import csv as _csv
                reader = _csv.reader(f)
                header = next(reader, [])
                return [str(h) for h in header]
        except Exception:
            return []
    # 设置任务上下文
    task_id = "测试"
    ensure_task_dirs(task_id)
    set_task_id(task_id)
    
    try:
        # 测试配置示例
        test_configs = [
            {
                "eventIntroduction": "美伊战争相关舆情事件，关注军事行动、外交谈判破裂、地区外溢及国际反应。",
                "dataFilePath": "sandbox/测试/过程文件/测试.csv",
                "retryContext": None
            }
        ]
        
        print("=" * 80)
        print("analysis_sentiment 工具测试（含愤怒/焦虑/质疑/同情/嘲讽/支持）")
        print("=" * 80)

        # 并发相关默认参数（可根据本机/限流情况调整）
        os.environ.setdefault("SONA_SENTIMENT_BATCH_PARALLEL_WORKERS", "4")
        os.environ.setdefault("SONA_SENTIMENT_BATCH_JITTER_MS", "50")
        print(f"[并发] 批次并发: {os.environ.get('SONA_SENTIMENT_BATCH_PARALLEL_WORKERS')} | 抖动(ms): {os.environ.get('SONA_SENTIMENT_BATCH_JITTER_MS')}")
        
        for i, config in enumerate(test_configs, 1):
            print(f"\n[测试 {i}/{len(test_configs)}]")
            print(f"事件介绍: {config['eventIntroduction']}")
            print(f"数据文件: {config['dataFilePath']}")
            print(f"重试上下文: {'无' if not config.get('retryContext') else '有'}")
            print("-" * 80)
            
            # 检查文件是否存在
            data_file = Path(config['dataFilePath'])
            if not data_file.exists():
                print(f"[WARN] 数据文件不存在: {config['dataFilePath']}")
                print("   请确保文件存在后再运行测试")
                continue
            
            # 打印表头（多编码尝试），并据此自动决定 contentColumns
            headers = _read_headers(data_file)
            if headers:
                print(f"[表头预览] {headers}")
            else:
                print("[表头预览] 读取失败（可能是编码或文件问题）")

            # 优先精确匹配“内容”，否则回退到常见候选命中第一个
            forced_content_cols: list[str] = []
            normalized_headers = [h.strip() for h in headers]
            if "内容" in normalized_headers:
                forced_content_cols = ["内容"]
            else:
                for cand in ["content", "contents", "正文", "摘要", "ocr", "segment"]:
                    if cand in [h.lower() for h in normalized_headers]:
                        # 还原原始大小写（取第一个命中）
                        idx = [h.lower() for h in normalized_headers].index(cand)
                        forced_content_cols = [headers[idx]]
                        break
            if forced_content_cols:
                print(f"[内容列] 使用指定列: {forced_content_cols}")
            else:
                print("[内容列] 未在表头中命中常见内容列，交由工具自动识别")

            try:
                # 调用工具
                invoke_params = {
                    "eventIntroduction": config["eventIntroduction"],
                    "dataFilePath": config["dataFilePath"],
                    # 若探测到内容列则强制传入，否则交由工具自动识别
                }
                if forced_content_cols:
                    invoke_params["contentColumns"] = forced_content_cols
                if config.get("retryContext"):
                    invoke_params["retryContext"] = config["retryContext"]
                
                result = analysis_sentiment.invoke(invoke_params)
                
                # 解析并打印结果
                if isinstance(result, str):
                    try:
                        parsed = json.loads(result)
                        
                        # 检查是否有错误
                        if "error" in parsed:
                            print(f"[ERR] 错误: {parsed['error']}")
                            continue
                        
                        print("\n[OK] 分析成功！")
                        print("\n结果详情:")
                        print(json.dumps(parsed, ensure_ascii=False, indent=2))
                        
                        # 验证关键字段
                        print("\n字段验证:")
                        statistics = parsed.get("statistics", {})
                        if statistics:
                            total = int(statistics.get("total", 0) or 0)
                            positive_count = int(statistics.get("positive_count", 0) or 0)
                            negative_count = int(statistics.get("negative_count", 0) or 0)
                            neutral_count = int(statistics.get("neutral_count", 0) or 0)
                            positive_ratio = float(statistics.get("positive_ratio", 0.0) or 0.0)
                            negative_ratio = float(statistics.get("negative_ratio", 0.0) or 0.0)
                            neutral_ratio = float(statistics.get("neutral_ratio", 0.0) or 0.0)
                            avg_score_analyzed = statistics.get("avg_score_analyzed", None)
                            score_scale = statistics.get("score_scale", "")

                            print(f"  - 总数据量: {total}")
                            print(f"  - 正面数量: {positive_count} ({positive_ratio * 100:.2f}%)")
                            print(f"  - 负面数量: {negative_count} ({negative_ratio * 100:.2f}%)")
                            print(f"  - 中性数量: {neutral_count} ({neutral_ratio * 100:.2f}%)")
                            if avg_score_analyzed is not None:
                                print(f"  - 已分析平均分: {avg_score_analyzed}")
                            if score_scale:
                                print(f"  - {score_scale}")
                            _print_emotion_block(statistics)
                        else:
                            print("  - 统计信息: 无")

                        content_columns = parsed.get("content_columns", [])
                        if content_columns:
                            print(f"  - 命中的内容列: {', '.join(content_columns)}")

                        scoring_model = parsed.get("scoring_model", "")
                        scoring_profile = parsed.get("scoring_profile", "")
                        if scoring_model or scoring_profile:
                            print(f"  - 打分模型/配置: {scoring_model} / {scoring_profile}")
                        
                        positive_summary = parsed.get("positive_summary", [])
                        if positive_summary:
                            print(f"  - 正面观点数量: {len(positive_summary)}")
                            print(f"  - 正面观点:")
                            for idx, view in enumerate(positive_summary[:3], 1):
                                print(f"    {idx}. {view}")
                        else:
                            print("  - 正面观点: 无")
                        
                        negative_summary = parsed.get("negative_summary", [])
                        if negative_summary:
                            print(f"  - 负面观点数量: {len(negative_summary)}")
                            print(f"  - 负面观点:")
                            for idx, view in enumerate(negative_summary[:3], 1):
                                print(f"    {idx}. {view}")
                        else:
                            print("  - 负面观点: 无")
                        
                        # 验证结果文件路径
                        result_file_path = parsed.get("result_file_path", "")
                        if result_file_path:
                            print(f"  - 结果文件路径: {result_file_path}")
                            result_file = Path(result_file_path)
                            if result_file.exists():
                                print(f"  [OK] 结果文件已保存: {result_file_path}")
                                print(f"  - 文件大小: {result_file.stat().st_size} 字节")
                                _preview_row_scores_from_file(result_file, limit=5)
                            else:
                                print(f"  [WARN] 结果文件路径存在但文件未找到: {result_file_path}")
                        else:
                            print("  [WARN] 未返回结果文件路径")

                        coarse_pie = parsed.get("coarse_pie_chart_path", "")
                        fine_pie = parsed.get("fine_emotion_pie_chart_path", "")
                        if coarse_pie or fine_pie:
                            print("  - 饼图数据文件:")
                            for label, p in (
                                ("正/负/中立", coarse_pie),
                                ("六类细粒度情绪", fine_pie),
                            ):
                                if not p:
                                    print(f"      {label}: (未返回路径)")
                                    continue
                                pf = Path(p)
                                ok = pf.exists()
                                print(f"      {label}: {p} {'[OK]' if ok else '[WARN] 未找到文件'}")
                        
                    except json.JSONDecodeError:
                        print("[WARN] 返回结果不是有效的 JSON:")
                        print(result)
                else:
                    print("返回结果:")
                    print(result)
                    
            except Exception as e:
                print(f"[ERR] 错误: {str(e)}")
                import traceback
                traceback.print_exc()
            
            print("\n" + "=" * 80)
        
        print("\n[OK] 测试完成！")
    finally:
        # 清理任务上下文
        set_task_id(None)


if __name__ == "__main__":
    main()
