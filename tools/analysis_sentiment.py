"""情感倾向分析工具：对内容列做与关键词一致的清洗后，使用 qwen-plus 打 0-10 分并输出六类细粒度情绪，再汇总。"""

from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import time

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool

from model.factory import get_sentiment_model
from tools._csv_io import read_csv_rows_all
from tools.keyword_stats import CONTENT_COLUMN_KEYWORDS, _identify_content_columns
from utils.content_text import clean_text_like_keyword_stats
from utils.path import ensure_task_dirs, get_task_process_dir
from utils.prompt_loader import get_analysis_sentiment_prompt
from utils.task_context import get_task_id

_BATCH_SIZE: int = 12
_MAX_CHARS_PER_TEXT: int = 2800

# 并发配置（通过环境变量可调）
import os  # noqa: E402
import random  # noqa: E402
import threading  # noqa: E402
from concurrent.futures import ThreadPoolExecutor, as_completed  # noqa: E402

def _env_int(name: str, default: int, low: int, high: int) -> int:
    raw = str(os.environ.get(name, str(default))).strip()
    try:
        val = int(raw)
    except Exception:
        val = default
    return max(low, min(high, val))

_SENTIMENT_BATCH_PARALLEL_WORKERS: int = _env_int("SONA_SENTIMENT_BATCH_PARALLEL_WORKERS", 4, 1, 8)
_SENTIMENT_BATCH_JITTER_MS: int = _env_int("SONA_SENTIMENT_BATCH_JITTER_MS", 100, 0, 1000)
_SENTIMENT_DYNAMIC_BATCH_SIZE: int = _env_int("SONA_SENTIMENT_BATCH_SIZE", 40, 4, 64)
_SENTIMENT_BATCH_RETRIES: int = _env_int("SONA_SENTIMENT_BATCH_RETRIES", 1, 0, 3)
_SENTIMENT_BATCH_TIMEOUT_SEC: int = _env_int("SONA_SENTIMENT_BATCH_TIMEOUT_SEC", 25, 5, 120)
_SENTIMENT_MAX_WALLTIME_SEC: int = _env_int("SONA_SENTIMENT_MAX_WALLTIME_SEC", 600, 15, 1800)
_SENTIMENT_MAX_ROWS: int = _env_int("SONA_SENTIMENT_MAX_ROWS", 2000, 50, 200000)
_SENTIMENT_EXAMPLES_MAX_CHARS: int = _env_int("SONA_SENTIMENT_EXAMPLES_MAX_CHARS", 4000, 500, 20000)


def _allow_sentiment_column_fallback() -> bool:
    """默认关闭：不复用 CSV「情感」列、不在 LLM 全失败时回退到列统计。"""
    return str(os.environ.get("SONA_SENTIMENT_ALLOW_COLUMN_FALLBACK", "0")).strip().lower() in {
        "1",
        "true",
        "yes",
        "y",
        "on",
    }


def sentiment_column_fallback_enabled() -> bool:
    """是否允许使用 CSV「情感」列或在流水线层做列统计兜底（供 workflow 调用）。"""
    return _allow_sentiment_column_fallback()


# 路径 B（LLM 打分）细粒度情绪：须与 score 语义自洽（模型在单条内同时输出）
FINE_EMOTIONS: Tuple[str, ...] = ("愤怒", "焦虑", "质疑", "同情", "嘲讽", "支持")


_SCORE_SYSTEM_PROMPT = """你是舆情情感分析助手。结合「事件背景」，对每条文本同时给出：
1) 整数情感分 score，范围 0～10（0 最负面，10 最正面；0～3 偏负面，4～6 偏中立，7～10 偏正面）
2) 细粒度情绪 emotion，只能从以下 6 个词中选且必须完全一致（不要加引号外的修饰）：
愤怒、焦虑、质疑、同情、嘲讽、支持

## 六类情绪（必须按「文本真实态度」选，不要凭体裁或是否有问号猜测）

- **愤怒**：明显气愤、抗议、辱骂、强烈谴责、要讨说法（带攻击性或激动指责）。
- **焦虑**：担心、害怕、不安、纠结、怕踩坑、怕错过、对结果不确定；也可覆盖**无明显褒贬的诉求/诉苦式碎碎念**（仍偏负面或压力感）。
- **质疑**：对事实、公正、官方/对方动机的不信任；要求解释、认为双标/造假/甩锅。**仅有「求换/收/出」或标价而无不信任语义，不算质疑。**
- **同情**：对他人**不幸、受损、被误解、遭遇不公**的共情、心疼、站弱者。**抓到稀有道具、晒欧、集齐毕业、心情好，一律不是同情**（那是满意/高兴 → 见「支持」）。
- **嘲讽**：反话、阴阳怪气、梗化贬损、奚落；**纯广告/明码标价转让，不是嘲讽**。
- **支持**：明确点赞、叫好、认同、感谢、愿意安利/捍卫；或对体验**满意、开心、祝贺**等偏正面立场。

## 易错纠正（务必遵守）

1. **交易/代练/资源互换/明码标价（如含「r」「代练」「出蛋」「换蛋」「走鱼」等）**：若无对平台/他人的指责，不要用「质疑/嘲讽/同情」；更常见是 **焦虑**（怕被骗、怕错过）或 **支持**（满意、推荐），请结合语气与 score。
2. **问号**：问价、求组队、求换物 ≠ 质疑；质疑须含「不信、认为不公、要求解释」等语义。
3. **emotion 与 score 大致同向**：高分（8～10）却标「愤怒/质疑/嘲讽」时，仅当正文确有强烈贬损或对抗；否则应改为更正面的一类（多为「支持」）。低分（0～3）却标「支持」时，须有明确称赞或捍卫，否则应改为「愤怒/质疑」等。

必须只输出一个 JSON 对象，不要 markdown 代码块，格式严格为：
{"items":[{"row":<整数行号>,"score":<0到10的整数>,"emotion":"<六选一>"}, ...]}

要求：
- items 长度与输入条数一致，每个 row 与输入中的 row 一致
- score 为整数；emotion 为上述六字之一，不得自创类别
"""


def _load_sentiment_examples_text() -> str:
    use_examples = str(os.environ.get("SONA_SENTIMENT_USE_EXAMPLES", "0")).strip().lower() in {
        "1",
        "true",
        "yes",
        "y",
        "on",
    }
    if not use_examples:
        return ""
    custom_path = str(os.environ.get("SONA_SENTIMENT_EXAMPLES_PATH", "")).strip()
    if custom_path:
        p = Path(custom_path).expanduser()
    else:
        p = Path(__file__).resolve().parents[1] / "prompt" / "sentiment_examples_zh_v1.md"
    if not p.exists() or not p.is_file():
        return ""
    try:
        text = p.read_text(encoding="utf-8")
    except Exception:
        return ""
    text = text.strip()
    if not text:
        return ""
    if len(text) > _SENTIMENT_EXAMPLES_MAX_CHARS:
        text = text[:_SENTIMENT_EXAMPLES_MAX_CHARS] + "\n...\n"
    return text


_SENTIMENT_EXAMPLES_TEXT: str = _load_sentiment_examples_text()


class _RequestMetrics:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.started = 0
        self.succeeded = 0
        self.failed = 0
        self.active = 0
        self.max_active = 0

    def on_start(self) -> None:
        with self._lock:
            self.started += 1
            self.active += 1
            if self.active > self.max_active:
                self.max_active = self.active

    def on_end(self, success: bool) -> None:
        with self._lock:
            self.active = max(0, self.active - 1)
            if success:
                self.succeeded += 1
            else:
                self.failed += 1

    def summary(self, *, elapsed_sec: float, rows_scored: int) -> Dict[str, Any]:
        elapsed = max(0.001, elapsed_sec)
        qps = rows_scored / elapsed
        qpm = qps * 60.0
        rpm = self.started / elapsed * 60.0
        return {
            "elapsed_sec": round(elapsed_sec, 3),
            "rows_scored": rows_scored,
            "requests_started": self.started,
            "requests_succeeded": self.succeeded,
            "requests_failed": self.failed,
            "qps": round(qps, 3),
            "qpm": round(qpm, 2),
            "rpm": round(rpm, 2),
            "max_concurrent_connections": self.max_active,
        }





def _identify_sentiment_column(data: List[Dict[str, Any]]) -> Optional[str]:
    if not data:
        return None
    sentiment_candidates = (
        "情感",
        "情感倾向",
        "情感分析",
        "情感分类",
        "情感标签",
        "sentiment",
        "emotion",
    )
    for col in data[0].keys():
        col_lower = str(col).lower()
        if any(key in col_lower for key in ("sentiment", "emotion")):
            return col
        if any(key in str(col) for key in ("情感", "倾向")):
            return col
        if any(key in str(col) for key in sentiment_candidates):
            return col
    return None


def _normalize_sentiment_label(value: Any) -> Optional[str]:
    raw = str(value or "").strip().lower()
    if not raw:
        return None
    mapping = {
        "正面": "正面",
        "积极": "正面",
        "positive": "正面",
        "pos": "正面",
        "1": "正面",
        "负面": "负面",
        "消极": "负面",
        "negative": "负面",
        "neg": "负面",
        "-1": "负面",
        "中性": "中立",
        "中立": "中立",
        "neutral": "中立",
        "0": "中立",
    }
    if raw in mapping:
        return mapping[raw]
    if raw in {"p", "n"}:
        return "正面" if raw == "p" else "负面"
    return None


def _label_to_score(label: str) -> int:
    if label == "正面":
        return 8
    if label == "负面":
        return 2
    return 5


def _score_to_coarse_label(score: Optional[int]) -> Optional[str]:
    if score is None:
        return None
    try:
        s = int(score)
    except Exception:
        return None
    if s <= 3:
        return "负面"
    if s <= 6:
        return "中立"
    return "正面"


def _should_use_existing_sentiment(
    data: List[Dict[str, Any]],
    sentiment_col: Optional[str],
) -> bool:
    if not sentiment_col or not data:
        return False
    non_empty = 0
    recognizable = 0
    for row in data:
        raw = row.get(sentiment_col, "")
        if str(raw or "").strip():
            non_empty += 1
            if _normalize_sentiment_label(raw) is not None:
                recognizable += 1
    if non_empty == 0:
        return False
    ratio = recognizable / non_empty
    min_non_empty = min(len(data), 20)
    return non_empty >= min_non_empty and ratio >= 0.6


def _to_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    s = str(value).strip().lower()
    if not s:
        return default
    if s in {"1", "true", "yes", "y", "on"}:
        return True
    if s in {"0", "false", "no", "n", "off"}:
        return False
    return default


def _row_raw_content_text(row: Dict[str, Any], content_columns: List[str]) -> str:
    parts: List[str] = []
    for col in content_columns:
        val = row.get(col, "")
        if val is None:
            continue
        text = str(val).strip()
        if text:
            parts.append(text)
    return " ".join(parts)


def _row_cleaned_content(row: Dict[str, Any], content_columns: List[str]) -> str:
    return clean_text_like_keyword_stats(_row_raw_content_text(row, content_columns))


def _clamp_score(value: Any) -> int:
    try:
        if isinstance(value, bool):
            return 5
        if isinstance(value, str):
            value = value.strip()
            m = re.search(r"-?\d+", value)
            if m:
                value = int(m.group())
            else:
                return 5
        n = int(round(float(value)))
        return max(0, min(10, n))
    except (TypeError, ValueError):
        return 5


def _label_from_score(score: int) -> str:
    if score <= 3:
        return "负面"
    if score <= 6:
        return "中立"
    return "正面"


def _emotion_from_score(score: int) -> str:
    """LLM 未返回合法 emotion 时，按 score 映射到六类之一（兜底）。"""
    if score <= 2:
        return "愤怒"
    if score <= 4:
        return "质疑"
    if score <= 6:
        return "焦虑"
    if score <= 8:
        return "同情"
    return "支持"


def _normalize_emotion(value: Any, *, score: int) -> str:
    """将模型输出或同义词归一到 FINE_EMOTIONS；失败则用 score 兜底。"""
    s = str(value or "").strip()
    if s in FINE_EMOTIONS:
        return s
    syn: Dict[str, str] = {
        "讽刺": "嘲讽",
        "反讽": "嘲讽",
        "阴阳怪气": "嘲讽",
        "谩骂": "愤怒",
        "气愤": "愤怒",
        "担心": "焦虑",
        "忧虑": "焦虑",
        "声援": "支持",
        "力挺": "支持",
        "共情": "同情",
        "体谅": "同情",
        "追问": "质疑",
        "怀疑": "质疑",
    }
    if s in syn:
        return syn[s]
    return _emotion_from_score(score)


# 细粒度情绪纠偏：缓解模型将「交易广告 / 晒欧」误标为质疑、同情、嘲讽等
_RE_TRADE_OR_SERVICE = re.compile(
    r"(?:代练|代肝|陪玩|陪打|白菜价|价格表|r接|接\s*r|\d+\s*r|可走鱼|走闲鱼|闲鱼)"
    r"|(?:出蛋|换蛋|选蛋|收蛋|求蛋|出闪|换闪|代抓|包出)",
    re.I,
)
_RE_POSITIVE_EXPERIENCE = re.compile(
    r"(?:闪(?:光)?|欧(?:气)?|毕业|集齐|圆满|抓(?:到|获)|运气(?:好)?|太(?:好|棒)了|"
    r"开心|满意|祝贺|恭喜|舒服了|爽到)",
)
_RE_DISTRUST_OR_SLAM = re.compile(
    r"(?:误封|不公|质疑|凭什么|为何|为啥|造假|欺骗|甩锅|装死|装瞎|"
    r"骗人|糊弄|韭菜|封号|辣鸡|垃圾|坑爹|该死|恶心|气死|退钱)",
    re.I,
)


def _nudge_emotion_for_text(cleaned: str, score: int, emotion: str) -> str:
    """
    在 LLM 结果上做保守纠偏，减少「体裁误判」（交易贴标质疑、晒欧标同情等）。
    不覆盖明显含贬损/不信任语义的样本。
    """
    if emotion not in FINE_EMOTIONS:
        return emotion
    t = str(cleaned or "").strip()
    if len(t) < 4:
        return emotion
    sc = int(_clamp_score(score))

    slam = bool(_RE_DISTRUST_OR_SLAM.search(t))
    trade_like = bool(_RE_TRADE_OR_SERVICE.search(t))
    pos_exp = bool(_RE_POSITIVE_EXPERIENCE.search(t))

    # 晒欧、好事：不是「同情」
    if emotion == "同情" and pos_exp and not slam:
        return "支持" if sc >= 5 else _emotion_from_score(sc)

    # 交易/服务味浓且无指责：质疑/嘲讽/同情 多为误标
    if emotion in ("质疑", "同情", "嘲讽") and trade_like and not slam:
        return "支持" if sc >= 7 else "焦虑"

    # 高分 + 强负面情绪标签：若无贬损词，多为漂移 → 支持
    if sc >= 8 and emotion in ("愤怒", "质疑", "嘲讽") and not slam:
        return "支持"

    # 低分却「支持/同情」：有贬损时拉回到质疑或按分兜底
    if sc <= 3 and emotion in ("支持", "同情") and slam:
        return "质疑" if emotion == "支持" else _emotion_from_score(sc)

    return emotion


def _emotion_from_coarse_label(label: str) -> str:
    """路径 A：仅有正/负/中立标签时映射到细粒度（粗粒度近似）。"""
    if label == "正面":
        return "支持"
    if label == "负面":
        return "质疑"
    return "焦虑"


def _parse_score_json(text: str) -> Dict[str, Any]:
    json_match = re.search(r"\{[\s\S]*\}", text)
    raw = json_match.group() if json_match else text
    return json.loads(raw)


def _score_batch(
    model: Any,
    *,
    event_introduction: str,
    batch: List[Tuple[int, str]],
    metrics: Optional[_RequestMetrics] = None,
) -> Dict[int, Dict[str, Any]]:
    """返回 row_index -> {\"score\": int, \"emotion\": str}（六类细粒度情绪）。"""
    payload = [{"row": idx, "text": txt[:_MAX_CHARS_PER_TEXT]} for idx, txt in batch]
    human = (
        f"事件背景：\n{event_introduction}\n\n"
        f"请为下列每条文本打分（JSON 中的 row 必须与下列 row 一致）：\n"
        f"{json.dumps(payload, ensure_ascii=False)}"
    )
    system_prompt = _SCORE_SYSTEM_PROMPT
    if _SENTIMENT_EXAMPLES_TEXT:
        system_prompt += (
            "\n\n以下是中文舆情情感标注示例与规则，请参考其口径完成打分：\n"
            + _SENTIMENT_EXAMPLES_TEXT
        )
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=human),
    ]
    last_error: Optional[Exception] = None
    response: Any = None
    default_row = {"score": 5, "emotion": _emotion_from_score(5)}
    for attempt in range(_SENTIMENT_BATCH_RETRIES + 1):
        if metrics:
            metrics.on_start()
        try:
            # Hard timeout guard: prevent one batch from hanging forever.
            _pool = ThreadPoolExecutor(max_workers=1)
            _fut = _pool.submit(model.invoke, messages)
            try:
                response = _fut.result(timeout=max(1, _SENTIMENT_BATCH_TIMEOUT_SEC))
            finally:
                # 关键：超时后不等待线程，避免单批次“表面超时，实际阻塞”。
                _pool.shutdown(wait=False, cancel_futures=True)
            if metrics:
                metrics.on_end(True)
            last_error = None
            break
        except Exception as e:
            last_error = e
            if metrics:
                metrics.on_end(False)
            if attempt < _SENTIMENT_BATCH_RETRIES:
                # 指数退避，缓解瞬时限流
                threading.Event().wait(min(1.5, 0.25 * (2**attempt)))
            continue
    if response is None and last_error is not None:
        return {idx: dict(default_row) for idx, _ in batch}
    result_text = response.content if hasattr(response, "content") else str(response)
    out: Dict[int, Dict[str, Any]] = {}
    try:
        obj = _parse_score_json(result_text)
        items = obj.get("items")
        if not isinstance(items, list):
            raise ValueError("missing items")
        for it in items:
            if not isinstance(it, dict):
                continue
            r = it.get("row")
            s = it.get("score")
            if r is None:
                continue
            try:
                ri = int(r)
            except (TypeError, ValueError):
                continue
            sc = _clamp_score(s)
            em = _normalize_emotion(it.get("emotion"), score=sc)
            txt_for_row = ""
            for jx, jt in batch:
                if jx == ri:
                    txt_for_row = str(jt or "")
                    break
            em = _nudge_emotion_for_text(txt_for_row, sc, em)
            out[ri] = {"score": sc, "emotion": em}
    except Exception:
        pass
    for idx, _ in batch:
        if idx not in out:
            out[idx] = dict(default_row)
    return out


def _score_batch_worker(
    *, event_introduction: str, batch: List[Tuple[int, str]], metrics: Optional[_RequestMetrics] = None
) -> Dict[int, Dict[str, Any]]:
    # 轻微抖动，降低并发瞬时触发限流
    if _SENTIMENT_BATCH_JITTER_MS > 0:
        try:
            time_to_sleep = random.random() * (_SENTIMENT_BATCH_JITTER_MS / 1000.0)
            if time_to_sleep > 0:
                threading.Event().wait(time_to_sleep)
        except Exception:
            pass
    # 每个线程内独立获取模型实例（model.factory 内部已做线程局部缓存）
    # 关键：获取模型本身也可能卡住，因此做硬超时兜底。
    try:
        _pool = ThreadPoolExecutor(max_workers=1)
        _fut = _pool.submit(get_sentiment_model)
        try:
            model = _fut.result(timeout=max(1, _SENTIMENT_BATCH_TIMEOUT_SEC))
        finally:
            _pool.shutdown(wait=False, cancel_futures=True)
    except Exception:
        dr = {"score": 5, "emotion": _emotion_from_score(5)}
        return {idx: dict(dr) for idx, _ in batch}
    try:
        return _score_batch(model, event_introduction=event_introduction, batch=batch, metrics=metrics)
    except Exception:
        # 兜底：该批次统一给中立分 + 焦虑情绪，保证流程不中断
        dr = {"score": 5, "emotion": _emotion_from_score(5)}
        return {idx: dict(dr) for idx, _ in batch}


def _build_statistics(
    total: int,
    scores_by_row: Dict[int, Optional[int]],
) -> Dict[str, Any]:
    positive_count = 0
    negative_count = 0
    neutral_count = 0
    analyzed_scores: List[int] = []

    for i in range(total):
        s = scores_by_row.get(i)
        if s is None:
            neutral_count += 1
            continue
        analyzed_scores.append(s)
        lab = _label_from_score(s)
        if lab == "正面":
            positive_count += 1
        elif lab == "负面":
            negative_count += 1
        else:
            neutral_count += 1

    avg_score: Optional[float] = None
    if analyzed_scores:
        avg_score = round(sum(analyzed_scores) / len(analyzed_scores), 4)

    return {
        "total": total,
        "positive_count": positive_count,
        "negative_count": negative_count,
        "neutral_count": neutral_count,
        "positive_ratio": round(positive_count / total, 4) if total else 0.0,
        "negative_ratio": round(negative_count / total, 4) if total else 0.0,
        "neutral_ratio": round(neutral_count / total, 4) if total else 0.0,
        "avg_score_analyzed": avg_score,
        "score_scale": "0-10（0 最低，10 最高；0-3 负面，4-6 中立，7-10 正面）",
    }


def _emotion_counts_from_pairs(rows: List[Tuple[Optional[int], Optional[str]]]) -> Dict[str, Any]:
    """按行 (score, emotion) 统计 FINE_EMOTIONS 及「其他」。"""
    counts: Dict[str, int] = {e: 0 for e in FINE_EMOTIONS}
    other = 0
    for s, em in rows:
        if em in FINE_EMOTIONS:
            counts[em] += 1
            continue
        if em:
            other += 1
            continue
        if s is not None:
            counts[_emotion_from_score(int(s))] += 1
        else:
            other += 1
    out: Dict[str, int] = {**counts, "其他": other}
    tot = len(rows)
    ratios = {k: round(v / tot, 4) if tot else 0.0 for k, v in out.items()}
    return {"emotion_counts": out, "emotion_ratios": ratios}


def _build_pie_coarse_payload(statistics: Dict[str, Any]) -> Dict[str, Any]:
    """供报告饼图：正/负/中立三维占比（与 statistics 中 count、ratio 一致）。"""
    total = int(statistics.get("total") or 0)
    return {
        "schema_version": 1,
        "chart_type": "pie_sentiment_coarse",
        "title": "情感倾向（正面 / 负面 / 中立）",
        "description": (
            "基于行级 0～10 分映射：0～3 负面，4～6 中立，7～10 正面；"
            "中立含清洗后无有效正文（score 为空）的行。"
        ),
        "total_rows": total,
        "slices": [
            {
                "label": "正面",
                "count": int(statistics.get("positive_count", 0) or 0),
                "ratio": float(statistics.get("positive_ratio", 0.0) or 0.0),
            },
            {
                "label": "负面",
                "count": int(statistics.get("negative_count", 0) or 0),
                "ratio": float(statistics.get("negative_ratio", 0.0) or 0.0),
            },
            {
                "label": "中立",
                "count": int(statistics.get("neutral_count", 0) or 0),
                "ratio": float(statistics.get("neutral_ratio", 0.0) or 0.0),
            },
        ],
    }


def _build_pie_fine_emotion_payload(
    statistics: Dict[str, Any],
    typical_expressions: Optional[Dict[str, List[str]]] = None,
) -> Dict[str, Any]:
    """供报告饼图：六类细粒度情绪 + 「其他」占比；可选 typical_expressions 供 HTML 模块化展示。"""
    total = int(statistics.get("total") or 0)
    counts = statistics.get("emotion_counts") if isinstance(statistics.get("emotion_counts"), dict) else {}
    ratios = statistics.get("emotion_ratios") if isinstance(statistics.get("emotion_ratios"), dict) else {}
    slices: List[Dict[str, Any]] = []
    for name in list(FINE_EMOTIONS) + ["其他"]:
        slices.append(
            {
                "label": name,
                "count": int(counts.get(name, 0) or 0),
                "ratio": float(ratios.get(name, 0.0) or 0.0),
            }
        )
    tex: Dict[str, List[str]] = {}
    if isinstance(typical_expressions, dict):
        for k in list(FINE_EMOTIONS) + ["其他"]:
            v = typical_expressions.get(k)
            if isinstance(v, list):
                tex[str(k)] = [str(x).strip() for x in v if str(x).strip()][:8]
            else:
                tex[str(k)] = []
    return {
        "schema_version": 1,
        "chart_type": "pie_emotion_fine",
        "title": "细粒度情绪（愤怒 / 焦虑 / 质疑 / 同情 / 嘲讽 / 支持 / 其他）",
        "description": (
            "「其他」含情绪字段缺失或与六字表不一致的样本；无正文行通常计入「其他」或随 score 兜底映射。"
        ),
        "total_rows": total,
        "slices": slices,
        "typical_expressions": tex,
    }


def _compute_agreement_with_existing(
    data: List[Dict[str, Any]],
    *,
    sentiment_col: Optional[str],
    scores_by_row: Dict[int, Optional[int]],
) -> Dict[str, Any]:
    if not sentiment_col:
        return {
            "available": False,
            "compared_rows": 0,
            "agreement_rows": 0,
            "agreement_rate": None,
        }

    compared = 0
    agreed = 0
    for i, row in enumerate(data):
        existing = _normalize_sentiment_label(row.get(sentiment_col, ""))
        new_label = _score_to_coarse_label(scores_by_row.get(i))
        if existing is None or new_label is None:
            continue
        compared += 1
        if existing == new_label:
            agreed += 1

    return {
        "available": compared > 0,
        "compared_rows": compared,
        "agreement_rows": agreed,
        "agreement_rate": round(agreed / compared, 4) if compared > 0 else None,
    }


def _extract_contents_by_label(
    row_meta: List[Dict[str, Any]],
    label: str,
    limit: int = 10,
) -> List[str]:
    texts = [m["cleaned"] for m in row_meta if m.get("label") == label and m.get("cleaned")]
    texts.sort(key=len, reverse=True)
    return texts[:limit]


def _bucket_emotion_label_for_typical(meta: Dict[str, Any]) -> str:
    """与饼图「其他」口径对齐：六字之一或「其他」。"""
    emo = meta.get("emotion")
    score = meta.get("score")
    if isinstance(emo, str) and emo.strip() in FINE_EMOTIONS:
        return emo.strip()
    raw = str(emo or "").strip()
    if raw and raw not in FINE_EMOTIONS:
        return "其他"
    if score is None:
        return "其他"
    return _emotion_from_score(int(_clamp_score(score)))


def _extract_emotion_typical_expressions(
    row_meta: List[Dict[str, Any]],
    *,
    per_emotion: int = 4,
    max_snippet_chars: int = 88,
    min_cleaned_chars: int = 4,
) -> Dict[str, List[str]]:
    """
    按细粒度情绪从正文清洗结果中抽取典型表达（去重、偏长样本优先），供报告饼图下模块化展示。

    短文本、符号较多的样本在去重阶段可能被过滤；若某情绪桶内仍有原文但列表为空，则至少保留一条最长摘录。
    """
    buckets: Dict[str, List[str]] = {e: [] for e in list(FINE_EMOTIONS) + ["其他"]}
    for m in row_meta:
        if not isinstance(m, dict):
            continue
        cleaned = str(m.get("cleaned") or "").strip()
        if len(cleaned) < min_cleaned_chars:
            continue
        lab = _bucket_emotion_label_for_typical(m)
        if lab not in buckets:
            lab = "其他"
        snippet = re.sub(r"\s+", " ", cleaned)[:max_snippet_chars]
        buckets[lab].append(snippet)

    out: Dict[str, List[str]] = {}
    for lab in list(FINE_EMOTIONS) + ["其他"]:
        items = list(buckets.get(lab, []))
        items.sort(key=len, reverse=True)
        seen: set[str] = set()
        uniq: List[str] = []
        for t in items:
            sig = re.sub(r"\W+", "", t[:48])
            if len(sig) < 3 and len(t) < 12:
                continue
            if sig in seen:
                continue
            seen.add(sig)
            if len(t) > 96:
                t = t[:93] + "..."
            uniq.append(t)
            if len(uniq) >= per_emotion:
                break
        if not uniq and items:
            t0 = re.sub(r"\s+", " ", items[0])[:max_snippet_chars]
            if len(t0) > 96:
                t0 = t0[:93] + "..."
            uniq = [t0]
        out[lab] = uniq
    return out


def _fallback_summary_from_contents(contents: List[str], max_items: int = 3) -> List[str]:
    out: List[str] = []
    seen = set()
    for c in contents:
        s = re.sub(r"\s+", " ", str(c or "")).strip()
        if not s:
            continue
        key = s[:80]
        if key in seen:
            continue
        seen.add(key)
        out.append((s[:48] + "...") if len(s) > 48 else s)
        if len(out) >= max_items:
            break
    return out


def _generate_result_filename(retryContext: Optional[str] = None) -> str:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_name = f"sentiment_analysis_{timestamp}"
    if retryContext:
        task_id = get_task_id()
        if task_id:
            process_dir = get_task_process_dir(task_id)
            if process_dir.exists():
                existing_files = list(process_dir.glob("sentiment_analysis_*.json"))
                if existing_files:
                    suffix_nums: List[int] = []
                    for file in existing_files:
                        match = re.search(r"sentiment_analysis_\d{8}_\d{6}_(\d+)\.json", file.name)
                        if match:
                            suffix_nums.append(int(match.group(1)))
                    if not suffix_nums:
                        return f"{base_name}_1.json"
                    return f"{base_name}_{max(suffix_nums) + 1}.json"
                return f"{base_name}_1.json"
    return f"{base_name}.json"


@tool
def analysis_sentiment(
    eventIntroduction: str,
    dataFilePath: str,
    retryContext: Optional[str] = None,
    preferExistingSentimentColumn: Optional[bool] = None,
    contentColumns: Optional[List[str]] = None,
) -> str:
    """
    描述：分析情感倾向。自动识别与关键词工具一致的内容列，先做相同规则的文本清洗，再使用通义 qwen-plus
    对每条文本打 0～10 分（0-3 负面，4-6 中立，7-10 正面），并标注六类细粒度情绪之一（愤怒、焦虑、质疑、同情、嘲讽、支持），
    统计占比与情绪分布并总结主要观点。
    说明：默认使用通义模型对正文逐条打分（路径 B，含细粒度情绪），不使用 CSV「情感」列兜底。
    仅当环境变量 SONA_SENTIMENT_ALLOW_COLUMN_FALLBACK=1 时，才允许 preferExistingSentimentColumn 走路径 A 或在 LLM 失败时回退列统计。
    成功时除完整结果 JSON 外，另存两份饼图专用数据：粗粒度三维占比、六类细粒度情绪占比。
    """
    import json as json_module

    previous_result = None
    suggestions = None
    if retryContext:
        try:
            retry_data = json_module.loads(retryContext) if isinstance(retryContext, str) else retryContext
            previous_result = retry_data.get("previous_result")
            suggestions = retry_data.get("suggestions")
        except Exception:
            pass

    try:
        all_data = read_csv_rows_all(dataFilePath)
    except Exception as e:
        return json_module.dumps(
            {
                "error": f"读取数据文件失败: {str(e)}",
                "statistics": {},
                "positive_summary": [],
                "negative_summary": [],
                "result_file_path": "",
            },
            ensure_ascii=False,
        )

    if not all_data:
        return json_module.dumps(
            {
                "error": "数据文件为空",
                "statistics": {},
                "positive_summary": [],
                "negative_summary": [],
                "result_file_path": "",
            },
            ensure_ascii=False,
        )

    fieldnames = list(all_data[0].keys())
    # 如果外部强制指定内容列，优先使用（仅保留存在于文件表头的列）
    forced_cols: List[str] = []
    if contentColumns:
        normalized = [str(c or "").strip() for c in contentColumns if str(c or "").strip()]
        header_set = {str(h) for h in fieldnames}
        forced_cols = [c for c in normalized if c in header_set]
    content_columns = forced_cols if forced_cols else _identify_content_columns(fieldnames)
    if not content_columns:
        return json_module.dumps(
            {
                "error": (
                    "无法识别内容列，请确保列名包含: "
                    + ", ".join(CONTENT_COLUMN_KEYWORDS)
                ),
                "statistics": {},
                "positive_summary": [],
                "negative_summary": [],
                "result_file_path": "",
            },
            ensure_ascii=False,
        )

    n = len(all_data)
    sentiment_col = _identify_sentiment_column(all_data)
    prefer_existing_sentiment = _to_bool(preferExistingSentimentColumn, default=False)
    if not _allow_sentiment_column_fallback():
        prefer_existing_sentiment = False
    # 默认路径 B（LLM 打分 + 细粒度情绪）。路径 A 仅在允许兜底且 preferExistingSentimentColumn=True 时启用。
    use_existing_sentiment = prefer_existing_sentiment and _should_use_existing_sentiment(all_data, sentiment_col)

    summary_model: Any = None
    scoring_model_name = ""
    scoring_profile = ""
    scores_by_row: Dict[int, Optional[int]] = {}
    emotions_by_row: Dict[int, Optional[str]] = {}
    row_meta: List[Dict[str, Any]] = []
    row_scores_brief: List[Dict[str, Any]] = []
    to_score: List[Tuple[int, str]] = []
    metrics = _RequestMetrics()
    chunks: List[List[Tuple[int, str]]] = []
    parallel_workers = 1
    scoring_elapsed = 0.0
    score_model: Any = None

    def _build_scores_from_existing_sentiment() -> None:
        nonlocal scores_by_row, emotions_by_row
        if not sentiment_col:
            return
        for i, row in enumerate(all_data):
            cleaned = _row_cleaned_content(row, content_columns)
            raw_label = row.get(sentiment_col, "")
            norm_label = _normalize_sentiment_label(raw_label)
            if norm_label is None:
                if not cleaned:
                    scores_by_row[i] = None
                    emotions_by_row[i] = None
                else:
                    scores_by_row[i] = 5
                    emotions_by_row[i] = _emotion_from_score(5)
            else:
                scores_by_row[i] = _label_to_score(norm_label)
                emotions_by_row[i] = _emotion_from_coarse_label(norm_label)

    # 快速抽样重判（可选）：在已选择路径 A 时，可用 LLM 对抽样重判以估计分布，避免全量耗时。
    # - 默认关闭；开启方式：SONA_SENTIMENT_FAST_SAMPLE=1，且 preferExistingSentimentColumn=true 并满足路径 A 条件
    # - 抽样规模：max(min(n*rate, max_rows), min_rows)
    fast_sample_enabled = str(os.environ.get("SONA_SENTIMENT_FAST_SAMPLE", "0")).strip().lower() in {
        "1",
        "true",
        "yes",
        "y",
        "on",
    }
    sample_rate_raw = str(os.environ.get("SONA_SENTIMENT_FAST_SAMPLE_RATE", "0.1")).strip()
    sample_max_raw = str(os.environ.get("SONA_SENTIMENT_FAST_SAMPLE_MAX_ROWS", "220")).strip()
    sample_min_raw = str(os.environ.get("SONA_SENTIMENT_FAST_SAMPLE_MIN_ROWS", "60")).strip()
    try:
        sample_rate = float(sample_rate_raw)
    except Exception:
        sample_rate = 0.1
    try:
        sample_max_rows = int(sample_max_raw)
    except Exception:
        sample_max_rows = 220
    try:
        sample_min_rows = int(sample_min_raw)
    except Exception:
        sample_min_rows = 60
    sample_rate = max(0.01, min(sample_rate, 0.5))
    sample_max_rows = max(40, min(sample_max_rows, 1200))
    sample_min_rows = max(20, min(sample_min_rows, 400))

    use_llm_sample = bool(
        fast_sample_enabled
        and sentiment_col
        and use_existing_sentiment  # 已选择路径 A 时，可用抽样 LLM 估计分布（速度优先）
    )

    if use_existing_sentiment and sentiment_col and (not use_llm_sample):
        _build_scores_from_existing_sentiment()
    else:
        # 路径 B：LLM 全量/上限内打分；或「路径 A + 抽样」时先铺底情感列再对抽样调用 LLM
        if use_llm_sample and sentiment_col:
            _build_scores_from_existing_sentiment()
        # LLM 重判路径（无情感列、或默认路径 B、或路径 A 下抽样重判）
        try:
            # 关键：获取模型实例也可能卡死，加入硬超时。
            _pool = ThreadPoolExecutor(max_workers=1)
            _fut = _pool.submit(get_sentiment_model)
            try:
                score_model = _fut.result(timeout=max(1, _SENTIMENT_BATCH_TIMEOUT_SEC))
            finally:
                _pool.shutdown(wait=False, cancel_futures=True)
        except Exception as e:
            if sentiment_col and _allow_sentiment_column_fallback():
                use_existing_sentiment = True
                _build_scores_from_existing_sentiment()
            else:
                return json_module.dumps(
                    {
                        "error": f"获取情感分析模型失败: {str(e)}",
                        "statistics": {},
                        "positive_summary": [],
                        "negative_summary": [],
                        "result_file_path": "",
                    },
                    ensure_ascii=False,
                )

        for i, row in enumerate(all_data):
            cleaned = _row_cleaned_content(row, content_columns)
            if not cleaned:
                scores_by_row[i] = None
                emotions_by_row[i] = None
                continue
            to_score.append((i, cleaned))

        # 抽样重判：仅取部分样本做 LLM 打分，统计用样本估计（速度优先）
        if use_llm_sample and to_score:
            import random

            target = int(round(len(to_score) * sample_rate))
            target = max(sample_min_rows, min(target, sample_max_rows, len(to_score)))
            random.seed(17)
            to_score = random.sample(to_score, k=target) if len(to_score) > target else to_score

        # Cap total rows to score to keep walltime bounded (can be tuned via env).
        if len(to_score) > _SENTIMENT_MAX_ROWS:
            to_score = to_score[:_SENTIMENT_MAX_ROWS]

        # 组批
        scoring_started_at = time.perf_counter()
        chunks = [
            to_score[start : start + _SENTIMENT_DYNAMIC_BATCH_SIZE]
            for start in range(0, len(to_score), _SENTIMENT_DYNAMIC_BATCH_SIZE)
        ]

        # 并发批次打分（按批次级别并发，线程安全且可控），不足时退回串行
        parallel_workers = min(_SENTIMENT_BATCH_PARALLEL_WORKERS, len(chunks)) if chunks else 1
        if parallel_workers <= 1:
            for chunk in chunks:
                if (time.perf_counter() - scoring_started_at) > float(_SENTIMENT_MAX_WALLTIME_SEC):
                    break
                part = _score_batch(score_model, event_introduction=eventIntroduction, batch=chunk, metrics=metrics)
                for idx, _txt in chunk:
                    cell = part.get(idx)
                    if isinstance(cell, dict):
                        sc = _clamp_score(cell.get("score"))
                        scores_by_row[idx] = sc
                        emotions_by_row[idx] = _normalize_emotion(cell.get("emotion"), score=sc)
                    else:
                        sc = _clamp_score(cell) if isinstance(cell, (int, float)) else 5
                        scores_by_row[idx] = sc
                        emotions_by_row[idx] = _emotion_from_score(sc)
        else:
            with ThreadPoolExecutor(max_workers=parallel_workers) as executor:
                futures = {}
                for chunk in chunks:
                    fut = executor.submit(_score_batch_worker, event_introduction=eventIntroduction, batch=chunk, metrics=metrics)
                    futures[fut] = chunk
                for fut in as_completed(list(futures.keys())):
                    chunk = futures.get(fut) or []
                    try:
                        part = fut.result()
                    except Exception:
                        dr = {"score": 5, "emotion": _emotion_from_score(5)}
                        part = {idx: dict(dr) for idx, _txt in chunk}
                    for idx, _txt in chunk:
                        cell = part.get(idx)
                        if isinstance(cell, dict):
                            sc = _clamp_score(cell.get("score"))
                            scores_by_row[idx] = sc
                            emotions_by_row[idx] = _normalize_emotion(cell.get("emotion"), score=sc)
                        else:
                            sc = _clamp_score(cell) if isinstance(cell, (int, float)) else 5
                            scores_by_row[idx] = sc
                            emotions_by_row[idx] = _emotion_from_score(sc)
        scoring_elapsed = time.perf_counter() - scoring_started_at

        # 关键兜底：LLM 打分若整体不可用/疑似超时（无成功请求），且存在“情感”列，则强制回退情感列
        try:
            succ = int(metrics.requests_succeeded or 0)
            started = int(metrics.requests_started or 0)
        except Exception:
            succ, started = 0, 0
        if sentiment_col and (started > 0 and succ <= 0):
            if _allow_sentiment_column_fallback():
                use_existing_sentiment = True
                scores_by_row = {}
                emotions_by_row = {}
                _build_scores_from_existing_sentiment()
            else:
                return json_module.dumps(
                    {
                        "error": (
                            "LLM 情感打分全部失败（无成功批次），且已关闭 CSV 情感列兜底"
                            "（SONA_SENTIMENT_ALLOW_COLUMN_FALLBACK=0）。"
                        ),
                        "statistics": {
                            "total": n,
                            "sentiment_source": "llm_scoring_failed",
                            "concurrency_metrics": metrics.summary(
                                elapsed_sec=scoring_elapsed, rows_scored=len(to_score)
                            ),
                        },
                        "positive_summary": [],
                        "negative_summary": [],
                        "result_file_path": "",
                    },
                    ensure_ascii=False,
                )

    for i in range(n):
        s = scores_by_row.get(i)
        if s is not None and emotions_by_row.get(i) is None:
            emotions_by_row[i] = _emotion_from_score(int(s))

    for i, row in enumerate(all_data):
        cleaned = _row_cleaned_content(row, content_columns)
        s = scores_by_row.get(i)
        if s is None:
            label = "中立"
        else:
            label = _label_from_score(s)
        emo = emotions_by_row.get(i)
        if emo is None and s is not None:
            emo = _emotion_from_score(int(s))
        if isinstance(emo, str) and emo in FINE_EMOTIONS:
            adj = _nudge_emotion_for_text(cleaned, int(s) if s is not None else 5, emo)
            if adj != emo:
                emotions_by_row[i] = adj
                emo = adj
        row_meta.append({"cleaned": cleaned, "label": label, "score": s, "emotion": emo})
        preview = (cleaned[:200] + "...") if len(cleaned) > 200 else cleaned
        row_scores_brief.append(
            {"row_index": i, "score": s, "label": label, "emotion": emo, "text_preview": preview}
        )

    emotion_typical_expressions = _extract_emotion_typical_expressions(row_meta)

    # 注意：若为抽样重判，则 scores_by_row 只覆盖部分行；统计应基于已打分样本估计比例
    if use_llm_sample:
        sampled_scores = {i: s for i, s in scores_by_row.items() if s is not None}
        # 若极端情况下无有效样本，则退回中立
        if not sampled_scores:
            sampled_scores = {0: 5}
        statistics = _build_statistics(len(sampled_scores), sampled_scores)
        statistics["total"] = n
        statistics["sampling"] = {
            "enabled": True,
            "sampled_rows": len(sampled_scores),
            "sample_rate": round(len(sampled_scores) / max(1, n), 4),
            "note": "情感分布为抽样估计（为提升速度，未对全量逐条重判）",
        }
        em_rows = [(sampled_scores[i], emotions_by_row.get(i)) for i in sorted(sampled_scores.keys())]
    else:
        statistics = _build_statistics(n, scores_by_row)
        em_rows = [(scores_by_row.get(i), emotions_by_row.get(i)) for i in range(n)]
    em_stats = _emotion_counts_from_pairs(em_rows)
    statistics["emotion_counts"] = em_stats["emotion_counts"]
    statistics["emotion_ratios"] = em_stats["emotion_ratios"]
    statistics["emotion_typical_expressions"] = emotion_typical_expressions
    statistics["content_columns"] = content_columns
    statistics["batching"] = {
        "batch_size": _SENTIMENT_DYNAMIC_BATCH_SIZE,
        "batch_count": len(chunks),
        "parallel_workers": parallel_workers,
        "retry_per_batch": _SENTIMENT_BATCH_RETRIES,
        "batch_timeout_sec": _SENTIMENT_BATCH_TIMEOUT_SEC,
        "max_walltime_sec": _SENTIMENT_MAX_WALLTIME_SEC,
        "max_rows": _SENTIMENT_MAX_ROWS,
        "max_chars_per_text": _MAX_CHARS_PER_TEXT,
        "mode": (
            "llm_sampling"
            if use_llm_sample
            else ("existing_sentiment_column" if use_existing_sentiment else "llm_scoring")
        ),
        "sentiment_column": sentiment_col or "",
    }
    statistics["concurrency_metrics"] = (
        metrics.summary(elapsed_sec=scoring_elapsed, rows_scored=len(to_score))
        if (not use_existing_sentiment) or use_llm_sample
        else {
            "elapsed_sec": 0.0,
            "rows_scored": 0,
            "requests_started": 0,
            "requests_succeeded": 0,
            "requests_failed": 0,
            "qps": 0.0,
            "qpm": 0.0,
            "rpm": 0.0,
            "max_concurrent_connections": 0,
        }
    )
    if use_existing_sentiment:
        # 显式选择路径 A 与「LLM 失败回退到列」区分展示
        if preferExistingSentimentColumn is True:
            statistics["sentiment_source"] = "existing_column"
            statistics["fallback_used"] = False
        else:
            statistics["sentiment_source"] = "existing_column_fallback"
            statistics["fallback_used"] = True
    else:
        statistics["sentiment_source"] = "llm_scoring"
    llm_rows = len(to_score) if ((not use_existing_sentiment) or use_llm_sample) else 0
    statistics["llm_coverage"] = round(llm_rows / n, 4) if n else 0.0
    req_started = int(statistics.get("concurrency_metrics", {}).get("requests_started", 0) or 0)
    req_succeeded = int(statistics.get("concurrency_metrics", {}).get("requests_succeeded", 0) or 0)
    statistics["parse_success_rate"] = (round(req_succeeded / req_started, 4) if req_started > 0 else 1.0)
    statistics["agreement_with_existing"] = _compute_agreement_with_existing(
        all_data, sentiment_col=sentiment_col, scores_by_row=scores_by_row
    )

    positive_contents = _extract_contents_by_label(row_meta, "正面", limit=10)
    negative_contents = _extract_contents_by_label(row_meta, "负面", limit=10)

    # 与提示词对齐：只要该方向已抽出样本就传入模型并要求总结（不再用占比>0.1 卡死，避免低占比时摘要全空）
    need_positive = bool(positive_contents)
    need_negative = bool(negative_contents)

    if use_existing_sentiment:
        positive_summary = _fallback_summary_from_contents(positive_contents, max_items=3)
        negative_summary = _fallback_summary_from_contents(negative_contents, max_items=3)
        parsed = {"positive_summary": positive_summary, "negative_summary": negative_summary}
    else:
        try:
            prompt_template = get_analysis_sentiment_prompt()
        except Exception as e:
            # 总结提示词加载失败：仍然保证返回可用统计（用简单摘要兜底）
            positive_summary = _fallback_summary_from_contents(positive_contents, max_items=3)
            negative_summary = _fallback_summary_from_contents(negative_contents, max_items=3)
            parsed = {"positive_summary": positive_summary, "negative_summary": negative_summary}
        else:
            retry_section = "无（首次分析）" if not previous_result else str(previous_result)
            suggestions_section = "无" if not suggestions else str(suggestions)

            prompt = prompt_template.format(
                event_introduction=eventIntroduction,
                statistics=json_module.dumps(statistics, ensure_ascii=False, indent=2),
                positive_contents="\n\n".join(positive_contents) if need_positive and positive_contents else "无",
                negative_contents="\n\n".join(negative_contents) if need_negative and negative_contents else "无",
                need_positive="是" if need_positive else "否",
                need_negative="是" if need_negative else "否",
                previous_result=retry_section,
                suggestions=suggestions_section,
            )

            try:
                summary_messages = [
                    SystemMessage(content="你是一个专业的情感倾向分析专家。"),
                    HumanMessage(content=prompt),
                ]
                summary_resp = score_model.invoke(summary_messages)
                result_text = summary_resp.content if hasattr(summary_resp, "content") else str(summary_resp)
            except Exception:
                # 模型总结失败：仍然可用（用简单摘要兜底）
                positive_summary = _fallback_summary_from_contents(positive_contents, max_items=3)
                negative_summary = _fallback_summary_from_contents(negative_contents, max_items=3)
                parsed = {"positive_summary": positive_summary, "negative_summary": negative_summary}
            else:
                try:
                    json_match = re.search(r"\{[\s\S]*\}", result_text)
                    parsed = json_module.loads(json_match.group() if json_match else result_text)
                except Exception:
                    positive_summary = _fallback_summary_from_contents(positive_contents, max_items=3)
                    negative_summary = _fallback_summary_from_contents(negative_contents, max_items=3)
                    parsed = {
                        "positive_summary": positive_summary,
                        "negative_summary": negative_summary,
                        "raw_result": result_text,
                    }

                if not isinstance(parsed, dict):
                    positive_summary = _fallback_summary_from_contents(positive_contents, max_items=3)
                    negative_summary = _fallback_summary_from_contents(negative_contents, max_items=3)
                    parsed = {
                        "positive_summary": positive_summary,
                        "negative_summary": negative_summary,
                        "raw_result": result_text,
                    }
                else:
                    # 模型偶发仍返回空数组：有样本时用截断摘要兜底，避免落盘/交互侧「全空」
                    ps = parsed.get("positive_summary")
                    ns = parsed.get("negative_summary")
                    if (not isinstance(ps, list) or not ps) and positive_contents:
                        parsed["positive_summary"] = _fallback_summary_from_contents(positive_contents, max_items=3)
                    if (not isinstance(ns, list) or not ns) and negative_contents:
                        parsed["negative_summary"] = _fallback_summary_from_contents(negative_contents, max_items=3)

    full_result: Dict[str, Any] = {
        "statistics": statistics,
        "positive_summary": parsed.get("positive_summary", []),
        "negative_summary": parsed.get("negative_summary", []),
        "content_columns": content_columns,
        "row_scores": row_scores_brief,
        "scoring_model": "existing_sentiment_column" if use_existing_sentiment else "qwen-plus",
        "scoring_profile": "existing_column" if use_existing_sentiment else "sentiment",
        "raw_summary": None,
        "coarse_pie_chart_path": "",
        "fine_emotion_pie_chart_path": "",
        "emotion_typical_expressions": emotion_typical_expressions,
    }

    task_id = get_task_id()
    if task_id:
        try:
            process_dir = ensure_task_dirs(task_id)
            filename = _generate_result_filename(retryContext)
            result_file = process_dir / filename
            stem = Path(filename).stem
            pie_coarse_file = process_dir / f"{stem}_pie_coarse.json"
            pie_fine_file = process_dir / f"{stem}_pie_fine.json"
            pie_coarse_payload = _build_pie_coarse_payload(statistics)
            pie_fine_payload = _build_pie_fine_emotion_payload(statistics, emotion_typical_expressions)
            full_result["coarse_pie_chart_path"] = ""
            full_result["fine_emotion_pie_chart_path"] = ""
            try:
                with open(pie_coarse_file, "w", encoding="utf-8", errors="replace") as f:
                    json_module.dump(pie_coarse_payload, f, ensure_ascii=False, indent=2)
                full_result["coarse_pie_chart_path"] = str(pie_coarse_file)
            except Exception as e2:
                full_result["save_error"] = (full_result.get("save_error") or "") + f" 粗粒度饼图保存失败: {e2}"
            try:
                with open(pie_fine_file, "w", encoding="utf-8", errors="replace") as f:
                    json_module.dump(pie_fine_payload, f, ensure_ascii=False, indent=2)
                full_result["fine_emotion_pie_chart_path"] = str(pie_fine_file)
            except Exception as e3:
                full_result["save_error"] = (full_result.get("save_error") or "") + f" 细粒度饼图保存失败: {e3}"
            full_result["result_file_path"] = str(result_file)
            with open(result_file, "w", encoding="utf-8", errors="replace") as f:
                json_module.dump(full_result, f, ensure_ascii=False, indent=2)
        except Exception as e:
            full_result["save_error"] = (full_result.get("save_error") or "") + f"保存主结果失败: {str(e)}"
            full_result["result_file_path"] = ""
            full_result["coarse_pie_chart_path"] = ""
            full_result["fine_emotion_pie_chart_path"] = ""
    else:
        full_result["save_error"] = "未找到任务ID，无法保存结果文件"
        full_result["result_file_path"] = ""
        full_result["coarse_pie_chart_path"] = ""
        full_result["fine_emotion_pie_chart_path"] = ""

    # 终端/交互展示：保持简洁（详细打分与样本请看保存文件）
    result_file_path = str(full_result.get("result_file_path") or "")
    summary_message = (
        f"情感打分完成：共 {n} 行，命中 {len(content_columns)} 个内容列；"
        f"QPS={statistics.get('concurrency_metrics', {}).get('qps', 0)}，"
        f"RPM={statistics.get('concurrency_metrics', {}).get('rpm', 0)}，"
        f"最大并发连接={statistics.get('concurrency_metrics', {}).get('max_concurrent_connections', 0)}。"
    )
    summary_payload: Dict[str, Any] = {
        "message": summary_message,
        "statistics": statistics,
        "content_columns": content_columns,
        "positive_summary": full_result.get("positive_summary", []),
        "negative_summary": full_result.get("negative_summary", []),
        "scoring_model": full_result.get("scoring_model", ""),
        "scoring_profile": full_result.get("scoring_profile", ""),
        "result_file_path": result_file_path,
        "coarse_pie_chart_path": str(full_result.get("coarse_pie_chart_path") or ""),
        "fine_emotion_pie_chart_path": str(full_result.get("fine_emotion_pie_chart_path") or ""),
    }
    if full_result.get("save_error"):
        summary_payload["save_error"] = full_result.get("save_error")

    return json_module.dumps(summary_payload, ensure_ascii=False)
