"""基于 Neo4j 图谱检索并调用大模型（Qwen-Plus）回答问题。"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import re
import sys
import textwrap
import unicodedata
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool

from utils.env_loader import get_env_config


def _get_text_generation_model_compat():  # noqa: D401
    """与 ``wiki_rag`` 等工具链对齐：使用 ``tools`` profile，避免过期的 ``get_text_generation_model`` 导入失败。"""
    from model.factory import get_tools_model

    return get_tools_model()

try:
    from rich.console import Console
    from rich.markdown import Markdown
    from rich.panel import Panel
    from rich.rule import Rule
    from rich.table import Table
except Exception:  # pragma: no cover
    Console = None  # type: ignore[assignment]
    Markdown = None  # type: ignore[assignment]
    Panel = None  # type: ignore[assignment]
    Rule = None  # type: ignore[assignment]
    Table = None  # type: ignore[assignment]


@dataclass
class Neo4jConfig:
    """Neo4j 连接配置。"""

    uri: str
    username: str
    password: str
    database: str


def _load_graph_database_class() -> Any:
    """延迟导入 neo4j 驱动，避免帮助命令触发第三方噪音。"""
    try:
        with contextlib.redirect_stderr(io.StringIO()):
            from neo4j import GraphDatabase as _GraphDatabase
    except Exception as exc:
        raise RuntimeError(
            "无法导入 neo4j 驱动，请先安装依赖（pip install neo4j），"
            "并检查本地环境兼容性。"
        ) from exc
    return _GraphDatabase


def _load_neo4j_config(database_override: str = "") -> Neo4jConfig:
    """从环境变量加载 Neo4j 配置（``NEO4J_URI`` / ``NEO4J_USERNAME`` / ``NEO4J_PASSWORD`` / ``NEO4J_DATABASE``）。

    ``/wiki`` 熊猫图谱预取与 ``neo4j_qa`` 工具共用此套变量，与 Graph RAG 的 ``SONA_NEO4J_*`` 分离。
    """
    get_env_config()
    uri = (os.getenv("NEO4J_URI") or "").strip()
    username = (os.getenv("NEO4J_USERNAME") or "").strip()
    password = (os.getenv("NEO4J_PASSWORD") or "").strip()
    database = (database_override or os.getenv("NEO4J_DATABASE") or "neo4j").strip()
    missing: List[str] = []
    if not uri:
        missing.append("NEO4J_URI")
    if not username:
        missing.append("NEO4J_USERNAME")
    if not password:
        missing.append("NEO4J_PASSWORD")
    if missing:
        raise ValueError(f"缺少 Neo4j 环境变量: {', '.join(missing)}")
    return Neo4jConfig(uri=uri, username=username, password=password, database=database)


def _extract_keywords(question: str) -> List[str]:
    """从问题中提取关键词。"""
    raw = re.findall(r"[\u4e00-\u9fffA-Za-z0-9_]+", question)
    stop_words = {"什么", "哪些", "如何", "为什么", "怎么", "一下", "请问", "一下子", "关于", "容易", "是否"}
    zh_dict = [
        "大熊猫",
        "熊猫",
        "疾病",
        "治疗",
        "天敌",
        "捕食",
        "敌害",
        "栖息地",
        "栖息",
        "分布",
        "居住",
        "环境",
        "历史",
        "发现",
        "定名",
        "行为",
        "习性",
        "外形",
        "繁殖",
        "食物",
        "气味",
        "气味标记",
        "交流",
        "沟通",
        "声音",
        "叫声",
        "生长",
        "发育",
        "生长发育",
        "周期",
        "月龄",
        "体重",
        "幼仔",
        "亚成年",
        "成年",
        "性成熟",
    ]
    keywords: List[str] = []
    seen: set[str] = set()
    for phrase in zh_dict:
        if phrase in question and phrase not in seen:
            seen.add(phrase)
            keywords.append(phrase)
    for token in raw:
        key = token.strip()
        if len(key) <= 1 or key in stop_words:
            continue
        if key not in seen:
            seen.add(key)
            keywords.append(key)
    return keywords[:10]


def _build_intent_terms(question: str) -> List[str]:
    """根据问题主题扩展意图词，提升检索召回精度。"""
    q = question.lower()
    terms: List[str] = []
    if any(token in q for token in ["疾病", "生病", "病", "健康", "症状", "治疗"]):
        terms.extend(["疾病", "病", "症状", "治疗", "感染", "炎"])
    if any(token in q for token in ["天敌", "为敌", "敌害", "捕食", "威胁"]):
        terms.extend(["天敌", "捕食", "敌害", "威胁", "黄喉貂", "金猫", "豹"])
    if any(token in q for token in ["栖息", "居住", "分布", "哪里", "环境"]):
        terms.extend(["栖息地", "分布", "环境", "生存", "地区"])
    if any(token in q for token in ["历史", "发现", "定名"]):
        terms.extend(["发现", "历史", "定名", "鉴定", "文献"])
    if any(token in q for token in ["行为", "习性", "活动"]):
        terms.extend(["行为", "习性", "活动", "采食", "交配"])
    if any(token in q for token in ["交流", "沟通", "气味", "气味标记", "声音", "叫声"]):
        terms.extend(["交流", "沟通", "气味标记", "声音交流", "叫声", "标记"])
    if any(token in q for token in ["生长", "发育", "生长发育", "周期", "月龄", "幼仔", "亚成年", "成年", "性成熟"]):
        terms.extend(["生长", "发育", "生长发育", "周期", "月龄", "幼仔", "亚成年", "成年", "性成熟", "体重", "恒牙"])
    if any(token in q for token in ["寿命", "最长寿", "年龄", "几岁", "活多久", "存活"]):
        terms.extend(["寿命", "最长寿", "年龄", "岁", "野外", "圈养", "存活"])
    if any(token in q for token in ["节约能量", "能量", "减少活动", "活动范围", "消耗", "代谢"]):
        terms.extend(["节约能量", "能量", "减少社会活动", "减小活动范围", "消耗", "代谢"])
    deduped: List[str] = []
    seen: set[str] = set()
    for term in terms:
        if term and term not in seen:
            seen.add(term)
            deduped.append(term)
    return deduped[:20]


def _detect_intent(question: str) -> str:
    """识别问题意图，用于检索和回答约束。"""
    q = question.lower()
    if any(token in q for token in ["疾病", "生病", "病", "健康", "症状", "治疗"]):
        return "disease"
    if any(token in q for token in ["天敌", "为敌", "敌害", "捕食", "威胁"]):
        return "predator"
    if any(token in q for token in ["朋友", "伴生", "近邻"]):
        return "companion"
    if any(token in q for token in ["栖息", "居住", "分布", "哪里", "环境"]):
        return "habitat"
    if any(token in q for token in ["历史", "发现", "定名"]):
        return "history"
    if any(token in q for token in ["行为", "习性", "活动"]):
        return "behavior"
    if any(token in q for token in ["食物", "吃什么", "进食", "主食"]):
        return "food"
    if any(token in q for token in ["消化", "胃", "肠", "盲肠"]):
        return "digestion"
    if any(token in q for token in ["交流", "沟通", "气味", "气味标记", "声音", "叫声"]):
        return "communication"
    if any(token in q for token in ["生长", "发育", "生长发育", "周期", "月龄", "幼仔", "亚成年", "成年", "性成熟"]):
        return "development"
    if any(token in q for token in ["寿命", "最长寿", "年龄", "几岁", "活多久", "存活"]):
        return "lifespan"
    if any(token in q for token in ["节约能量", "能量", "减少活动", "活动范围", "消耗", "代谢"]):
        return "energy"
    return "general"


def _build_intent_predicates(intent: str) -> List[str]:
    """不同意图下优先命中的关系词。"""
    mapping: Dict[str, List[str]] = {
        "disease": ["疾病", "病", "感染", "治疗", "症状"],
        "predator": ["天敌", "捕食", "敌害", "威胁"],
        "companion": ["伴生", "近邻", "朋友", "共栖"],
        "habitat": ["栖息", "分布", "生存环境", "地区", "海拔", "气候"],
        "history": ["发现", "定名", "鉴定", "记载", "历史", "年代"],
        "behavior": ["行为", "习性", "活动", "交配", "育幼", "刻板行为"],
        "food": ["食用", "主食", "进食", "食物", "采食"],
        "digestion": ["消化", "胃", "肠", "盲肠", "消化道", "营养吸收"],
        "communication": ["交流", "沟通", "气味标记", "标记", "声音", "叫声", "发情"],
        "development": ["生长", "发育", "成长", "阶段", "月龄", "体重", "性成熟", "恒牙", "独立生活", "成年"],
        "lifespan": ["寿命", "最长寿", "年龄", "活", "存活", "出生于"],
        "energy": ["节约能量", "减少社会活动", "减小活动范围", "缩短怀孕期", "消耗", "代谢"],
    }
    return mapping.get(intent, [])


def _build_topic_filters(question: str, intent: str) -> List[str]:
    """根据问题和意图推断候选 topic，用于先按 topic 缩小检索范围。"""
    q = question.lower()
    topics: List[str] = []
    if intent in {"predator", "companion"}:
        topics.append("生态关系")
    if intent == "disease":
        topics.append("疾病健康")
    if intent == "habitat":
        topics.append("生存环境")
    if intent == "history":
        topics.append("历史背景")
    if intent in {"behavior", "communication"}:
        topics.append("行为习性")
    if intent in {"digestion", "food"}:
        topics.extend(["生理构造", "行为习性"])
    if intent == "development":
        topics.extend(["生理构造", "行为习性", "个体档案"])
    if intent == "lifespan":
        topics.extend(["个体档案", "行为习性", "生理构造"])
    if intent == "energy":
        topics.extend(["生理构造", "行为习性"])

    if any(token in q for token in ["生长", "发育", "月龄", "体重", "幼仔", "成年", "性成熟"]):
        topics.extend(["生理构造", "行为习性", "个体档案"])
    if any(token in q for token in ["交流", "气味标记", "叫声", "沟通"]):
        topics.append("行为习性")
    if any(token in q for token in ["天敌", "伴生", "捕食"]):
        topics.append("生态关系")

    deduped: List[str] = []
    seen: set[str] = set()
    for topic in topics:
        if topic and topic not in seen:
            seen.add(topic)
            deduped.append(topic)
    return deduped


def _rank_and_dedup_rows(rows: List[Dict[str, str]]) -> List[Dict[str, str]]:
    """按(subject,predicate,object,evidence)去重，并按分数降序排序。"""
    by_key: Dict[Tuple[str, str, str, str], Dict[str, str]] = {}
    for row in rows:
        key = (
            str(row.get("subject", "")),
            str(row.get("predicate", "")),
            str(row.get("object", "")),
            str(row.get("evidence", "")),
        )
        score = float(row.get("total_score", 0) or 0)
        existed = by_key.get(key)
        if existed is None or float(existed.get("total_score", 0) or 0) < score:
            by_key[key] = row
    deduped = list(by_key.values())
    deduped.sort(key=lambda x: float(x.get("total_score", 0) or 0), reverse=True)
    return deduped


def _prepare_query_keywords(keywords: List[str]) -> List[str]:
    """去除过于泛化的词，避免检索被“大熊猫”等高频词污染。"""
    deduped: List[str] = []
    seen: set[str] = set()
    for token in keywords:
        item = str(token).strip()
        if not item or item in seen:
            continue
        seen.add(item)
        deduped.append(item)
    if len(deduped) <= 1:
        return deduped
    generic = {"大熊猫", "熊猫"}
    narrowed = [token for token in deduped if token not in generic]
    return narrowed or deduped


def _merge_query_terms(*term_groups: List[str]) -> List[str]:
    """合并查询词并去重，过滤过长噪音词。"""
    merged: List[str] = []
    seen: set[str] = set()
    for group in term_groups:
        for token in group:
            item = str(token).strip()
            if not item or item in seen:
                continue
            # 过长中文串通常是整句误分词，检索价值低。
            if len(item) > 12 and re.fullmatch(r"[\u4e00-\u9fff]+", item):
                continue
            seen.add(item)
            merged.append(item)
    return merged[:30]


def _select_candidate_sources(
    session: Any,
    *,
    keywords: List[str],
    intent_terms: List[str],
    topic_filters: List[str],
    top_n: int = 8,
) -> List[str]:
    """
    第一阶段：先按问题关键词/意图词召回候选 source_file。
    这样可以先缩小检索范围，再做关系级精排，减少主题漂移。
    """
    if not keywords and not intent_terms:
        return []

    source_query = """
UNWIND $terms AS t
MATCH ()-[r]->()
WITH t, r,
  (
    CASE WHEN size($topic_filters) > 0 AND coalesce(r.topic, '') IN $topic_filters THEN 8 ELSE 0 END +
    CASE WHEN coalesce(r.source_file, '') CONTAINS t THEN 6 ELSE 0 END +
    CASE WHEN coalesce(r.predicate, '') CONTAINS t THEN 3 ELSE 0 END +
    CASE WHEN coalesce(endNode(r).id, '') CONTAINS t THEN 3 ELSE 0 END +
    CASE WHEN coalesce(r.evidence, '') CONTAINS t THEN 1 ELSE 0 END
  ) AS s
WHERE s > 0
WITH coalesce(r.source_file, '') AS source_file, sum(s) AS score
WHERE source_file <> ''
RETURN source_file, score
ORDER BY score DESC
LIMIT $top_n
"""
    terms = keywords + [term for term in intent_terms if term not in keywords]
    rows = list(session.run(source_query, terms=terms[:20], topic_filters=topic_filters, top_n=max(1, top_n)))
    return [str(r["source_file"]).strip() for r in rows if str(r["source_file"]).strip()]


def _fetch_knowledge(question: str, top_k: int, database: str) -> List[Dict[str, str]]:
    """从 Neo4j 检索相关三元组。"""
    config = _load_neo4j_config(database_override=database)
    graph_database = _load_graph_database_class()
    driver = graph_database.driver(config.uri, auth=(config.username, config.password))
    keywords = _extract_keywords(question)
    intent = _detect_intent(question)
    intent_terms = _build_intent_terms(question)
    intent_predicates = _build_intent_predicates(intent)
    topic_filters = _build_topic_filters(question, intent)
    if not keywords:
        keywords.append("大熊猫")
    keywords = _prepare_query_keywords(keywords)
    query_terms = _merge_query_terms(keywords, intent_terms, intent_predicates)
    if not query_terms:
        query_terms = keywords or ["大熊猫"]
    require_intent = len(intent_terms) > 0
    min_total_score = 4 if len(query_terms) >= 2 else 1

    query = """
UNWIND $query_terms AS kw
MATCH (s)-[r]->(o)
WITH s, r, o, kw,
  (
    CASE WHEN coalesce(r.predicate, '') CONTAINS kw THEN 6 ELSE 0 END +
    CASE WHEN coalesce(o.id, '') CONTAINS kw THEN 4 ELSE 0 END +
    CASE WHEN coalesce(s.id, '') CONTAINS kw THEN 3 ELSE 0 END +
    CASE WHEN coalesce(r.evidence, '') CONTAINS kw THEN 2 ELSE 0 END
  ) AS kw_score
WHERE kw_score > 0
WITH s, r, o, sum(kw_score) AS score
WHERE (size($candidate_sources) = 0) OR coalesce(r.source_file, '') IN $candidate_sources
WITH s, r, o, score,
  reduce(intent_score = 0, t IN $intent_terms |
    intent_score +
    CASE
      WHEN coalesce(r.predicate, '') CONTAINS t OR coalesce(o.id, '') CONTAINS t OR coalesce(r.evidence, '') CONTAINS t
      THEN 2 ELSE 0
    END
  ) AS intent_score
WITH s, r, o, score, intent_score,
  CASE
    WHEN size($topic_filters) = 0 THEN 0
    WHEN coalesce(r.topic, '') IN $topic_filters THEN 6
    ELSE 0
  END AS topic_bias,
  reduce(pred_score = 0, p IN $intent_predicates |
    pred_score + CASE WHEN coalesce(r.predicate, '') CONTAINS p THEN 3 ELSE 0 END
  ) AS predicate_bias
WHERE ((NOT $require_intent) OR intent_score > 0)
  AND ((NOT $require_topic) OR coalesce(r.topic, '') IN $topic_filters)
  AND (score + intent_score + topic_bias + predicate_bias) >= $min_total_score
RETURN
  coalesce(s.id, '') AS subject,
  coalesce(r.predicate, type(r)) AS predicate,
  coalesce(o.id, '') AS object,
  coalesce(r.evidence, '') AS evidence,
  coalesce(r.source_file, '') AS source_file,
  coalesce(r.topic, '') AS topic,
  (score + intent_score + topic_bias + predicate_bias) AS total_score
ORDER BY total_score DESC, size(coalesce(r.evidence, '')) DESC
LIMIT $top_k
"""
    try:
        with driver.session(database=config.database) as session:
            candidate_sources = _select_candidate_sources(
                session,
                keywords=keywords,
                intent_terms=intent_terms,
                topic_filters=topic_filters,
                top_n=8,
            )
            rows: List[Dict[str, str]] = []
            for require_topic in [bool(topic_filters), False]:
                rows = [
                    dict(record)
                    for record in session.run(
                        query,
                        query_terms=query_terms,
                        candidate_sources=candidate_sources,
                        intent_terms=intent_terms,
                        intent_predicates=intent_predicates,
                        topic_filters=topic_filters,
                        require_intent=require_intent,
                        require_topic=require_topic,
                        min_total_score=min_total_score,
                        top_k=max(1, top_k),
                    )
                ]
                if rows:
                    break
            if rows:
                return _rank_and_dedup_rows(rows)
            return []
    finally:
        driver.close()


def is_panda_graph_wiki_query(question: str) -> bool:
    """
    判断用户问题是否与「熊猫 / 大熊猫」知识图谱主题相关。

    供 ``/wiki`` 等上层在命中时优先从 Neo4j 拉取三元组证据，再与本地 Markdown 知识库合并。
    """
    q = str(question or "").strip()
    if not q:
        return False
    markers_zh = (
        "熊猫",
        "大熊猫",
        "国宝",
        "竹熊",
        "猫熊",
        "滚滚",
        "川陕",
        "卧龙",
        "碧峰峡",
        "繁育基地",
        "熊猫基地",
        "成都大熊猫",
        "秦岭大熊猫",
    )
    if any(m in q for m in markers_zh):
        return True
    ql = q.lower()
    if "giant panda" in ql or "ailuropoda" in ql:
        return True
    return False


def fetch_graph_triples_for_wiki(
    question: str,
    *,
    top_k: int = 16,
    database: str = "",
) -> List[Dict[str, str]]:
    """
    仅从 Neo4j 检索相关三元组，不调用图谱问答里的合成 LLM。

    未配置 ``NEO4J_URI`` / 账号或驱动不可用时返回空列表，不向外抛异常。

    与 Graph RAG 的 ``SONA_NEO4J_*`` 无关：仅使用 ``NEO4J_*``（如 Aura 实例 ID 作 ``NEO4J_USERNAME``）。
    """
    if not str(question or "").strip():
        return []
    lim = max(1, min(int(top_k or 16), 40))
    try:
        return _fetch_knowledge(question, top_k=lim, database=database)
    except Exception:
        return []


def _llm_content_to_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: List[str] = []
        for item in content:
            if isinstance(item, dict):
                text = item.get("text")
                if isinstance(text, str):
                    parts.append(text)
            else:
                parts.append(str(item))
        return "\n".join(parts).strip()
    return str(content)


def _build_context_rows(rows: List[Dict[str, str]]) -> str:
    lines: List[str] = []
    for idx, item in enumerate(rows, start=1):
        lines.append(
            f"{idx}. {item['subject']} -[{item['predicate']}]-> {item['object']} | 证据: {item['evidence']} | 来源: {item['source_file']}"
        )
    return "\n".join(lines)


def _pick_supporting_sources(answer: str, evidences: List[Dict[str, Any]], limit: int = 10) -> List[str]:
    """只返回在回答文本中被实际引用到的来源文档。"""
    answer_text = (answer or "").strip()
    if not answer_text:
        return []

    matched: List[str] = []
    seen: set[str] = set()
    for item in evidences:
        if not isinstance(item, dict):
            continue
        source = str(item.get("source_file", "")).strip()
        evidence = str(item.get("evidence", "")).strip()
        predicate = str(item.get("predicate", "")).strip()
        if not source or not evidence:
            continue

        # 优先用证据片段匹配；证据较长时使用前后片段降低偶然误命中。
        short_head = evidence[:24]
        short_tail = evidence[-24:] if len(evidence) > 30 else evidence
        evidence_hit = (short_head and short_head in answer_text) or (short_tail and short_tail in answer_text)
        predicate_hit = predicate and predicate in answer_text
        if evidence_hit or predicate_hit:
            if source not in seen:
                seen.add(source)
                matched.append(source)
            if len(matched) >= max(1, limit):
                break
    if matched:
        return matched[: max(1, limit)]

    # 兜底：当回答被模型重写导致无法命中文本片段时，按证据得分回填来源，避免空结果。
    scored_sources: List[Tuple[float, str]] = []
    seen_fallback: set[str] = set()
    for item in evidences:
        if not isinstance(item, dict):
            continue
        source = str(item.get("source_file", "")).strip()
        if not source or source in seen_fallback:
            continue
        score = float(item.get("total_score", 0) or 0)
        seen_fallback.add(source)
        scored_sources.append((score, source))
    scored_sources.sort(key=lambda x: x[0], reverse=True)
    return [source for _, source in scored_sources[: max(1, limit)]]


def _persona_instructions(persona: str) -> str:
    """回答语气/角色设定。"""
    persona_map: Dict[str, str] = {
        "default": "语气：清晰、客观、偏科普；适合成年人快速理解。",
        "kid": (
            "语气：面向 6-10 岁小朋友的科普讲解员；用词简单、句子短；"
            "多用类比；避免恐吓性描述；必要时用“我们可以理解为…”帮助理解。"
        ),
        "educator": "语气：耐心、鼓励式；像课堂老师；适当分点；避免堆砌术语。",
        "expert": "语气：专业、克制；术语可保留但需简短解释；结构更紧凑。",
        "story": "语气：轻故事化但仍基于证据；不要编造情节；比喻要克制。",
    }
    return persona_map.get(persona.strip(), persona_map["default"])


def _get_console(*, use_rich: bool) -> Any:
    if use_rich and Console is not None:
        return Console(highlight=False, soft_wrap=True)
    return None


def _is_small_screen(console: Any) -> bool:
    """根据终端宽度判断是否应使用紧凑布局。"""
    if console is None:
        return False
    try:
        return int(console.size.width) < 120
    except Exception:
        return False


def _strip_markdown_for_terminal(text: str) -> str:
    """将模型输出的 Markdown 简化为终端友好的纯文本。"""
    normalized = (text or "").replace("\r\n", "\n").replace("\r", "\n")
    normalized = re.sub(r"^\s{0,3}#{1,6}\s*", "", normalized, flags=re.MULTILINE)
    normalized = normalized.replace("**", "").replace("__", "")
    normalized = normalized.replace("`", "")
    # Windows 默认终端常见编码为 GBK，遇到 emoji/特殊符号会直接抛 UnicodeEncodeError。
    # 这里做最小集合的兼容清理，避免影响正文中文内容。
    for sym in ["✅", "⚠️", "⚠", "❌", "ℹ️", "ℹ", "🔍", "📌", "🚫"]:
        normalized = normalized.replace(sym, "")
    # Windows 默认编码可能无法打印 '•'，改用 ASCII '-' 保证兼容。
    normalized = re.sub(r"^\s*[-*]\s+", "- ", normalized, flags=re.MULTILINE)
    normalized = re.sub(r"\n{3,}", "\n\n", normalized)
    return normalized.strip()


def _wrap_for_terminal(text: str, console: Any) -> str:
    """按终端宽度硬换行，避免视觉截断。"""
    raw = _strip_markdown_for_terminal(text)
    if not raw:
        return ""
    width = 88
    if console is not None:
        try:
            width = max(40, int(console.size.width) - 4)
        except Exception:
            width = 88
    # 先按自然段切分，再对每个段内的行做折行，尽量保留原有的段落结构。
    paragraphs = re.split(r"\n\s*\n", raw)
    wrapped_paragraphs: List[str] = []
    for paragraph in paragraphs:
        lines = [ln.strip() for ln in paragraph.split("\n") if ln.strip()]
        if not lines:
            continue
        wrapped_lines = [_hard_wrap_by_display_width(line, width=width) for line in lines]
        wrapped_paragraphs.append("\n".join(wrapped_lines))
    return "\n\n".join(wrapped_paragraphs).strip()


def _char_display_width(ch: str) -> int:
    """估算字符显示宽度：中日韩宽字符按 2，其余按 1。"""
    if not ch:
        return 0
    if unicodedata.east_asian_width(ch) in {"W", "F"}:
        return 2
    return 1


def _hard_wrap_by_display_width(text: str, width: int) -> str:
    """按显示宽度硬换行，避免终端自动换行导致的截断。"""
    line = text or ""
    if width <= 0:
        return line
    out_lines: List[str] = []
    current: List[str] = []
    current_width = 0
    for ch in line:
        ch_w = _char_display_width(ch)
        if current and current_width + ch_w > width:
            out_lines.append("".join(current))
            current = [ch]
            current_width = ch_w
        else:
            current.append(ch)
            current_width += ch_w
    if current:
        out_lines.append("".join(current))
    return "\n".join(out_lines)


def _print_text_module(console: Any, title: str, content: str, style: str = "cyan") -> None:
    """统一的模块化文本输出。"""
    content_text = (content or "").strip() or "(空)"
    if console is None:
        print(f"\n=== {title} ===")
        print(_wrap_for_terminal(content_text, console=None))
        return
    console.print(Rule(title, style=style))
    wrapped = _wrap_for_terminal(content_text, console=console)
    for line in wrapped.split("\n"):
        console.print(line, soft_wrap=False, overflow="ignore")


def _summarize_list_lines(items: Any, *, limit: int, label: str) -> List[str]:
    """将列表摘要为多行，避免单行过长。"""
    if not isinstance(items, list):
        return [f"{label}: (none)"]
    cleaned = [str(x).strip() for x in items if str(x).strip()]
    if not cleaned:
        return [f"{label}: (none)"]
    shown = cleaned[: max(1, limit)]
    lines: List[str] = [f"{label}[{idx+1}]: {value}" for idx, value in enumerate(shown)]
    remaining = len(cleaned) - len(shown)
    if remaining > 0:
        lines.append(f"{label}: ... +{remaining} more")
    return lines


def _build_session_dir(session_name: str = "") -> Path:
    """创建 JSON 会话输出目录。"""
    base_dir = Path("sandbox") / "qa_sessions"
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_name = re.sub(r"[^0-9A-Za-z_\-]+", "_", (session_name or "").strip()).strip("_")
    session_id = safe_name or f"session_{stamp}"
    session_dir = base_dir / session_id
    session_dir.mkdir(parents=True, exist_ok=True)
    return session_dir


def _save_query_result(
    *,
    session_dir: Path,
    turn_index: int,
    question: str,
    parsed: Dict[str, Any],
) -> Path:
    """保存单轮问答结果为 JSON。"""
    payload = {
        "saved_at": datetime.now().isoformat(timespec="seconds"),
        "turn_index": turn_index,
        "question": question,
        "result": parsed,
    }
    file_path = session_dir / f"turn_{turn_index:03d}.json"
    file_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return file_path


def _print_cli_result(
    *,
    parsed: Dict[str, Any],
    show_sources: bool,
    use_rich: bool,
) -> None:
    console = _get_console(use_rich=use_rich)
    status = "OK" if parsed.get("ok") else "ERROR"
    if console is None:
        header = (
            f"status: {status}\n"
            f"database: {parsed.get('database', '')}\n"
            f"persona: {parsed.get('persona', '')}\n"
            f"rows_count: {parsed.get('rows_count', 0)}"
        )
        query_plan = parsed.get("query_plan", {})
        hit_graph = parsed.get("hit_graph", {})
        quality = parsed.get("quality", {})
        if isinstance(quality, dict) and quality:
            header += f"\nquality: {quality.get('score_0_100', 0)} ({quality.get('level', 'n/a')})"
        _print_text_module(console, "运行信息", header, style="cyan")
        if isinstance(query_plan, dict) and query_plan:
            query_text = (
                f"{str(query_plan.get('cypher', '')).strip()}\n\n"
                f"params: {json.dumps(query_plan.get('params', {}), ensure_ascii=False)}"
            )
            _print_text_module(console, "Cypher 解析", query_text, style="blue")
        if isinstance(hit_graph, dict) and hit_graph:
            graph_lines: List[str] = [f"triple_hits: {hit_graph.get('triple_hits', 0)}"]
            graph_lines.extend(_summarize_list_lines(hit_graph.get("relationships", []), limit=10, label="relationships"))
            graph_lines.extend(_summarize_list_lines(hit_graph.get("subject_nodes", []), limit=10, label="subject_nodes"))
            graph_lines.extend(_summarize_list_lines(hit_graph.get("object_nodes", []), limit=10, label="object_nodes"))
            _print_text_module(console, "命中节点与关系", "\n".join(graph_lines), style="blue")
        _print_text_module(
            console,
            "回答",
            str(parsed.get("answer", "") or ""),
            style="green",
        )
        if show_sources:
            evidences = parsed.get("evidences", [])
            if isinstance(evidences, list):
                source_files = _pick_supporting_sources(
                    answer=str(parsed.get("answer", "")),
                    evidences=evidences,
                    limit=10,
                )
                source_text = "supporting_source_files:"
                if source_files:
                    for source in source_files:
                        source_text += f"\n- {source}"
                else:
                    source_text += "\n- (no supporting source found in answer text)"
                _print_text_module(console, "依据来源（文档）", source_text, style="yellow")
        if parsed.get("error"):
            _print_text_module(console, "错误", f"{parsed['error']}", style="red")
        return

    console.print(Rule("neo4j_qa", style="cyan"))
    header = (
        f"status: {status}\n"
        f"database: {parsed.get('database', '')}\n"
        f"persona: {parsed.get('persona', '')}\n"
        f"rows_count: {parsed.get('rows_count', 0)}"
    )
    quality = parsed.get("quality", {})
    if isinstance(quality, dict) and quality:
        header += f"\nquality: {quality.get('score_0_100', 0)} ({quality.get('level', 'n/a')})"
    _print_text_module(console, "运行信息", header, style="cyan")

    query_plan = parsed.get("query_plan", {})
    if isinstance(query_plan, dict) and query_plan:
        cypher_text = str(query_plan.get("cypher", "")).strip()
        params_text = json.dumps(query_plan.get("params", {}), ensure_ascii=False)
        query_text = f"{cypher_text}\n\nparams: {params_text}"
        _print_text_module(console, "Cypher 解析", query_text, style="blue")

    hit_graph = parsed.get("hit_graph", {})
    if isinstance(hit_graph, dict) and hit_graph:
        graph_lines: List[str] = [f"triple_hits: {hit_graph.get('triple_hits', 0)}"]
        graph_lines.extend(_summarize_list_lines(hit_graph.get("relationships", []), limit=10, label="relationships"))
        graph_lines.extend(_summarize_list_lines(hit_graph.get("subject_nodes", []), limit=10, label="subject_nodes"))
        graph_lines.extend(_summarize_list_lines(hit_graph.get("object_nodes", []), limit=10, label="object_nodes"))
        _print_text_module(console, "命中节点与关系", "\n".join(graph_lines), style="blue")

    _print_text_module(console, "回答", str(parsed.get("answer", "") or ""), style="green")

    if show_sources:
        evidences = parsed.get("evidences", [])
        source_files: List[str] = []
        if isinstance(evidences, list):
            source_files = _pick_supporting_sources(
                answer=str(parsed.get("answer", "")),
                evidences=evidences,
                limit=10,
            )
        if source_files:
            source_text = ""
            for path in source_files:
                source_text += f"- {path}\n"
            _print_text_module(console, "依据来源（文档）", source_text.strip(), style="yellow")
        else:
            _print_text_module(console, "依据来源（文档）", "(本轮未匹配到可展示的来源文件)", style="yellow")

    if parsed.get("error"):
        _print_text_module(console, "错误", str(parsed["error"]), style="red")


def _generate_answer(question: str, rows: List[Dict[str, str]], strict_mode: bool, persona: str) -> str:
    llm = _get_text_generation_model_compat()
    context_text = _build_context_rows(rows)
    intent = _detect_intent(question)
    intent_hint = {
        "disease": "优先总结疾病类别、典型病名与症状线索。",
        "predator": "优先回答天敌名单与其威胁对象（幼仔/病弱个体等）。",
        "companion": "优先回答伴生动物与同域共栖关系。",
        "habitat": "优先回答分布区、栖息地类型、地理环境。",
        "history": "优先回答时间线（起源、发现、定名）。",
        "behavior": "优先回答行为特点并按类别归纳。",
        "food": "优先回答食物类型并区分野外/圈养。",
        "digestion": "优先回答消化系统结构、消化相关行为与证据边界。",
        "communication": "优先回答交流方式（气味标记/声音）及其作用场景。",
        "development": "优先回答生长发育阶段（如月龄、体重、独立与性成熟等）并按时间线组织。",
        "lifespan": "优先回答野外与圈养寿命范围、最长寿个体及相关数字信息。",
        "energy": "优先回答节约能量策略（减少活动、缩小活动范围等）及其原因。",
    }.get(intent, "优先回答与问题最相关的事实。")
    system_prompt = (
        "你是熊猫知识库问答助手。你只能基于给定知识作答，不得编造。"
        "回答应尽量全面，优先覆盖定义、要点、场景差异和补充说明。"
        "当问题可分维度时，使用分点结构，不要只给一句话结论。"
        "依据部分保持最小化，只标注文档来源，不展开长证据句。"
        "避免机械重复。"
        f"{_persona_instructions(persona)}"
    )
    if strict_mode:
        system_prompt += "当证据不足时，必须明确说“知识库暂无足够依据”。"
    user_prompt = (
        f"问题：{question}\n\n"
        f"知识库检索结果（最多 {len(rows)} 条）：\n{context_text}\n\n"
        f"回答偏好：{intent_hint}\n\n"
        "请输出：\n"
        "1) 全面回答（建议 3-6 个要点，必要时分“野外/圈养”“原因/影响”等小节）\n"
        "2) 依据来源（精简列出 2-8 个来源文档，格式：- 来源：<source_file>）\n"
        "要求：依据只标注来源，不要复述大段证据原文。\n"
        "如果证据不足，直接回答“知识库暂无足够依据”。"
    )
    response = llm.invoke([SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)])
    return _llm_content_to_text(response.content).strip()


def _generate_fallback_answer(question: str, persona: str) -> str:
    """
    当知识库证据不足时，调用通用大模型能力进行兜底回答。
    """
    llm = _get_text_generation_model_compat()
    system_prompt = (
        "你是熊猫科普助手。当前知识库未命中证据。"
        "请基于通用知识谨慎回答，明确这是“非知识库证据回答”，避免编造具体数字与细节。"
        "若不确定，请明确说明不确定性。"
        f"{_persona_instructions(persona)}"
    )
    user_prompt = (
        f"问题：{question}\n\n"
        "请输出：\n"
        "1) 先给出简洁回答（可分点）\n"
        "2) 再单独给出“说明：以下内容来自通用知识，不是当前知识库证据”\n"
    )
    response = llm.invoke([SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)])
    return _llm_content_to_text(response.content).strip()


def _build_cypher_preview(question: str, intent: str, top_k: int) -> Dict[str, Any]:
    """由大模型生成 Cypher 查询预览（用于展示，不直接执行）。"""
    llm = _get_text_generation_model_compat()
    prompt = (
        "你是 Neo4j 查询规划助手。根据用户问题输出简洁 Cypher 方案。"
        "只输出 JSON，字段为 cypher, params。"
        "cypher 仅使用 MATCH (s)-[r]->(o) 这种结构，并尽量使用 predicate/evidence/source_file 条件。\n"
        f"question: {question}\nintent: {intent}\nlimit: {max(1, top_k)}\n"
        '仅输出：{"cypher":"MATCH ... RETURN ... LIMIT $top_k","params":{"top_k":20,"keywords":["..."]}}'
    )
    response = llm.invoke(
        [
            SystemMessage(content="你只输出合法 JSON。"),
            HumanMessage(content=prompt),
        ]
    )
    text = _llm_content_to_text(response.content).strip()
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end <= start:
        raise ValueError(f"Cypher 预览生成失败：{text}")
    parsed = json.loads(text[start : end + 1])
    cypher = str(parsed.get("cypher", "")).strip()
    params = parsed.get("params", {})
    if not isinstance(params, dict):
        params = {}
    if not cypher:
        raise ValueError("Cypher 预览为空")
    return {"cypher": cypher, "params": params}


def _build_hit_graph(rows: List[Dict[str, str]]) -> Dict[str, Any]:
    """从命中三元组提取节点与关系摘要。"""
    subjects: List[str] = []
    objects: List[str] = []
    relationships: List[str] = []
    seen_s: set[str] = set()
    seen_o: set[str] = set()
    seen_r: set[str] = set()
    for item in rows:
        s = str(item.get("subject", "")).strip()
        o = str(item.get("object", "")).strip()
        r = str(item.get("predicate", "")).strip()
        if s and s not in seen_s:
            seen_s.add(s)
            subjects.append(s)
        if o and o not in seen_o:
            seen_o.add(o)
            objects.append(o)
        if r and r not in seen_r:
            seen_r.add(r)
            relationships.append(r)
    return {
        "triple_hits": len(rows),
        "subject_nodes": subjects[:20],
        "object_nodes": objects[:20],
        "relationships": relationships[:20],
    }


def _score_to_level(score_0_100: float) -> str:
    if score_0_100 >= 85:
        return "excellent"
    if score_0_100 >= 70:
        return "good"
    if score_0_100 >= 50:
        return "fair"
    return "poor"


def _build_quality_prompt(question: str, answer: str, evidences: List[Dict[str, Any]]) -> str:
    evidence_lines: List[str] = []
    for idx, item in enumerate(evidences[:12], start=1):
        evidence_lines.append(
            f"{idx}. {item.get('subject', '')} -[{item.get('predicate', '')}]-> {item.get('object', '')} | 证据: {item.get('evidence', '')}"
        )
    evidence_text = "\n".join(evidence_lines) if evidence_lines else "(无检索证据)"
    return (
        "请你作为熊猫知识库问答质量评估器，按如下维度进行 0-5 打分并只输出 JSON：\n"
        "- correctness: 回答事实正确性\n"
        "- completeness: 对问题覆盖完整性\n"
        "- groundedness: 回答与给定证据一致性\n"
        "- concise_clarity: 表达清晰度与简洁度\n"
        "- reason: 一句话说明\n\n"
        f"question: {question}\n"
        f"answer: {answer}\n"
        f"evidences:\n{evidence_text}\n\n"
        '仅输出：{"correctness": number, "completeness": number, "groundedness": number, "concise_clarity": number, "reason": "..."}'
    )


def _evaluate_quality_by_llm(question: str, answer: str, evidences: List[Dict[str, Any]]) -> Dict[str, Any]:
    """使用大模型对本轮回答进行质量评估。"""
    # 规则护栏：若直接拒答/暂无依据，则质量分必须很低（避免出现“拒答=满分”的误判）。
    no_evidence_phrase = "知识库暂无足够依据。"
    normalized_answer = (answer or "").strip()
    # 强护栏：若本轮完全没有检索证据（evidences 为空），无论 fallback 还是其它模式，
    # 质量评估都必须反映“groundedness 为 0 / 基于证据的可信度不足”，
    # 否则容易出现“无证据却被打到 good（如 70 分）”的误判。
    if not evidences:
        has_no_evidence_phrase = no_evidence_phrase in normalized_answer
        # 如果回答里已经明确声明“知识库暂无足够依据”，则更符合拒答/无证据场景；
        # 否则说明模型在无证据前提下仍给出细节结论，置信度更低。
        score = 10.0 if has_no_evidence_phrase else 25.0
        return {
            "mode": "no_evidence_guardrail",
            "score_0_100": score,
            "level": _score_to_level(score),
            "correctness_0_5": 1.0 if score >= 25.0 else 0.0,
            "completeness_0_5": 1.0 if score >= 25.0 else 0.0,
            "groundedness_0_5": 0.0,
            "concise_clarity_0_5": 2.0 if score >= 25.0 else 1.0,
            "reason": "本轮没有检索证据（evidences 为空），因此 groundedness 必须为 0；即便为 fallback 模式提供一般性回答，也不应评为 good 级别。",
        }
    if not normalized_answer or normalized_answer == no_evidence_phrase:
        score = 10.0 if evidences else 0.0
        return {
            "mode": "refusal_guardrail",
            "score_0_100": float(score),
            "level": "poor",
            "correctness_0_5": 0.0,
            "completeness_0_5": 0.0,
            "groundedness_0_5": 0.0,
            "concise_clarity_0_5": 0.0,
            "reason": "本轮未给出有效回答（暂无依据/拒答），评分按最低档处理。",
        }

    llm = _get_text_generation_model_compat()
    prompt = _build_quality_prompt(question=question, answer=answer, evidences=evidences)
    response = llm.invoke(
        [
            SystemMessage(content="你是严格评估器，只输出合法 JSON，不要输出其它文本。"),
            HumanMessage(content=prompt),
        ]
    )
    text = _llm_content_to_text(response.content).strip()
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end <= start:
        raise ValueError(f"质量评估模型输出非 JSON: {text}")
    obj = json.loads(text[start : end + 1])
    correctness = max(0.0, min(5.0, float(obj.get("correctness", 0.0))))
    completeness = max(0.0, min(5.0, float(obj.get("completeness", 0.0))))
    groundedness = max(0.0, min(5.0, float(obj.get("groundedness", 0.0))))
    concise_clarity = max(0.0, min(5.0, float(obj.get("concise_clarity", 0.0))))
    reason = str(obj.get("reason", "")).strip()
    score_0_100 = ((correctness + completeness + groundedness + concise_clarity) / 20.0) * 100.0
    return {
        "mode": "llm_judge",
        "score_0_100": round(score_0_100, 2),
        "level": _score_to_level(score_0_100),
        "correctness_0_5": round(correctness, 4),
        "completeness_0_5": round(completeness, 4),
        "groundedness_0_5": round(groundedness, 4),
        "concise_clarity_0_5": round(concise_clarity, 4),
        "reason": reason,
    }


@tool
def neo4j_qa(
    question: str,
    top_k: int = 20,
    database: str = "",
    strict_mode: bool = True,
    persona: str = "default",
) -> str:
    """
    描述：基于 Neo4j 图谱检索知识，并调用 Qwen-Plus 进行带证据回答。
    输入：
    - question：用户问题。
    - top_k：最多检索多少条关系，默认 20。
    - database：Neo4j 数据库名，默认读取 NEO4J_DATABASE 或 neo4j。
    - strict_mode：严格模式；证据不足时明确返回“暂无依据”。
    - persona：回答语气/角色（default/kid/educator/expert/story）。
    输出：JSON 字符串，包含 answer 与 evidences。
    """
    result: Dict[str, Any] = {
        "ok": False,
        "question": question,
        "top_k": max(1, top_k),
        "database": "",
        "persona": persona.strip() or "default",
        "answer_mode": "knowledge_base",
        "rows_count": 0,
        "answer": "",
        "evidences": [],
        "query_plan": {},
        "hit_graph": {},
        "quality": {},
    }
    try:
        config = _load_neo4j_config(database_override=database)
        result["database"] = config.database
        intent = _detect_intent(question=question)
        try:
            result["query_plan"] = _build_cypher_preview(
                question=question,
                intent=intent,
                top_k=max(1, top_k),
            )
        except Exception as exc:
            result["query_plan"] = {
                "cypher": "MATCH (s)-[r]->(o) WHERE ... RETURN s,r,o LIMIT $top_k",
                "params": {"top_k": max(1, top_k)},
                "error": f"{type(exc).__name__}: {exc}",
            }
        rows = _fetch_knowledge(question=question, top_k=max(1, top_k), database=config.database)
        result["rows_count"] = len(rows)
        result["evidences"] = rows
        result["hit_graph"] = _build_hit_graph(rows)
        if not rows:
            result["answer_mode"] = "llm_fallback"
            result["answer"] = _generate_fallback_answer(
                question=question,
                persona=persona,
            )
            result["quality"] = _evaluate_quality_by_llm(
                question=question,
                answer=str(result.get("answer", "")),
                evidences=[],
            )
            result["ok"] = True
            return json.dumps(result, ensure_ascii=False)
        result["answer"] = _generate_answer(
            question=question,
            rows=rows,
            strict_mode=strict_mode,
            persona=persona,
        )
        result["quality"] = _evaluate_quality_by_llm(
            question=question,
            answer=str(result.get("answer", "")),
            evidences=rows,
        )
        result["ok"] = True
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
    return json.dumps(result, ensure_ascii=False)


def _build_cli_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="基于 Neo4j 图谱 + Qwen-Plus 进行问答。")
    parser.add_argument(
        "--question",
        default="",
        help="要提问的问题文本。",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=20,
        help="检索关系条数上限，默认 20。",
    )
    parser.add_argument(
        "--database",
        default="",
        help="Neo4j 数据库名，默认读取 NEO4J_DATABASE 或 neo4j。",
    )
    parser.add_argument(
        "--strict",
        dest="strict_mode",
        action="store_true",
        help="严格模式（证据不足时明确返回暂无依据）。默认开启。",
    )
    parser.add_argument(
        "--no-strict",
        dest="strict_mode",
        action="store_false",
        help="关闭严格模式。",
    )
    parser.set_defaults(strict_mode=True)
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="进入终端交互问答模式（输入 exit/quit 退出）。",
    )
    parser.add_argument(
        "--show-sources",
        action="store_true",
        help="显示本轮命中的证据来源文件（最多10条，去重）。",
    )
    parser.add_argument(
        "--persona",
        default="default",
        choices=["default", "kid", "educator", "expert", "story"],
        help="回答语气/角色：default/kid/educator/expert/story。",
    )
    parser.add_argument(
        "--plain",
        action="store_true",
        help="禁用 rich 美化输出（纯文本）。",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="输出完整 JSON；默认输出摘要。",
    )
    parser.add_argument(
        "--save-json",
        dest="save_json",
        action="store_true",
        help="每轮问答自动保存 JSON 到 sandbox/qa_sessions（默认开启）。",
    )
    parser.add_argument(
        "--no-save-json",
        dest="save_json",
        action="store_false",
        help="关闭自动保存 JSON。",
    )
    parser.set_defaults(save_json=True)
    parser.add_argument(
        "--session-name",
        default="",
        help="会话目录名（仅保留字母数字下划线横线），默认自动生成。",
    )
    return parser


def _run_one_cli_query(
    question: str,
    top_k: int,
    database: str,
    strict_mode: bool,
    verbose: bool,
    show_sources: bool,
    persona: str,
    use_rich: bool,
    save_json: bool,
    session_dir: Path | None,
    turn_index: int,
) -> bool:
    result_json = neo4j_qa.invoke(
        {
            "question": question,
            "top_k": top_k,
            "database": database,
            "strict_mode": strict_mode,
            "persona": persona,
        }
    )
    try:
        parsed = json.loads(result_json)
    except Exception:
        print(result_json)
        return False

    if verbose:
        print(json.dumps(parsed, ensure_ascii=False, indent=2))
    else:
        _print_cli_result(parsed=parsed, show_sources=show_sources, use_rich=use_rich)
    if save_json and session_dir is not None:
        save_path = _save_query_result(
            session_dir=session_dir,
            turn_index=turn_index,
            question=question,
            parsed=parsed,
        )
        print(f"[saved] {save_path}")
    return bool(parsed.get("ok"))


def main() -> int:
    parser = _build_cli_parser()
    args = parser.parse_args()
    use_rich = (not args.plain) and (Console is not None)
    persona = str(args.persona or "default")
    show_sources = bool(args.show_sources)
    session_dir = _build_session_dir(session_name=str(args.session_name or "")) if args.save_json else None
    if args.interactive:
        if use_rich and Console is not None and Panel is not None and Markdown is not None:
            console = Console(highlight=False, soft_wrap=True)
            console.print(
                Panel(
                    Markdown(
                        "进入 **Neo4j 问答交互模式**。\n\n"
                        "- 输入问题即可开始\n"
                        "- 输入 `exit` / `quit` 退出\n"
                        "- 输入 `:persona kid` 切换为儿童科普语气（可选：`default/educator/expert/story`）\n"
                        "- 输入 `:sources on` / `:sources off` 切换是否显示来源文档\n"
                        "- 输入 `:help` 查看帮助"
                    ),
                    title="Panda · 熊猫知识库问答",
                    border_style="cyan",
                )
            )
        else:
            print("进入 Neo4j 问答交互模式（输入 exit / quit 退出）")
            print("命令：:persona <default|kid|educator|expert|story>  |  :sources on|off  |  :help")
        turn_index = 1
        while True:
            try:
                question = input("\nquestion> ").strip()
            except (KeyboardInterrupt, EOFError):
                print("\n已退出交互模式。")
                return 0
            if not question:
                continue
            if question.lower() in {"exit", "quit"}:
                print("已退出交互模式。")
                return 0
            if question.startswith(":"):
                parts = question.split()
                cmd = parts[0].lower()
                if cmd in {":help", ":h"}:
                    print("命令：")
                    print("  :persona <default|kid|educator|expert|story>")
                    print("  :sources on|off")
                    print("  :help")
                    continue
                if cmd in {":persona", ":p"}:
                    if len(parts) < 2:
                        print(f"当前 persona: {persona}")
                        continue
                    next_persona = parts[1].strip().lower()
                    allowed = {"default", "kid", "educator", "expert", "story"}
                    if next_persona not in allowed:
                        print(f"不支持的 persona: {next_persona}，可选：{', '.join(sorted(allowed))}")
                        continue
                    persona = next_persona
                    print(f"已切换 persona: {persona}")
                    continue
                if cmd in {":sources", ":src"}:
                    if len(parts) < 2:
                        print(f"当前 sources: {'on' if show_sources else 'off'}")
                        continue
                    mode = parts[1].strip().lower()
                    if mode in {"on", "true", "1", "yes"}:
                        show_sources = True
                    elif mode in {"off", "false", "0", "no"}:
                        show_sources = False
                    else:
                        print("用法：:sources on|off")
                        continue
                    print(f"已切换 sources: {'on' if show_sources else 'off'}")
                    continue
                print("未知命令，输入 :help 查看帮助")
                continue
            _run_one_cli_query(
                question=question,
                top_k=args.top_k,
                database=args.database,
                strict_mode=args.strict_mode,
                verbose=args.verbose,
                show_sources=show_sources,
                persona=persona,
                use_rich=use_rich,
                save_json=bool(args.save_json),
                session_dir=session_dir,
                turn_index=turn_index,
            )
            turn_index += 1

    if not args.question.strip():
        parser.error("非交互模式下必须提供 --question，或使用 --interactive")
    ok = _run_one_cli_query(
        question=args.question.strip(),
        top_k=args.top_k,
        database=args.database,
        strict_mode=args.strict_mode,
        verbose=args.verbose,
        show_sources=args.show_sources,
        persona=persona,
        use_rich=use_rich,
        save_json=bool(args.save_json),
        session_dir=session_dir,
        turn_index=1,
    )
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
