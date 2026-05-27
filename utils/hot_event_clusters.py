"""热点抓取条目：标题相似度聚类 + 领域关键词分类（与舆情十一类正交）。"""

from __future__ import annotations

import json
import os
import re
import unicodedata
from difflib import SequenceMatcher
from typing import Any, Dict, FrozenSet, List, Optional, Tuple

# 与 ClassifyNode 舆情类型正交：主题领域（含体育、司法等，减少落入「其他」）
DOMAIN_ORDER: List[str] = [
    "司法与监督",
    "体育竞技",
    "国际关系与军事",
    "政治与公共安全",
    "财政金融",
    "产业与公司",
    "消费市场",
    "科技互联网",
    "文体娱乐",
    "教育科研",
    "医疗健康",
    "生态环境",
    "社会民生",
    "其他",
]

# 领域 -> 标题关键词（子串匹配；列表顺序 = 优先级，先命中先返回）
_DOMAIN_KEYWORDS: List[Tuple[str, Tuple[str, ...]]] = [
    (
        "司法与监督",
        (
            "死缓",
            "被判",
            "公诉",
            "刑拘",
            "立案",
            "落马",
            "反腐",
            "纪委",
            "被查",
            "通报",
            "军报",
            "有贪必肃",
            "职务犯罪",
            "行贿",
            "受贿",
        ),
    ),
    (
        "体育竞技",
        (
            "国乒",
            "孙颖莎",
            "王楚钦",
            "林昀儒",
            "莫雷加德",
            "逆转",
            "女团",
            "男团",
            "乒",
            "LPL",
            "NIP",
            "零封",
            "国足",
            "女足",
            "世界杯",
            "欧洲杯",
            "欧冠",
            "英超",
            "西甲",
            "NBA",
            "CBA",
            "奥运",
            "残奥",
            "足协",
            "转会",
            "假赛",
            "国乒男团",
            "国乒女团",
        ),
    ),
    (
        "医疗健康",
        (
            "汉坦病毒",
            "病毒",
            "医保",
            "医院",
            "卫健委",
            "疫情",
            "疫苗",
            "药品",
            "集采",
            "癌症",
            "流感",
            "中医",
            "门诊",
            "刷牙",
            "儿童用药",
            "公共卫生",
        ),
    ),
    (
        "教育科研",
        (
            "高考",
            "考研",
            "教育部",
            "高校",
            "论文",
            "科研",
            "院士",
            "实验室",
            "学术",
            "义务教育",
            "双减",
            "留学",
            "大学",
        ),
    ),
    (
        "生态环境",
        (
            "环保",
            "督察",
            "碳中和",
            "污染",
            "气候",
            "新能源",
            "光伏",
            "风电",
            "减排",
            "绿水青山",
            "生态",
            "黑水",
            "农田",
        ),
    ),
    (
        "科技互联网",
        (
            "人工智能",
            "AI",
            "大模型",
            "ChatGPT",
            "算力",
            "云计算",
            "开源",
            "苹果",
            "谷歌",
            "微软",
            "安卓",
            "iOS",
            "5G",
            "6G",
            "互联网",
            "App",
            "算法",
            "数据安全",
            "黑客",
            "歼-35",
            "歼35",
            "隐身战机",
            "阶跃星辰",
            "融资",
            "半导体",
            "芯片",
        ),
    ),
    (
        "财政金融",
        (
            "央行",
            "降息",
            "加息",
            "A股",
            "港股",
            "美股",
            "基金",
            "债券",
            "汇率",
            "人民币",
            "美元",
            "GDP",
            "财政",
            "税收",
            "银行",
            "保险",
            "证券",
            "期货",
            "比特币",
            "ETF",
            "溢价",
            "净值",
            "停牌",
            "外汇市场",
            "干预外汇",
            "雪球",
        ),
    ),
    (
        "国际关系与军事",
        (
            "美军",
            "伊朗",
            "以色列",
            "乌克兰",
            "北约",
            "联合国",
            "外交",
            "制裁",
            "军演",
            "导弹",
            "海峡",
            "霍尔木兹",
            "空袭",
            "冲突",
            "战争",
            "维和",
            "大使",
            "领事",
            "关税",
            "贸易战",
            "免签",
            "卢拉",
            "特朗普",
            "高市",
        ),
    ),
    (
        "政治与公共安全",
        (
            "中央",
            "国务院",
            "人大",
            "政协",
            "党纪",
            "巡视",
            "公安",
            "法院",
            "检察",
            "立法",
            "选举",
            "部长",
            "省长",
            "市委书记",
            "刑侦",
            "治安",
            "反恐",
            "校园霸凌",
            "鞭刑",
        ),
    ),
    (
        "文体娱乐",
        (
            "电影",
            "票房",
            "综艺",
            "浪姐",
            "演唱会",
            "明星",
            "剧集",
            "动漫",
            "游戏",
            "电竞",
            "音乐节",
            "张杰",
            "雨霖铃",
            "开机",
            "庆功宴",
            "Vlog",
            "B站",
            "华强买瓜",
        ),
    ),
    (
        "产业与公司",
        (
            "上市",
            "财报",
            "裁员",
            "并购",
            "重组",
            "董事长",
            "CEO",
            "华为",
            "阿里",
            "腾讯",
            "字节",
            "比亚迪",
            "宁德",
            "造车",
            "供应链",
            "追觅",
            "易点天下",
            "闻泰",
            "五粮液",
            "中信建投",
        ),
    ),
    (
        "消费市场",
        (
            "电商",
            "退款",
            "仅退款",
            "直播带货",
            "618",
            "双11",
            "胖东来",
            "山姆",
            "零售",
            "消费",
            "涨价",
            "降价",
            "促销",
            "外卖",
            "打车",
            "网约车",
            "会员",
            "淘宝免单",
            "假日经济",
            "出游",
            "体验经济",
        ),
    ),
    (
        "社会民生",
        (
            "就业",
            "社保",
            "养老金",
            "住房",
            "公积金",
            "户籍",
            "交通",
            "地铁",
            "铁路",
            "春运",
            "天气",
            "暴雨",
            "地震",
            "民生",
            "维权",
            "纠纷",
            "小县城",
            "微信",
            "语音消息",
            "刷手机",
            "刻板行为",
            "汉坦",
            "冥王星",
            "逆行",
            "夏天开始",
        ),
    ),
]


def normalize_title_for_match(title: str) -> str:
    """用于相似度比较的标题归一化。"""
    if not title:
        return ""
    t = unicodedata.normalize("NFKC", str(title)).strip().lower()
    t = re.sub(r"\s+", "", t)
    t = re.sub(r"[《》「」【】\[\]\"'（）()、，。！？!?,.\-_#:：]", "", t)
    return t


def _title_similarity(a: str, b: str) -> float:
    na, nb = normalize_title_for_match(a), normalize_title_for_match(b)
    if not na or not nb:
        return 0.0
    if na == nb:
        return 1.0
    return float(SequenceMatcher(None, na, nb).ratio())


def _hot_value(item: Dict[str, Any]) -> float:
    try:
        return float(item.get("hot_value") or 0)
    except (TypeError, ValueError):
        return 0.0


def classify_domain_for_title(title: str) -> str:
    """根据代表标题关键词映射领域；无命中为「其他」。"""
    if not title:
        return "其他"
    t = str(title)
    for domain, kws in _DOMAIN_KEYWORDS:
        for kw in kws:
            if kw in t:
                return domain
    return "其他"


def parse_llm_cluster_domain_json(raw: str, valid: FrozenSet[str]) -> Dict[str, str]:
    """解析 LLM 返回的 JSON 对象 cluster_id -> 领域名称。"""
    text = raw.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s*```\s*$", "", text).strip()
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return {}
    if not isinstance(data, dict):
        return {}
    out: Dict[str, str] = {}
    for k, v in data.items():
        if not isinstance(k, str) or not isinstance(v, str):
            continue
        kk = k.strip()
        vv = v.strip()
        if vv in valid:
            out[kk] = vv
    return out


def rebuild_stats_with_domains(
    stats: Dict[str, Any],
    cluster_id_to_domain: Dict[str, str],
    *,
    method: str = "llm_batch",
) -> Dict[str, Any]:
    """用 LLM（或外部）给出的 cluster_id -> 领域 覆盖统计，并重建 clusters_by_domain。"""
    domain_order = list(stats.get("domain_order") or DOMAIN_ORDER)
    valid = frozenset(domain_order)
    flat_in = list(stats.get("flat_clusters") or [])
    new_flat: List[Dict[str, Any]] = []
    for row in flat_in:
        cid = str(row.get("cluster_id") or "")
        d = cluster_id_to_domain.get(cid)
        if d is None or d not in valid:
            d = row.get("domain") or "其他"
        if d not in valid:
            d = "其他"
        new_flat.append({**row, "domain": d})

    clusters_by_domain: Dict[str, List[Dict[str, Any]]] = {d: [] for d in domain_order}
    for row in new_flat:
        dom = str(row.get("domain") or "其他")
        if dom not in clusters_by_domain:
            dom = "其他"
        clusters_by_domain[dom].append(row)

    for d in domain_order:
        clusters_by_domain[d].sort(key=lambda x: (-float(x.get("max_hot") or 0), -int(x.get("size") or 0)))

    meta = dict(stats.get("meta") or {})
    meta["domain_method"] = method
    return {
        **stats,
        "domain_order": domain_order,
        "flat_clusters": new_flat,
        "clusters_by_domain": clusters_by_domain,
        "meta": meta,
    }


def _effective_max_clusters(explicit: Optional[int]) -> int:
    if explicit is not None:
        m = explicit
    else:
        try:
            m = int(os.environ.get("HOT_CLUSTER_MAX", "50"))
        except ValueError:
            m = 50
    return max(25, min(m, 100))


def cluster_news_by_domain(
    news_list: List[Dict[str, Any]],
    *,
    max_clusters: Optional[int] = None,
    join_similarity: float = 0.38,
    cap_merge_min_similarity: float = 0.18,
) -> Dict[str, Any]:
    """
    对 news_list 做按热度降序的贪心聚类，再按簇代表标题打领域标签。

    ``max_clusters`` 默认读环境变量 ``HOT_CLUSTER_MAX``（默认 50），可减轻簇打满后的「兜底大簇」。

    Returns:
        domain_cluster_stats: domain_order, clusters_by_domain, flat_clusters（便于 JSON）
    """
    mc = _effective_max_clusters(max_clusters)
    items = [dict(x) for x in news_list if (x.get("title") or "").strip()]
    items.sort(key=_hot_value, reverse=True)

    clusters: List[Dict[str, Any]] = []

    for item in items:
        title = str(item.get("title") or "")
        best_i = -1
        best_sim = 0.0
        for i, c in enumerate(clusters):
            sim = _title_similarity(title, str(c["representative"].get("title") or ""))
            if sim > best_sim:
                best_sim = sim
                best_i = i

        if best_i >= 0 and best_sim >= join_similarity:
            clusters[best_i]["members"].append(item)
            continue

        if len(clusters) < mc:
            clusters.append({"representative": item, "members": [item]})
            continue

        if best_i >= 0 and best_sim >= cap_merge_min_similarity:
            clusters[best_i]["members"].append(item)
        elif clusters:
            clusters[-1]["members"].append(item)
        else:
            clusters.append({"representative": item, "members": [item]})

    flat: List[Dict[str, Any]] = []
    clusters_by_domain: Dict[str, List[Dict[str, Any]]] = {d: [] for d in DOMAIN_ORDER}

    for idx, c in enumerate(clusters):
        rep = c["representative"]
        members: List[Dict[str, Any]] = c["members"]
        rep_title = str(rep.get("title") or "")
        domain = classify_domain_for_title(rep_title)
        if domain not in clusters_by_domain:
            domain = "其他"

        source_ids: List[str] = []
        seen_sid: set[str] = set()
        for m in members:
            sid = str(m.get("source_id") or m.get("source") or "").strip() or "unknown"
            if sid not in seen_sid:
                seen_sid.add(sid)
                source_ids.append(sid)

        titles = [str(m.get("title") or "") for m in members if m.get("title")]
        sample = titles[:5]
        max_hot = max((_hot_value(m) for m in members), default=0.0)

        row = {
            "cluster_id": f"c{idx}",
            "representative_title": rep_title,
            "domain": domain,
            "size": len(members),
            "sample_titles": sample,
            "source_ids": source_ids,
            "max_hot": max_hot,
        }
        flat.append(row)
        clusters_by_domain[domain].append(row)

    for d in DOMAIN_ORDER:
        clusters_by_domain[d].sort(key=lambda x: (-float(x.get("max_hot") or 0), -int(x.get("size") or 0)))

    return {
        "domain_order": list(DOMAIN_ORDER),
        "clusters_by_domain": clusters_by_domain,
        "flat_clusters": flat,
        "meta": {
            "input_count": len(news_list),
            "cluster_count": len(flat),
            "max_clusters": mc,
            "join_similarity": join_similarity,
            "domain_method": "keyword",
        },
    }
