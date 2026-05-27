# coding=utf-8
import json
import os
import re
import time
import webbrowser
import sys
import yaml
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, TypedDict, Annotated, Tuple
from urllib.parse import quote
import operator
import pytz
import requests

from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, BaseMessage
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, END

from utils.hot_event_clusters import (
    DOMAIN_ORDER as HOT_DOMAIN_ORDER,
    cluster_news_by_domain,
    parse_llm_cluster_domain_json,
    rebuild_stats_with_domains,
)


def load_env_file(env_path: Optional[str] = None) -> None:
    candidate_paths = []
    if env_path:
        candidate_paths.append(Path(env_path))
    else:
        candidate_paths.append(Path(os.environ.get("ENV_FILE", ".env")))
        candidate_paths.append(Path.cwd() / ".env")
        candidate_paths.append(Path(__file__).resolve().parent / ".env")
        for base in [Path.cwd(), Path(__file__).resolve().parent]:
            current = base
            for _ in range(4):
                candidate_paths.append(current / ".env")
                if current.parent == current:
                    break
                current = current.parent
    unique_paths = []
    for p in candidate_paths:
        if p not in unique_paths:
            unique_paths.append(p)
    path = next((p for p in unique_paths if p.exists()), None)
    if not path:
        return
    encodings = ["utf-8", "utf-8-sig", "utf-16", "utf-16-le", "utf-16-be"]
    last_error = None
    for encoding in encodings:
        try:
            with open(path, "r", encoding=encoding) as f:
                for raw_line in f:
                    line = raw_line.strip()
                    if not line or line.startswith("#") or "=" not in line:
                        continue
                    key, value = line.split("=", 1)
                    key = key.strip()
                    if key.startswith("export "):
                        key = key[7:].strip()
                    key = key.lstrip("\ufeff")
                    value = value.strip().strip('"').strip("'")
                    if key and key not in os.environ:
                        os.environ[key] = value
            return
        except Exception as e:
            last_error = e
    if last_error:
        print(f"读取 .env 失败: {last_error}")


def apply_env_aliases() -> None:
    """将 Sona .env / model.yaml 映射为 hottopics 所需的 INSIGHT_/QUERY_ 变量（通义走 compatible-mode）。"""
    try:
        from utils.hot_topics_env import prepare_hot_topics_environment

        prepare_hot_topics_environment()
    except Exception:
        pass


load_env_file()
apply_env_aliases()


# === 配置加载 ===
def load_config():
    """加载配置文件"""
    config_path = os.environ.get("CONFIG_PATH", "config/config.yaml")
    if not Path(config_path).exists():
        if Path("../config/config.yaml").exists():
            config_path = "../config/config.yaml"
        else:
            print(f"配置文件 {config_path} 不存在，使用默认配置")
            return {"crawler": {"request_interval": 1000}, "platforms": []}

    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


CONFIG = load_config()

# run() 写入 docs/case candidate 快照后，供 CLI 等读取展示路径
LAST_HOT_SNAPSHOT_JSON: Optional[str] = None


# === 基础工具函数 ===
def get_beijing_time():
    return datetime.now(pytz.timezone("Asia/Shanghai"))


def ensure_directory_exists(directory: str):
    Path(directory).mkdir(parents=True, exist_ok=True)


# === 数据存储和加载工具函数 ===
def get_hourly_data_dir() -> Path:
    """获取按小时存储数据的目录"""
    data_dir = Path("data_langgraph_hourly")
    ensure_directory_exists(str(data_dir))
    return data_dir


def save_hourly_data(
    platform_id: str,
    platform_name: str,
    items: List[Dict[str, Any]],
    timestamp: Optional[datetime] = None,
) -> Path:
    """保存每小时的数据到文件"""
    if timestamp is None:
        timestamp = get_beijing_time()

    data_dir = get_hourly_data_dir()
    # 按日期和小时组织目录结构：YYYYMMDD/HH/
    date_str = timestamp.strftime("%Y%m%d")
    hour_str = timestamp.strftime("%H")
    hour_dir = data_dir / date_str / hour_str
    ensure_directory_exists(str(hour_dir))

    # 文件名：platform_id_timestamp.json
    filename = f"{platform_id}_{timestamp.strftime('%Y%m%d_%H%M%S')}.json"
    file_path = hour_dir / filename

    data = {
        "platform_id": platform_id,
        "platform_name": platform_name,
        "timestamp": timestamp.isoformat(),
        "items": items,
    }

    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    return file_path


def load_past_hours_data(lookback_hours: int = 12) -> List[Dict[str, Any]]:
    """加载过去N小时的所有平台数据"""
    data_dir = get_hourly_data_dir()
    now = get_beijing_time()
    cutoff = now - timedelta(hours=lookback_hours)

    all_items: List[Dict[str, Any]] = []

    # 遍历所有日期和小时目录
    for date_dir in sorted(data_dir.glob("20*"), reverse=True):
        date_str = date_dir.name
        try:
            date_obj = datetime.strptime(date_str, "%Y%m%d")
        except ValueError:
            continue

        for hour_dir in sorted(date_dir.glob("*"), reverse=True):
            hour_str = hour_dir.name
            try:
                hour_obj = int(hour_str)
                if hour_obj < 0 or hour_obj > 23:
                    continue
            except ValueError:
                continue

            # 构建完整时间戳
            file_timestamp = datetime.combine(date_obj.date(), datetime.min.time().replace(hour=hour_obj))
            file_timestamp = pytz.timezone("Asia/Shanghai").localize(file_timestamp)

            if file_timestamp < cutoff:
                continue

            # 加载该小时目录下的所有文件
            for json_file in hour_dir.glob("*.json"):
                try:
                    with open(json_file, "r", encoding="utf-8") as f:
                        data = json.load(f)
                        items = data.get("items", [])
                        # 为每个item添加时间戳信息
                        for item in items:
                            item["_fetch_timestamp"] = data.get("timestamp")
                            item["_platform_id"] = data.get("platform_id")
                            item["_platform_name"] = data.get("platform_name")
                        all_items.extend(items)
                except Exception as e:
                    print(f"加载文件失败 {json_file}: {e}")

    return all_items


def load_historical_ranks(lookback_hours: int = 24) -> Dict[tuple, Dict[str, Any]]:
    """
    加载历史排名数据，用于趋势分析
    返回: {(source_id, title): {"avg_rank": float, "min_rank": int, "count": int, "last_rank": int, "first_seen": datetime}}
    """
    data_dir = get_hourly_data_dir()
    now = get_beijing_time()
    cutoff = now - timedelta(hours=lookback_hours)

    # 聚合历史数据: (source_id, title) -> 排名列表和时间戳
    history: Dict[tuple, Dict[str, Any]] = {}

    for date_dir in sorted(data_dir.glob("20*"), reverse=False):
        date_str = date_dir.name
        try:
            date_obj = datetime.strptime(date_str, "%Y%m%d")
        except ValueError:
            continue

        for hour_dir in sorted(date_dir.glob("*"), reverse=False):
            hour_str = hour_dir.name
            try:
                hour_obj = int(hour_str)
                if hour_obj < 0 or hour_obj > 23:
                    continue
            except ValueError:
                continue

            file_timestamp = datetime.combine(date_obj.date(), datetime.min.time().replace(hour=hour_obj))
            file_timestamp = pytz.timezone("Asia/Shanghai").localize(file_timestamp)

            if file_timestamp < cutoff:
                continue

            for json_file in hour_dir.glob("*.json"):
                try:
                    with open(json_file, "r", encoding="utf-8") as f:
                        data = json.load(f)
                        items = data.get("items", [])
                        for item in items:
                            source_id = item.get("source_id") or item.get("source") or ""
                            title = item.get("title") or ""
                            if not source_id or not title:
                                continue
                            key = (source_id, title)
                            rank = item.get("rank")
                            if rank is not None:
                                if key not in history:
                                    history[key] = {"ranks": [], "first_seen": file_timestamp, "last_seen": file_timestamp}
                                history[key]["ranks"].append(rank)
                                # 更新首次出现时间
                                if file_timestamp < history[key]["first_seen"]:
                                    history[key]["first_seen"] = file_timestamp
                                # 更新最后出现时间
                                if file_timestamp > history[key]["last_seen"]:
                                    history[key]["last_seen"] = file_timestamp
                except Exception as e:
                    continue

    # 转换为统计信息
    result: Dict[tuple, Dict[str, Any]] = {}
    for key, data in history.items():
        ranks = data.get("ranks", [])
        if len(ranks) > 0:
            result[key] = {
                "avg_rank": sum(ranks) / len(ranks),
                "min_rank": min(ranks),
                "max_rank": max(ranks),
                "count": len(ranks),
                "last_rank": ranks[-1] if ranks else None,
                "first_seen": data.get("first_seen"),
                "last_seen": data.get("last_seen"),
            }

    return result


def calculate_trend(current_rank: int, historical_stats: Dict[str, Any]) -> Tuple[str, Optional[int]]:
    """
    计算趋势：基于历史数据比较当前排名
    返回: (趋势类型, 在榜天数)
    趋势类型: "new" | "up" | "down" | "stable"
    在榜天数: 首次出现至今的天数（仅当有历史数据时）
    """
    if not historical_stats:
        return ("new", None)  # 新上榜

    avg_rank = historical_stats.get("avg_rank", 0)
    min_rank = historical_stats.get("min_rank", 999)
    last_rank = historical_stats.get("last_rank", 999)

    # 计算在榜天数
    days_on_list = None
    first_seen = historical_stats.get("first_seen")
    if first_seen:
        now = get_beijing_time()
        # 计算天数差
        delta = now - first_seen
        days_on_list = max(1, delta.days)  # 至少1天

    # 阈值设置
    RANK_CHANGE_THRESHOLD = 3  # 排名变化超过3位视为上升/下降

    # 比较当前排名与历史平均/上次排名
    if current_rank < last_rank - RANK_CHANGE_THRESHOLD:
        return ("up", days_on_list)  # 上升
    elif current_rank > last_rank + RANK_CHANGE_THRESHOLD:
        return ("down", days_on_list)  # 下降
    elif abs(current_rank - avg_rank) <= RANK_CHANGE_THRESHOLD:
        return ("stable", days_on_list)  # 稳定
    else:
        # 综合判断
        if current_rank < avg_rank:
            return ("up", days_on_list)
        elif current_rank > avg_rank:
            return ("down", days_on_list)
        return ("stable", days_on_list)


def html_escape(text: str) -> str:
    """HTML转义"""
    if not isinstance(text, str):
        text = str(text) if text is not None else ""
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
        .replace("'", "&#x27;")
    )


def _build_domain_cluster_section(domain_cluster_stats: Optional[Dict[str, Any]]) -> str:
    """生成「事件聚类与领域」HTML 片段。"""
    if not domain_cluster_stats:
        return ""
    domain_order = domain_cluster_stats.get("domain_order") or []
    cbd = domain_cluster_stats.get("clusters_by_domain") or {}
    if not any((cbd.get(d) or []) for d in domain_order):
        return ""
    parts: List[str] = [
        '            <div class="section">\n',
        '                <div class="section-title">事件聚类与领域</div>\n',
        '                <div class="section-body domain-cluster-intro">'
        "基于抓取标题的相似度合并为事件簇，并按主题领域做关键词标注"
        "（与下方「热点事件」的舆情类型为不同维度）。</div>\n",
    ]
    for domain in domain_order:
        clusters = cbd.get(domain) or []
        if not clusters:
            continue
        parts.append(
            f'            <div class="topic-category">{html_escape(domain)}'
            f' <span class="domain-cluster-count">（{len(clusters)} 簇）</span></div>\n'
        )
        for cl in clusters:
            rep = html_escape(str(cl.get("representative_title") or ""))
            size = int(cl.get("size") or 0)
            max_hot = cl.get("max_hot")
            try:
                hot_s = f"{float(max_hot):.0f}"
            except (TypeError, ValueError):
                hot_s = "—"
            sids_raw = cl.get("source_ids") or []
            sids = ", ".join(html_escape(str(x)) for x in sids_raw[:10])
            samples = [s for s in (cl.get("sample_titles") or []) if s][:4]
            samples_html = "".join(
                f'                <div class="cluster-sample">· {html_escape(str(s))}</div>\n' for s in samples
            )
            parts.append(
                "            <div class=\"cluster-card\">\n"
                f'                <div class="cluster-rep"><span class="domain-badge">{html_escape(domain)}</span> {rep}</div>\n'
                f'                <div class="cluster-meta">{size} 条 · 峰值热度 {hot_s} · 平台: {sids or "—"}</div>\n'
                f'                <div class="cluster-samples">{samples_html}</div>\n'
                "            </div>\n"
            )
    parts.append("            </div>\n")
    return "".join(parts)


def format_forum_discussion_html(raw: str) -> str:
    """
    将 ForumNode 输出的「重大事件剖析」拆成模块化卡片（Insight / Media / Query / 综合判断）。
    若无法识别小节标题，则回退为整块转义展示。
    """
    text = (raw or "").strip()
    if not text:
        return '<p class="forum-empty">暂无重大事件剖析。</p>'

    if text.startswith("【重大事件剖析】"):
        text = re.sub(r"^【重大事件剖析】\s*", "", text, count=1).lstrip("：:").strip()

    _MARKERS: List[Tuple[str, str, re.Pattern[str]]] = [
        ("insight", "Insight · 深度观察", re.compile(r"【\s*Insight\s*深度观察\s*】")),
        ("media", "Media · 媒体观点", re.compile(r"【\s*Media\s*媒体观点\s*】")),
        ("query", "Query · 关键事实", re.compile(r"【\s*Query\s*关键事实\s*】")),
        ("judgment", "综合判断", re.compile(r"(?:^|\n)\s*综合判断\s*[:：]?\s*")),
    ]

    def _next_match(from_pos: int) -> Optional[Tuple[int, int, str, str]]:
        best: Optional[Tuple[int, int, str, str]] = None
        for kind, title, cre in _MARKERS:
            m = cre.search(text, from_pos)
            if not m:
                continue
            if best is None or m.start() < best[0]:
                best = (m.start(), m.end(), kind, title)
        return best

    pos = 0
    preamble = ""
    first = _next_match(0)
    if first and first[0] > 0:
        preamble = text[: first[0]].strip()

    segments: List[Tuple[str, str, str]] = []
    while pos < len(text):
        cur = _next_match(pos)
        if not cur:
            break
        start, end, kind, title = cur
        nxt = _next_match(end)
        body_end = nxt[0] if nxt else len(text)
        body = text[end:body_end].strip()
        if body:
            segments.append((kind, title, body))
        pos = body_end

    if len(segments) < 1:
        return f'<div class="forum-legacy">{html_escape(text).replace(chr(10), "<br/>")}</div>'

    def _body_html(body: str) -> str:
        paras = [p.strip() for p in re.split(r"\n\s*\n+", body) if p.strip()]
        if not paras:
            return ""
        blocks: List[str] = []
        for p in paras:
            inner = html_escape(p).replace("\n", "<br/>")
            blocks.append(f'<p class="forum-mod-para">{inner}</p>')
        return "\n".join(blocks)

    cards: List[str] = ['<div class="forum-modules">\n']
    if preamble and len(preamble) > 8:
        cards.append(
            '            <div class="forum-mod forum-mod-preamble">\n'
            '                <header class="forum-mod-head"><span class="forum-mod-kicker">导读</span></header>\n'
            '                <div class="forum-mod-content">'
            + html_escape(preamble).replace("\n", "<br/>")
            + "</div>\n            </div>\n"
        )
    for kind, title, body in segments:
        body_html = _body_html(body)
        if not body_html:
            continue
        cards.append(
            f'            <article class="forum-mod forum-mod-{kind}">\n'
            f'                <header class="forum-mod-head">\n'
            f'                    <span class="forum-mod-kicker">{html_escape(title)}</span>\n'
            f"                </header>\n"
            f'                <div class="forum-mod-content">\n{body_html}\n'
            f"                </div>\n"
            f"            </article>\n"
        )
    cards.append("            </div>")
    return "".join(cards)


def _is_docker_env() -> bool:
    """检测是否在 Docker 容器内运行（容器内不自动打开浏览器）"""
    if os.environ.get("DOCKER_CONTAINER") == "true":
        return True
    return os.path.exists("/.dockerenv")


def _request_with_proxy_fallback(url: str, headers: Dict[str, str], timeout: int = 15) -> Optional[str]:
    """
    带代理回退的 GET 请求：
    1) 若配置启用代理，先走代理；
    2) 代理失败时自动回退直连，避免因本地代理不可用导致全量抓取失败。
    """
    crawler_config = CONFIG.get("crawler", {}) if CONFIG else {}
    proxy_url = crawler_config.get("default_proxy", "http://127.0.0.1:7897")
    use_proxy = bool(crawler_config.get("use_proxy", False) and proxy_url)

    attempts: List[Optional[Dict[str, str]]] = []
    if use_proxy:
        attempts.append({"http": proxy_url, "https": proxy_url})
    attempts.append(None)  # 始终保留直连兜底

    last_error: Optional[Exception] = None
    for proxies in attempts:
        try:
            response = requests.get(url, headers=headers, timeout=timeout, proxies=proxies)
            if response.status_code == 200:
                return response.text
        except Exception as e:
            last_error = e
            # 若代理失败，继续尝试直连；若直连失败则最终返回 None
            continue
    if last_error:
        raise last_error
    return None


# === 状态定义 ===
class TrendState(TypedDict):
    """定义工作流状态"""

    # 原始新闻数据
    news_data: Dict[str, Any]
    # 原始抓取的新闻列表（用于并行节点合并）
    raw_items: Annotated[List[Dict[str, Any]], operator.add]
    # 结构化分析结果
    analysis_result: Dict[str, Any]
    # 十一类舆情分类统计结果
    classification_stats: Dict[str, Any]
    # 抓取条目：相似度聚类 + 主题领域（与舆情十一类正交）
    domain_cluster_stats: Dict[str, Any]
    # 论坛讨论内容
    forum_discussion: str
    # 最终HTML报告
    html_report: str
    # 执行过程中的消息记录（可选，用于调试）
    messages: Annotated[List[BaseMessage], operator.add]
    # 错误信息
    error: Optional[str]


# === 节点定义 ===
class BaseFetchNode:
    """基础抓取节点"""

    def __init__(self, platform_id: str, platform_name: str):
        self.platform_id = platform_id
        self.platform_name = platform_name

    def fetch_data(self) -> Optional[str]:
        """抓取数据"""
        url = f"https://newsnow.busiyi.world/api/s?id={self.platform_id}&latest"
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Accept": "application/json, text/plain, */*",
            "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
            "Referer": "https://newsnow.busiyi.world/",
        }

        try:
            return _request_with_proxy_fallback(url, headers=headers, timeout=15)
        except Exception as e:
            print(f"Error fetching {self.platform_name} ({self.platform_id}): {e}")
        return None

    def parse_data(self, raw_data: str) -> List[Dict[str, Any]]:
        """解析数据"""
        items = []
        try:
            data = json.loads(raw_data)
            raw_items = data.get("items", [])
            # 每个平台只保留前 10 条高热度话题
            top_items = raw_items[:10]
            for idx, item in enumerate(top_items, 1):
                items.append(
                    {
                        "source_id": self.platform_id,
                        "source": self.platform_name,
                        "source_name": self.platform_name,
                        "title": item.get("title"),
                        "rank": idx,
                        "url": item.get("url", ""),
                        "mobile_url": item.get("mobileUrl", ""),
                        "hot_value": item.get("hotValue", 0),
                    }
                )
        except Exception as e:
            print(f"解析 {self.platform_name} 数据失败: {e}")
        return items

    def __call__(self, state: TrendState) -> TrendState:
        """执行抓取"""
        print(f"--- [Fetch{self.platform_name}Node] 开始抓取 {self.platform_name} ---")
        timestamp = get_beijing_time()

        raw_data = self.fetch_data()
        if not raw_data:
            print(f"⚠️  警告: {self.platform_name} 抓取失败")
            # 返回空列表，LangGraph会自动合并
            return {"raw_items": []}

        items = self.parse_data(raw_data)
        print(f"✅ {self.platform_name} 抓取完成，共 {len(items)} 条新闻")

        # 保存到每小时数据目录
        try:
            save_path = save_hourly_data(self.platform_id, self.platform_name, items, timestamp)
            print(f"   数据已保存: {save_path}")
        except Exception as e:
            print(f"   保存数据失败: {e}")

        # 返回抓取的数据（LangGraph会自动合并到raw_items）
        return {"raw_items": items}


# 为每个平台创建Fetch节点
class FetchWeiboNode(BaseFetchNode):
    def __init__(self):
        super().__init__("weibo", "微博")


class FetchZhihuNode(BaseFetchNode):
    def __init__(self):
        super().__init__("zhihu", "知乎")


class FetchToutiaoNode(BaseFetchNode):
    def __init__(self):
        super().__init__("toutiao", "今日头条")


class FetchBaiduNode(BaseFetchNode):
    def __init__(self):
        super().__init__("baidu", "百度热搜")


class FetchDouyinNode(BaseFetchNode):
    def __init__(self):
        super().__init__("douyin", "抖音")


class FetchBilibiliNode(BaseFetchNode):
    def __init__(self):
        super().__init__("bilibili-hot-search", "bilibili 热搜")


class FetchThepaperNode(BaseFetchNode):
    def __init__(self):
        super().__init__("thepaper", "澎湃新闻")


class FetchTiebaNode(BaseFetchNode):
    def __init__(self):
        super().__init__("tieba", "贴吧")


class FetchIfengNode(BaseFetchNode):
    def __init__(self):
        super().__init__("ifeng", "凤凰网")


class FetchClsNode(BaseFetchNode):
    def __init__(self):
        super().__init__("cls-hot", "财联社热门")


class FetchWallstreetcnNode(BaseFetchNode):
    def __init__(self):
        super().__init__("wallstreetcn-hot", "华尔街见闻")


# === 新增平台 Fetch 节点 ===
class FetchXueqiuNode(BaseFetchNode):
    """雪球 - 财经舆情"""
    def __init__(self):
        super().__init__("xueqiu", "雪球")


class Fetch36krNode(BaseFetchNode):
    """36氪 - 科技热点"""
    def __init__(self):
        super().__init__("36kr", "36氪")


class FetchHupuNode(BaseFetchNode):
    """虎扑 - 体育/社区舆情"""
    def __init__(self):
        super().__init__("hupu", "虎扑")


class FetchV2exNode(BaseFetchNode):
    """V2ex - 技术社区热点"""
    def __init__(self):
        super().__init__("v2ex", "V2ex")


class FetchSspaiNode(BaseFetchNode):
    """少数派 - 数字产品"""
    def __init__(self):
        super().__init__("sspai", "少数派")


class FetchKuaishouNode(BaseFetchNode):
    """快手 - 短视频热点"""
    def __init__(self):
        super().__init__("kuaishou", "快手")


class FetchIthomeNode(BaseFetchNode):
    """IT之家 - IT/数码"""
    def __init__(self):
        super().__init__("ithome", "IT之家")


class FetchChongbuluoNode(BaseFetchNode):
    """抽屉新热榜 - 综合热点"""
    def __init__(self):
        super().__init__("chongbuluo", "抽屉新热榜")


class SpiderNode:
    """抓取节点"""

    def __init__(self):
        # 默认平台列表（向后兼容）
        self.sources = {
            "weibo": "微博热搜",
            "zhihu": "知乎热榜",
            "bilibili-hot-search": "B站热搜",
            "toutiao": "今日头条",
            "douyin": "抖音热榜",
            "36kr": "36氪",
            "sspai": "少数派",
        }
        if CONFIG and "platforms" in CONFIG:
            # 只包含 enabled: true 的平台（支持 enabled 字段的动态配置）
            self.sources = {
                p["id"]: p.get("name", p["id"])
                for p in CONFIG["platforms"]
                if p.get("enabled", True)  # 默认启用
            }

    def fetch_data(self, id_value: str) -> Optional[str]:
        url = f"https://newsnow.busiyi.world/api/s?id={id_value}&latest"
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Accept": "application/json, text/plain, */*",
            "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
            "Referer": "https://newsnow.busiyi.world/",
        }

        try:
            return _request_with_proxy_fallback(url, headers=headers, timeout=15)
        except Exception as e:
            print(f"Error fetching {id_value}: {e}")
        return None

    def __call__(self, state: TrendState) -> TrendState:
        print("--- [SpiderNode] 开始抓取新闻 ---")
        news_list: List[Dict[str, Any]] = []
        for source_id, source_name in self.sources.items():
            print(f"正在抓取: {source_name} ({source_id})")
            raw_data = self.fetch_data(source_id)
            if raw_data:
                try:
                    data = json.loads(raw_data)
                    items = data.get("items", [])
                    # 为了提高样本量，这里尽量多保留一些热搜项（例如前 30 条）
                    top_items = items[:30]
                    for idx, item in enumerate(top_items, 1):
                        news_list.append(
                            {
                                "source_id": source_id,
                                "source": source_name,
                                "source_name": source_name,
                                "title": item.get("title"),
                                # 平台内排行（1 表示热搜第1）
                                "rank": idx,
                                "url": item.get("url", ""),
                                "mobile_url": item.get("mobileUrl", ""),
                                "hot_value": item.get("hotValue", 0),
                            }
                        )
                except Exception as e:
                    print(f"解析 {source_name} 失败: {e}")
            time.sleep(1)

        print(f"本次抓取完成，共 {len(news_list)} 条新闻")
        if len(news_list) == 0:
            print("⚠️  警告: 未能抓取到任何新闻，可能是 API 服务问题或网络连接问题")

        # 将本次抓取结果保存为快照，并合并最近 24 小时内的历史快照，扩大样本量
        try:
            snapshot_dir = Path("data_langgraph")
            ensure_directory_exists(str(snapshot_dir))
            timestamp = get_beijing_time()
            snapshot_name = f"snapshot_{timestamp.strftime('%Y%m%d_%H%M%S')}.json"
            snapshot_path = snapshot_dir / snapshot_name
            with open(snapshot_path, "w", encoding="utf-8") as f:
                json.dump({"timestamp": timestamp.isoformat(), "items": news_list}, f, ensure_ascii=False)
        except Exception as e:
            print(f"保存快照失败: {e}")

        # 合并最近 24 小时的快照
        aggregated: Dict[tuple, Dict[str, Any]] = {}
        now = get_beijing_time()
        lookback_hours = 24
        cutoff = now - timedelta(hours=lookback_hours)

        try:
            for p in snapshot_dir.glob("snapshot_*.json"):
                try:
                    ts_str = p.stem.replace("snapshot_", "")
                    dt = datetime.strptime(ts_str, "%Y%m%d_%H%M%S")
                except Exception:
                    continue
                if dt < cutoff:
                    continue
                with open(p, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    items = data.get("items") or []
                    for item in items:
                        key = (item.get("source_id") or item.get("source") or "", item.get("title") or "")
                        if not key[1]:
                            continue
                        existing = aggregated.get(key)
                        if existing is None:
                            aggregated[key] = dict(item)
                        else:
                            # 保留更靠前的排名和更高的热度
                            rank_new = item.get("rank")
                            rank_old = existing.get("rank")
                            if rank_new is not None and (rank_old is None or rank_new < rank_old):
                                existing["rank"] = rank_new
                            hot_new = item.get("hot_value")
                            hot_old = existing.get("hot_value")
                            if hot_new is not None and (hot_old is None or hot_new > hot_old):
                                existing["hot_value"] = hot_new
            merged_list = list(aggregated.values()) if aggregated else news_list
            print(f"合并最近{lookback_hours}小时快照后，共 {len(merged_list)} 条去重新闻")
        except Exception as e:
            print(f"合并历史快照失败，退回使用本次抓取结果: {e}")
            merged_list = news_list

        return {"news_data": {"news_list": merged_list}}


class NormalizeNewsNode:
    """清洗、去重、排序节点 - 带趋势分析"""

    def __init__(self):
        pass

    def clean_title(self, title: str) -> str:
        """清洗标题：去除多余空格、特殊字符等"""
        if not title:
            return ""
        # 去除首尾空格
        title = title.strip()
        # 去除多个连续空格
        import re

        title = re.sub(r"\s+", " ", title)
        return title

    def normalize_news(
        self,
        raw_items: List[Dict[str, Any]],
        historical_ranks: Optional[Dict[tuple, Dict[str, Any]]] = None,
    ) -> List[Dict[str, Any]]:
        """清洗、去重、排序新闻 - 带趋势分析"""
        if not raw_items:
            return []

        # 1. 清洗：清理标题
        cleaned_items = []
        for item in raw_items:
            title = item.get("title", "")
            cleaned_title = self.clean_title(title)
            if cleaned_title:
                item_copy = dict(item)
                item_copy["title"] = cleaned_title
                cleaned_items.append(item_copy)

        # 2. 去重：基于 (source_id, title) 去重，保留历史排名信息
        seen: Dict[tuple, Dict[str, Any]] = {}
        deduplicated_items = []
        for item in cleaned_items:
            source_id = item.get("source_id") or item.get("source") or ""
            title = item.get("title") or ""
            key = (source_id, title)
            if not title:  # 标题为空则跳过
                continue

            existing = seen.get(key)
            if existing is None:
                seen[key] = item
                # 记录历史排名信息用于后续趋势计算
                if historical_ranks and key in historical_ranks:
                    item["_historical_stats"] = historical_ranks[key]
                deduplicated_items.append(item)
            else:
                # 保留更靠前的排名和更高的热度
                rank_new = item.get("rank")
                rank_old = existing.get("rank")
                if rank_new is not None and (rank_old is None or rank_new < rank_old):
                    existing["rank"] = rank_new

                hot_new = item.get("hot_value")
                hot_old = existing.get("hot_value")
                if hot_new is not None and (hot_old is None or hot_new > hot_old):
                    existing["hot_value"] = hot_new

        # 3. 计算趋势并添加趋势标记
        for item in deduplicated_items:
            current_rank = item.get("rank", 999)
            historical_stats = item.get("_historical_stats")
            trend, days_on_list = calculate_trend(current_rank, historical_stats)
            item["trend"] = trend
            item["days_on_list"] = days_on_list  # 在榜天数
            # 清理内部使用的临时字段
            item.pop("_historical_stats", None)

        # 4. 排序：按平台权重和平台内排名排序
        platform_order = ["weibo", "baidu", "douyin", "zhihu", "toutiao", "tieba", "thepaper", "cls-hot", "ifeng", "wallstreetcn-hot"]
        platform_index = {pid: idx for idx, pid in enumerate(platform_order)}

        def sort_key(item: Dict[str, Any]) -> tuple:
            source_id = item.get("source_id") or item.get("source") or "other"
            platform_rank = platform_index.get(source_id, len(platform_index))
            item_rank = item.get("rank") or 9999
            return (platform_rank, item_rank)

        sorted_items = sorted(deduplicated_items, key=sort_key)

        return sorted_items

    def __call__(self, state: TrendState) -> TrendState:
        """执行清洗、去重、排序 - 带趋势分析"""
        print("--- [NormalizeNewsNode] 开始清洗、去重、排序 + 趋势分析 ---")

        # 从状态中获取原始数据
        news_data = state.get("news_data", {})
        raw_items = news_data.get("raw_items", [])

        # 加载过去12小时的历史数据
        print("正在加载过去12小时的历史数据...")
        historical_items = load_past_hours_data(lookback_hours=12)
        print(f"从历史数据中加载了 {len(historical_items)} 条新闻")

        # 加载过去24小时的历史排名用于趋势计算
        print("正在加载历史排名数据用于趋势分析...")
        historical_ranks = load_historical_ranks(lookback_hours=24)
        print(f"历史排名数据: {len(historical_ranks)} 个话题")

        # 若当前抓取与近12小时都为空，扩大窗口做离线兜底，提升 /hot 在网络异常时的可用性
        if not raw_items and not historical_items:
            fallback_hours = max(24, int(os.environ.get("HOT_FALLBACK_LOOKBACK_HOURS", "168")))
            print(f"当前无实时数据，尝试回退加载最近{fallback_hours}小时历史快照...")
            historical_items = load_past_hours_data(lookback_hours=fallback_hours)
            print(f"回退窗口加载到 {len(historical_items)} 条新闻")

        # 合并当前抓取的数据和历史数据
        all_items = raw_items + historical_items

        # 执行清洗、去重、排序，传入历史排名用于趋势计算
        normalized_items = self.normalize_news(all_items, historical_ranks)

        # 统计趋势分布
        trend_counts = {"new": 0, "up": 0, "down": 0, "stable": 0}
        for item in normalized_items:
            trend = item.get("trend", "stable")
            if trend in trend_counts:
                trend_counts[trend] += 1

        print(f"✅ 清洗、去重、排序完成，共 {len(normalized_items)} 条新闻")
        print(f"📊 趋势分布: 新上榜 {trend_counts['new']} | 上升 {trend_counts['up']} | 下降 {trend_counts['down']} | 稳定 {trend_counts['stable']}")

        return {"news_data": {"news_list": normalized_items, "raw_items": []}, "raw_items": []}


def _hot_domain_llm_enabled() -> bool:
    """默认开启批量 LLM 领域标注；设 HOT_DOMAIN_USE_LLM=0 可关闭。需 INSIGHT_ENGINE_API_KEY。"""
    v = os.environ.get("HOT_DOMAIN_USE_LLM", "1").strip().lower()
    if v in ("0", "false", "no", "off"):
        return False
    return bool(os.environ.get("INSIGHT_ENGINE_API_KEY"))


def _batch_llm_cluster_domains(flat_clusters: List[Dict[str, Any]], domain_order: List[str]) -> Dict[str, str]:
    """将各簇代表标题一次性交给与 Insight 相同的 OpenAI 兼容端点，返回 cluster_id -> 领域。"""
    if not flat_clusters:
        return {}
    api_key = os.environ.get("INSIGHT_ENGINE_API_KEY")
    if not api_key:
        return {}
    base_url = os.environ.get("INSIGHT_ENGINE_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
    model = os.environ.get("INSIGHT_ENGINE_MODEL_NAME", "qwen-plus")
    lines = "\n".join(
            f'{r.get("cluster_id")}\t{str(r.get("representative_title") or "").strip()}'
            for r in flat_clusters
        )
    allowed = "\n".join(f"- {d}" for d in domain_order)
    system_prompt = (
        "你是中文新闻「主题领域」分类器。用户给出若干事件簇的代表标题，请为每个 cluster_id 从允许列表中"
        "选且仅能选一个领域标签。标签必须与列表中的原文完全一致（含标点、不得改写或缩写）。"
        "只输出一个 JSON 对象，键为 cluster_id（如 c0），值为领域名称。不要 Markdown 代码围栏，不要解释。"
    )
    user_prompt = f"允许的领域名称（必须完全一致）：\n{allowed}\n\n每行一个簇，格式为：cluster_id<TAB>代表标题\n{lines}"
    try:
        llm = ChatOpenAI(
            api_key=api_key,
            base_url=base_url,
            model=model,
            temperature=0.0,
            timeout=120,
        )
        resp = llm.invoke([SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)])
        raw = getattr(resp, "content", None) or str(resp)
        valid = frozenset(domain_order)
        return parse_llm_cluster_domain_json(str(raw), valid)
    except Exception as e:
        print(f"⚠️  LLM 领域分类失败（沿用关键词）: {e}")
        return {}


class ClusterDomainNode:
    """对 normalize 后的全量新闻做标题相似度聚类，并按主题领域打标签。"""

    def __init__(self) -> None:
        pass

    def __call__(self, state: TrendState) -> TrendState:
        print("--- [ClusterDomainNode] 事件聚类与领域分类 ---")
        news_list = state.get("news_data", {}).get("news_list") or []
        stats = cluster_news_by_domain(news_list)
        n = int(stats.get("meta", {}).get("cluster_count") or 0)
        print(f"✅ 聚类完成，共 {n} 个事件簇")

        order = list(stats.get("domain_order") or HOT_DOMAIN_ORDER)
        if _hot_domain_llm_enabled():
            mapping = _batch_llm_cluster_domains(stats.get("flat_clusters") or [], order)
            if mapping:
                stats = rebuild_stats_with_domains(stats, mapping, method="llm_batch")
                print(f"✅ LLM 已标注 {len(mapping)} 个簇的领域（{stats.get('meta', {}).get('domain_method')}）")
            else:
                meta = stats.setdefault("meta", {})
                meta["domain_method"] = meta.get("domain_method") or "keyword"
        return {"domain_cluster_stats": stats}


class StartFetchNode:
    """开始抓取的入口节点"""

    def __init__(self):
        pass

    def __call__(self, state: TrendState) -> TrendState:
        """初始化抓取状态"""
        print("--- [StartFetchNode] 开始并行抓取所有平台 ---")
        return {"raw_items": []}


class MergeFetchNode:
    """合并所有Fetch节点的结果"""

    def __init__(self):
        pass

    def __call__(self, state: TrendState) -> TrendState:
        """合并所有Fetch节点的结果"""
        print("--- [MergeFetchNode] 合并所有抓取结果 ---")
        # raw_items已经通过Annotated自动合并了
        raw_items = state.get("raw_items", [])
        print(f"✅ 所有平台抓取完成，共 {len(raw_items)} 条原始新闻")

        # 将raw_items转移到news_data中
        return {"news_data": {"raw_items": raw_items}}


class InsightNode:
    """分析节点"""

    def __init__(self):
        api_key = os.environ.get("INSIGHT_ENGINE_API_KEY")
        if api_key:
            self.llm = ChatOpenAI(
                api_key=api_key,
                base_url=os.environ.get("INSIGHT_ENGINE_BASE_URL", "https://api.moonshot.cn/v1"),
                model=os.environ.get("INSIGHT_ENGINE_MODEL_NAME", "moonshot-v1-8k"),
                temperature=0.7,
            )
        else:
            self.llm = None

    @staticmethod
    def _fallback_insight(news_list: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        无模型可用时的规则兜底：按排名/热度提取热点标题。
        """
        if not news_list:
            return {"top_topics": [], "summary": "今日无新闻数据"}
        # 按平台内排名优先、热度次优先
        sorted_news = sorted(
            news_list,
            key=lambda x: (
                int(x.get("rank") or 9999),
                -float(x.get("hot_value") or 0),
            ),
        )
        top_topics: List[Dict[str, Any]] = []
        seen: set[str] = set()
        for item in sorted_news:
            title = str(item.get("title") or "").strip()
            if not title or title in seen:
                continue
            seen.add(title)
            rank = int(item.get("rank") or 50)
            hot = float(item.get("hot_value") or 0)
            heat_score = max(20.0, min(100.0, 100.0 - rank * 2 + (hot / 1000000.0)))
            top_topics.append(
                {
                    "topic": title,
                    "sentiment": "中性",
                    "comment": f"来自{item.get('source_name') or item.get('source') or '多平台'}热榜，建议持续跟踪事件演化。",
                    "heat_score": round(heat_score, 1),
                    "category": "其他",
                }
            )
            if len(top_topics) >= 8:
                break
        return {
            "top_topics": top_topics,
            "summary": "当前报告由规则引擎生成（未启用大模型深度归因），建议在模型可用时重新生成以获得更细粒度洞察。",
        }

    def __call__(self, state: TrendState) -> TrendState:
        print("--- [InsightNode] 开始分析舆情 ---")
        news_list = state.get("news_data", {}).get("news_list", [])
        if not news_list:
            print("⚠️  警告: 没有新闻数据可分析，跳过分析步骤")
            return {"error": "No news to analyze", "analysis_result": {"top_topics": [], "summary": "今日无新闻数据"}}

        if self.llm is None:
            print("⚠️  未检测到 INSIGHT_ENGINE_API_KEY，使用规则兜底分析")
            return {"analysis_result": self._fallback_insight(news_list)}

        # 优化1: 智能筛选新闻，减少输入量，避免触发API限制
        # 优先选择：高排名、高热度、重要平台的新闻
        def news_priority_score(news_item: Dict[str, Any]) -> float:
            """计算新闻优先级分数"""
            source_id = news_item.get("source_id", "").lower()
            rank = news_item.get("rank", 9999)
            hot_value = news_item.get("hot_value", 0)

            # 平台权重（从高到低）
            platform_weights = {
                "weibo": 10.0,
                "baidu": 9.0,
                "douyin": 8.0,
                "zhihu": 7.0,
                "toutiao": 6.0,
                "tieba": 5.0,
                "thepaper": 4.0,
                "cls-hot": 3.0,
                "ifeng": 2.0,
                "wallstreetcn-hot": 1.0,
            }
            platform_weight = platform_weights.get(source_id, 0.5)

            # 排名权重（排名越靠前分数越高）
            rank_weight = max(0, 30 - rank) / 30.0

            # 热度权重（归一化到0-1）
            hot_weight = min(1.0, hot_value / 1000000.0) if hot_value else 0

            # 综合分数
            score = platform_weight * 0.5 + rank_weight * 0.3 + hot_weight * 0.2
            return score

        # 按优先级排序（不去除数量限制，直接分析全部新闻）
        sorted_news = sorted(news_list, key=news_priority_score, reverse=True)
        # 不再限制数量，直接分析所有新闻
        selected_news = sorted_news

        if len(news_list) > 500:
            print(f"📊 分析全部 {len(news_list)} 条新闻（已移除120条限制）")

        # 优化2: 简化数据格式，只保留必要信息
        news_text = "\n".join([f"- {n['title']}" for n in selected_news])  # 移除来源信息，减少字符数

        # 优化3: 简化Prompt，移除敏感关键词示例，使用更中性的表述
        system_prompt = "你是一个专业的舆情分析助手，擅长从新闻标题中提取热点事件并进行分类分析。"
        user_prompt = f"""
请分析以下热点新闻标题列表，提取最重要的舆情事件：

{news_text}

任务要求：
1. 合并描述同一事件的不同标题，提取核心事件
2. 输出最多8个最重要的事件，覆盖不同领域
3. 每个事件包含字段：
   - topic: 具体事件标题（不要使用抽象类别）
   - sentiment: 情感倾向（正面/负面/中性）
   - comment: 简要点评（1-2句话）
   - heat_score: 热度值（0-100，基于相关标题数量和重要性）
   - category: 类别（从以下选项选择）：
     ["经济类舆论", "突发事件舆论", "法治类舆论", "文娱类舆论", "科教类舆论", "国际关系类舆论", "健康类舆论", "治理类舆论", "民生类舆论", "生态环境类舆论", "其他"]
4. 提供整体趋势summary（2-3句话）

请以JSON格式返回（不要使用Markdown代码块）：
{{
    "top_topics": [
        {{"topic": "事件标题", "sentiment": "负面", "comment": "点评", "heat_score": 85.0, "category": "类别"}},
        ...
    ],
    "summary": "趋势总结"
}}
"""

        try:
            messages = [SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)]

            # 优化4: 添加小延迟，避免请求频率过高
            time.sleep(0.5)

            response = self.llm.invoke(messages)
            content = response.content

            # 清理可能的 Markdown 标记
            json_str = content.replace("```json", "").replace("```", "").strip()

            # 尝试解析JSON，如果失败则尝试修复不完整的JSON
            try:
                analysis_result = json.loads(json_str)
            except json.JSONDecodeError as json_err:
                # 尝试修复不完整的JSON
                print(f"⚠️  JSON解析失败，尝试修复: {json_err}")
                try:
                    import re

                    # 方法1: 尝试找到所有完整的topic对象（通过平衡大括号）
                    # 查找 "top_topics": [ 之后的内容
                    topics_match = re.search(r'"top_topics"\s*:\s*\[', json_str)
                    if topics_match:
                        start_pos = topics_match.end()
                        # 查找所有完整的topic对象
                        complete_objects = []
                        brace_count = 0
                        in_string = False
                        escape_next = False
                        obj_start = None

                        for i in range(start_pos, len(json_str)):
                            char = json_str[i]
                            if escape_next:
                                escape_next = False
                                continue
                            if char == "\\":
                                escape_next = True
                                continue
                            if char == '"' and not escape_next:
                                in_string = not in_string
                                continue
                            if not in_string:
                                if char == "{":
                                    if brace_count == 0:
                                        obj_start = i
                                    brace_count += 1
                                elif char == "}":
                                    brace_count -= 1
                                    if brace_count == 0 and obj_start is not None:
                                        # 找到了一个完整的对象
                                        complete_objects.append((obj_start, i + 1))
                                        obj_start = None

                        # 如果找到了完整的对象，尝试构建有效的JSON
                        if complete_objects:
                            # 提取所有完整对象的文本
                            obj_texts = [json_str[start:end] for start, end in complete_objects]
                            # 构建有效的JSON
                            partial_json = '{"top_topics": [' + ",".join(obj_texts) + '], "summary": "JSON响应被截断，仅提取了部分数据"}'
                            try:
                                analysis_result = json.loads(partial_json)
                                print(f"⚠️  JSON被截断，已修复并提取了 {len(analysis_result.get('top_topics', []))} 个话题")
                            except Exception as parse_err:
                                print(f"⚠️  修复后的JSON仍无法解析: {parse_err}")
                                raise json_err
                        else:
                            raise json_err
                    else:
                        raise json_err
                except Exception as fix_err:
                    # 如果修复也失败，返回一个默认结构，但保留错误信息
                    print(f"⚠️  JSON解析失败且无法修复: {json_err}")
                    # 尝试至少提取一些文本信息作为summary
                    summary_text = "分析过程中JSON响应被截断，无法完整解析"
                    if "summary" in json_str.lower():
                        summary_match = re.search(r'"summary"\s*:\s*"([^"]*)"', json_str)
                        if summary_match:
                            summary_text = summary_match.group(1) + " (部分数据)"

                    analysis_result = {"top_topics": [], "summary": summary_text}

            return {"analysis_result": analysis_result}
        except Exception as e:
            print(f"InsightNode Error: {e}")
            # 网络抖动/模型不可用时，回退到规则分析，避免后续节点拿不到 top_topics
            fallback_result = self._fallback_insight(news_list)
            fallback_result["summary"] = (
                f"{fallback_result.get('summary', '')}（LLM失败自动回退：{str(e)[:160]}）"
            )
            return {"error": str(e), "analysis_result": fallback_result}


class ClassifyNode:
    """十一类舆情分类统计节点"""

    def __init__(self):
        # 十一类舆情分类标准（采用《十大舆情分类》标准）
        self.category_order = [
            "经济类舆论",
            "突发事件舆论",
            "法治类舆论",
            "文娱类舆论",
            "科教类舆论",
            "国际关系类舆论",
            "健康类舆论",
            "治理类舆论",
            "民生类舆论",
            "生态环境类舆论",
            "其他",
        ]

    def __call__(self, state: TrendState) -> TrendState:
        print("--- [ClassifyNode] 开始十一类舆情分类统计 ---")
        analysis_result = state.get("analysis_result", {})
        top_topics = analysis_result.get("top_topics") or []

        if not top_topics:
            print("⚠️  警告: 没有分析结果可分类，跳过分类统计步骤")
            return {
                "classification_stats": {
                    "topics_by_category": {c: [] for c in self.category_order},
                    "category_display_order": [],
                    "category_heat_map": {c: 0.0 for c in self.category_order},
                }
            }

        # 按类别对事件进行分组
        topics_by_category: Dict[str, List[Dict[str, Any]]] = {c: [] for c in self.category_order}
        for t in top_topics:
            cat = t.get("category") or "其他"
            if cat not in topics_by_category:
                cat = "其他"
            topics_by_category[cat].append(t)

        # 计算每个类别的最大舆情热度
        def _category_heat(cat: str) -> float:
            topics = topics_by_category.get(cat) or []
            max_heat = 0.0
            for tt in topics:
                try:
                    h = float(tt.get("heat_score") or 0)
                except (TypeError, ValueError):
                    h = 0.0
                if h > max_heat:
                    max_heat = h
            return max_heat

        category_heat_map = {cat: _category_heat(cat) for cat in self.category_order}

        # 按类别的"最大舆情热度"从高到低排序，只展示有热点事件的类别
        category_display_order = [c for c in self.category_order if topics_by_category.get(c)]
        category_display_order.sort(key=lambda c: (-category_heat_map[c], self.category_order.index(c)))

        print(f"分类统计完成，共 {len(category_display_order)} 个类别有热点事件")

        return {
            "classification_stats": {
                "topics_by_category": topics_by_category,
                "category_display_order": category_display_order,
                "category_heat_map": category_heat_map,
            }
        }


class ForumNode:
    """论坛讨论节点"""

    def __init__(self):
        api_key = os.environ.get("QUERY_ENGINE_API_KEY")
        if api_key:
            self.llm = ChatOpenAI(
                api_key=api_key,
                base_url=os.environ.get("QUERY_ENGINE_BASE_URL", "https://api.moonshot.cn/v1"),
                model=os.environ.get("QUERY_ENGINE_MODEL_NAME", "moonshot-v1-8k"),
                temperature=0.7,
            )
        else:
            self.llm = None

    def __call__(self, state: TrendState) -> TrendState:
        print("--- [ForumNode] 开始论坛讨论 ---")
        analysis_result = state.get("analysis_result", {})

        topic = "今日热点"
        if "top_topics" in analysis_result and isinstance(analysis_result["top_topics"], list):
            if len(analysis_result["top_topics"]) > 0:
                topic = analysis_result["top_topics"][0].get("topic", "今日热点")
            else:
                return {"forum_discussion": "暂无重大事件剖析（当前无有效热点主题）。"}

        if self.llm is None:
            return {
                "forum_discussion": (
                    f"【重大事件剖析】\n"
                    f"当前焦点：{topic}\n"
                    "由于未启用 QUERY 大模型，本段采用模板化分析："
                    "建议从事件主体、传播平台、情绪极性、关键节点用户与回应节奏五个维度持续跟踪。"
                )
            }

        print(f"讨论话题: {topic}")

        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "你是一名资深舆情分析师，擅长从多个视角对重大舆情事件进行结构化剖析，给出清晰、简明的要点总结，而不是还原聊天记录。",
                ),
                (
                    "human",
                    """
            请围绕“{topic}”生成一段 **重大事件剖析** 的总结性内容，而不是对话脚本。

            要求：
            1. 从三个固定视角分别进行小结，每个视角 2-4 句话：
               - Insight（深度观察）: 事件背后的深层原因、潜在影响、结构性变化。
               - Media（媒体观点）: 主流媒体与社交舆论如何报道和解读，整体情绪倾向。
               - Query（关键事实）: 目前已经确认的关键事实、尚存的不确定点。
            2. 在最后再给出一个“综合判断”，用 3-5 句话，说明该事件的核心风险/机遇、未来可能演变方向，以及需要重点关注的人群或领域。
            3. 直接输出一段可阅读的中文分析文本即可，可以使用清晰的小标题（如“【Insight 深度观察】”），但不要使用 Markdown 代码块或列表标记。
            """,
                ),
            ]
        )

        chain = prompt | self.llm
        try:
            response = chain.invoke({"topic": topic})
            return {"forum_discussion": response.content}
        except Exception as e:
            print(f"ForumNode Error: {e}")
            return {
                "forum_discussion": (
                    f"【重大事件剖析】\n"
                    f"当前焦点：{topic}\n"
                    "模型调用失败，已采用规则化兜底："
                    "建议重点跟踪传播峰值拐点、主流媒体叙事变化、负面情绪聚集议题及权威回应节奏。\n"
                    f"（错误摘要：{str(e)[:160]}）"
                )
            }


def render_langgraph_html_report(
    news_list: List[Dict],
    analysis_result: Dict,
    forum_discussion: str,
    classification_stats: Optional[Dict[str, Any]] = None,
    domain_cluster_stats: Optional[Dict[str, Any]] = None,
) -> str:
    """使用与 main.py 一致的 HTML 模板生成舆情报告，保留来源与链接、支持保存为图片"""
    now = get_beijing_time()
    start_time = now - timedelta(hours=12)
    total_news = len(news_list)
    top_topics = analysis_result.get("top_topics") or []
    summary = analysis_result.get("summary") or "暂无总结"
    domain_section_html = _build_domain_cluster_section(domain_cluster_stats)

    # 使用 ClassifyNode 生成的分类统计结果，如果没有则回退到原有逻辑
    if classification_stats:
        topics_by_category = classification_stats.get("topics_by_category", {})
        category_display_order = classification_stats.get("category_display_order", [])
    else:
        # 回退逻辑：如果没有分类统计结果，则使用原有计算方式（保持向后兼容）
        category_order = [
            "经济类舆论",
            "突发事件舆论",
            "法治类舆论",
            "文娱类舆论",
            "科教类舆论",
            "国际关系类舆论",
            "健康类舆论",
            "治理类舆论",
            "民生类舆论",
            "生态环境类舆论",
            "其他",
        ]
        topics_by_category: Dict[str, List[Dict[str, Any]]] = {c: [] for c in category_order}
        for t in top_topics:
            cat = t.get("category") or "其他"
            if cat not in topics_by_category:
                cat = "其他"
            topics_by_category[cat].append(t)

        def _category_heat(cat: str) -> float:
            topics = topics_by_category.get(cat) or []
            max_heat = 0.0
            for tt in topics:
                try:
                    h = float(tt.get("heat_score") or 0)
                except (TypeError, ValueError):
                    h = 0.0
                if h > max_heat:
                    max_heat = h
            return max_heat

        category_display_order = [c for c in category_order if topics_by_category.get(c)]
        category_display_order.sort(key=lambda c: (-_category_heat(c), category_order.index(c)))

    # 平台权重与排序（用于原始信息表格 & 热度理解）
    # 这里假定配置中的 source_id 使用这些英文标识；若不匹配则自动归为“其他”
    platform_order = [
        "weibo",  # 微博
        "baidu-hot",  # 百度热搜（示例 ID，需与实际配置对应）
        "douyin",  # 抖音
        "zhihu",  # 知乎
        "toutiao",  # 今日头条
        "tieba",  # 贴吧
        "thepaper",  # 澎湃新闻
        "cls",  # 财联社热门
        "ifeng",  # 凤凰网
        "wallstreetcn",  # 华尔街见闻
    ]
    platform_index = {pid: idx for idx, pid in enumerate(platform_order)}

    # 按来源分组新闻（同时保留 source_id，便于排序和权重）
    platforms: Dict[str, Dict[str, Any]] = {}
    for n in news_list:
        source_id = n.get("source_id") or n.get("source") or "other"
        source_name = n.get("source_name") or n.get("source") or source_id
        if source_id not in platforms:
            platforms[source_id] = {"name": source_name, "items": []}
        platforms[source_id]["items"].append(n)

    html = """<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>SONA舆情态势感知</title>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/html2canvas/1.4.1/html2canvas.min.js" integrity="sha512-BNaRQnYJYiPSqHHDb58B0yaPfCu+Wgds8Gp/gU33kqBtgNS4tSPHuGibyoeqMV/TJlSKda6FXzoEyYGjTe+vXA==" crossorigin="anonymous" referrerpolicy="no-referrer"></script>
    <style>
        * { box-sizing: border-box; }
        body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', system-ui, sans-serif; margin: 0; padding: 16px; background: #fafafa; color: #333; line-height: 1.5; }
        .container { max-width: 600px; margin: 0 auto; background: white; border-radius: 12px; overflow: hidden; box-shadow: 0 2px 16px rgba(0,0,0,0.06); }
        .header { background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 100%); color: white; padding: 32px 24px; text-align: center; position: relative; }
        .save-buttons { position: absolute; top: 16px; right: 16px; display: flex; gap: 8px; }
        .save-btn { background: rgba(255, 255, 255, 0.2); border: 1px solid rgba(255, 255, 255, 0.3); color: white; padding: 8px 16px; border-radius: 6px; cursor: pointer; font-size: 13px; font-weight: 500; transition: all 0.2s ease; backdrop-filter: blur(10px); white-space: nowrap; }
        .save-btn:hover { background: rgba(255, 255, 255, 0.3); border-color: rgba(255, 255, 255, 0.5); transform: translateY(-1px); }
        .save-btn:disabled { opacity: 0.6; cursor: not-allowed; }
        .header-title { font-size: 22px; font-weight: 700; margin: 0 0 20px 0; }
        .header-info { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; font-size: 14px; opacity: 0.95; }
        .info-item { text-align: center; }
        .info-label { display: block; font-size: 12px; opacity: 0.8; margin-bottom: 4px; }
        .info-value { font-weight: 600; font-size: 16px; }
        .content { padding: 24px; }
        .section { margin-bottom: 32px; }
        .section:last-child { margin-bottom: 0; }
        .section-title { font-size: 16px; font-weight: 600; color: #1a1a1a; margin: 0 0 16px 0; padding-bottom: 8px; border-bottom: 1px solid #f0f0f0; }
        .section-body { font-size: 14px; color: #374151; line-height: 1.6; white-space: pre-wrap; }
        .topic-category { margin-bottom: 18px; font-size: 14px; font-weight: 600; color: #4b5563; }
        .topic-item { margin-bottom: 16px; padding: 12px; background: #f8fafc; border-radius: 8px; border-left: 4px solid #4f46e5; }
        .topic-name { font-weight: 600; color: #1e293b; margin-bottom: 4px; }
        .event-tag { display: inline-block; background: #4f46e5; color: white; padding: 2px 8px; border-radius: 4px; font-size: 12px; font-weight: 600; margin-right: 6px; }
        .event-tag-new { background: #059669; }
        .event-tag-up { background: #dc2626; }
        .event-tag-down { background: #6b7280; }
        .event-tag-stable { background: #9ca3af; }
        .category-tag { display: inline-block; background: #f0f9ff; color: #0369a1; padding: 2px 8px; border-radius: 4px; font-size: 12px; font-weight: 500; margin-right: 8px; border: 1px solid #bae6fd; }
        .topic-meta { font-size: 12px; color: #64748b; margin-bottom: 6px; }
        .topic-comment { font-size: 13px; color: #475569; }
        .source-group { margin-bottom: 24px; }
        .source-title { color: #666; font-size: 13px; font-weight: 600; margin: 0 0 12px 0; padding-bottom: 6px; border-bottom: 1px solid #f5f5f5; }
        .news-item { margin-bottom: 16px; padding: 12px 0; border-bottom: 1px solid #f5f5f5; display: flex; gap: 12px; align-items: flex-start; }
        .news-item:last-child { border-bottom: none; }
        .news-num { color: #999; font-size: 12px; font-weight: 600; min-width: 20px; text-align: center; flex-shrink: 0; background: #f1f5f9; border-radius: 50%; width: 22px; height: 22px; display: flex; align-items: center; justify-content: center; }
        .news-content { flex: 1; min-width: 0; }
        .news-link { color: #2563eb; text-decoration: none; }
        .news-link:hover { text-decoration: underline; }
        .news-link:visited { color: #7c3aed; }
        .news-source { color: #666; font-size: 12px; margin-bottom: 4px; }
        .news-hot { font-size: 11px; color: #dc2626; font-weight: 500; }
        .raw-table-wrapper { overflow-x: auto; }
        .raw-table { width: 100%; border-collapse: collapse; font-size: 13px; }
        .raw-table thead { background: #f1f5f9; }
        .raw-table th, .raw-table td { padding: 8px 10px; border-bottom: 1px solid #e5e7eb; text-align: left; }
        .raw-table th { color: #4b5563; font-weight: 600; font-size: 12px; }
        .raw-table tbody tr:hover { background: #f9fafb; }
        .raw-table .platform-cell { white-space: nowrap; color: #374151; font-weight: 500; }
        .raw-table .rank-cell { width: 60px; color: #6b7280; }
        .raw-table .trend-cell { width: 50px; text-align: center; }
        .raw-table .title-cell { color: #111827; }
        .trend-badge { font-size: 14px; font-weight: bold; }
        .trend-new { color: #059669; }
        .trend-up { color: #dc2626; }
        .trend-down { color: #6b7280; }
        .trend-stable { color: #9ca3af; }
        .raw-table .hot-badge { margin-left: 6px; font-size: 11px; color: #b91c1c; }
        .footer { margin-top: 24px; padding: 20px 24px; background: #f8f9fa; border-top: 1px solid #e5e7eb; text-align: center; }
        .footer-content { font-size: 13px; color: #6b7280; line-height: 1.6; }
        .project-name { font-weight: 600; color: #374151; }
        .domain-cluster-intro { white-space: normal; font-size: 13px; color: #64748b; margin-bottom: 16px; line-height: 1.5; }
        .domain-cluster-count { font-weight: 500; color: #94a3b8; }
        .cluster-card { margin-bottom: 14px; padding: 12px; background: #faf5ff; border-radius: 8px; border-left: 4px solid #7c3aed; }
        .cluster-rep { font-size: 14px; font-weight: 600; color: #1e293b; margin-bottom: 6px; line-height: 1.45; }
        .domain-badge { display: inline-block; background: #ede9fe; color: #5b21b6; padding: 2px 8px; border-radius: 4px; font-size: 11px; font-weight: 600; margin-right: 8px; vertical-align: middle; }
        .cluster-meta { font-size: 12px; color: #64748b; margin-bottom: 6px; }
        .cluster-samples { font-size: 12px; color: #475569; }
        .cluster-sample { margin-top: 4px; padding-left: 2px; }
        .forum-modules { display: flex; flex-direction: column; gap: 14px; }
        .forum-mod { border-radius: 12px; border: 1px solid #e5e7eb; overflow: hidden; background: #fff; box-shadow: 0 1px 4px rgba(15, 23, 42, 0.06); }
        .forum-mod-head { padding: 12px 16px; border-bottom: 1px solid rgba(0,0,0,0.06); display: flex; align-items: center; gap: 10px; }
        .forum-mod-kicker { font-size: 13px; font-weight: 700; letter-spacing: 0.02em; color: #1e293b; }
        .forum-mod-content { padding: 14px 16px 16px; font-size: 14px; line-height: 1.7; color: #374151; }
        .forum-mod-para { margin: 0 0 12px 0; }
        .forum-mod-para:last-child { margin-bottom: 0; }
        .forum-mod-insight .forum-mod-head { background: linear-gradient(90deg, #eef2ff 0%, #e0e7ff 100%); border-bottom-color: #c7d2fe; }
        .forum-mod-insight .forum-mod-kicker::before { content: "◇"; margin-right: 8px; color: #4f46e5; font-size: 12px; }
        .forum-mod-media .forum-mod-head { background: linear-gradient(90deg, #ecfdf5 0%, #d1fae5 100%); border-bottom-color: #a7f3d0; }
        .forum-mod-media .forum-mod-kicker::before { content: "◎"; margin-right: 8px; color: #059669; font-size: 12px; }
        .forum-mod-query .forum-mod-head { background: linear-gradient(90deg, #fffbeb 0%, #fef3c7 100%); border-bottom-color: #fde68a; }
        .forum-mod-query .forum-mod-kicker::before { content: "✦"; margin-right: 8px; color: #d97706; font-size: 11px; }
        .forum-mod-judgment .forum-mod-head { background: linear-gradient(90deg, #faf5ff 0%, #f3e8ff 100%); border-bottom-color: #e9d5ff; }
        .forum-mod-judgment .forum-mod-kicker::before { content: "◆"; margin-right: 8px; color: #7c3aed; font-size: 12px; }
        .forum-mod-preamble { border-style: dashed; background: #f8fafc; }
        .forum-mod-preamble .forum-mod-head { background: #f1f5f9; border-bottom-color: #e2e8f0; }
        .forum-legacy { font-size: 14px; line-height: 1.65; color: #374151; white-space: normal; }
        .forum-empty { color: #64748b; font-size: 14px; margin: 0; }
        .forum-section-body { white-space: normal; }
        @media (max-width: 480px) { body { padding: 12px; } .header { padding: 24px 20px; } .content { padding: 20px; } .header-info { grid-template-columns: 1fr; } .save-buttons { position: static; margin-bottom: 16px; justify-content: center; flex-direction: column; width: 100%; } .save-btn { width: 100%; } }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <div class="save-buttons">
                <button class="save-btn" onclick="saveAsImage()" title="保存为图片">📷</button>
                <button class="save-btn" onclick="saveAsMultipleImages()" title="分段保存">📑</button>
            </div>
            <div class="header-title">SONA 舆情态势感知</div>
            <div class="header-info">
                <div class="info-item"><span class="info-label">报告类型</span><span class="info-value">全网热点</span></div>
                <div class="info-item"><span class="info-label">新闻总数</span><span class="info-value">"""
    html += str(total_news)
    html += """ 条</span></div>
                <div class="info-item"><span class="info-label">热点话题</span><span class="info-value">"""
    html += str(len(top_topics))
    html += """ 个</span></div>
                <div class="info-item"><span class="info-label">时间范围</span><span class="info-value">"""
    html += start_time.strftime("%m-%d %H:%M") + " ~ " + now.strftime("%m-%d %H:%M")
    html += """</span></div>
            </div>
        </div>
        <div class="content">
            <div class="section">
                <div class="section-title">今日热点概览</div>
                <div class="section-body">"""
    html += html_escape(summary)
    html += """</div>
            </div>
"""
    html += domain_section_html
    html += """            <div class="section">
                <div class="section-title">热点事件</div>
"""

    # 趋势标签映射（中文描述）
    TREND_LABELS = {
        "new": "新上榜",
        "up": "急剧上升",
        "down": "明显下滑",
        "stable": "保持平稳",
    }
    
    # 为每个热点事件分配序号，并根据热度排序
    # 先按 heat_score 降序排序
    sorted_topics = sorted(top_topics, key=lambda t: (t.get("heat_score") or 0), reverse=True)
    
    # 同义词组 - 用于扩展匹配
    # 注意：这里的组内词会相互扩展，即匹配到任一词等于匹配整个组
    SYNONYM_GROUPS = [
        ['美伊', '美国伊朗', '伊美', '伊朗', '美伊关系'],  # 美伊相关词汇
        ['伊朗', '波斯', '德黑兰'],
        ['美国', '美利坚', '华盛顿', '特朗普', '拜登'],
        ['俄罗斯', '俄国', '普京'],
        ['乌克兰', '基辅'],
        ['以色列', '特拉维夫', '内塔尼亚胡'],
        ['中东', '西亚'],
        ['冲突', '战争', '敌对', '对抗'],
        ['紧张', '局势紧张'],
        ['升级', '恶化', '加剧'],
        ['协议', '共识', '谈判', '对话'],
        ['推迟', '延迟', '暂缓'],
        ['袭击', '打击', '攻击'],
        ['表态', '发言', '讲话', '声明'],
        ['心理战', '心理'],
    ]
    
    # 提取每个事件的关键词（用于后续匹配原始信息）
    # 使用更智能的关键词提取：提取具有区分性的关键词 + 同义词扩展
    def extract_keywords(topic_title: str, max_keywords: int = 10) -> List[str]:
        """提取标题关键词用于匹配
        
        改进：
        1. 提取更多关键词（从3个增加到10个）
        2. 优先选择有区分性的词（排除常见词汇）
        3. 同义词扩展：匹配同义词组中的任一词等于匹配整个组
        """
        if not topic_title:
            return []
        
        # 常见词（排除这些词以提高匹配精度）
        common_words = {"的", "了", "是", "在", "和", "与", "或", "被", "有", "我", "你", "他", "她", "它", "们",
                        "这", "那", "都", "也", "就", "而", "但", "并", "对", "为", "以", "及", "从", "到",
                        "一个", "我们", "你们", "他们", "她们", "什么", "怎么", "如何", "为什么",
                        "今天", "昨天", "明天", "现在", "目前", "今年", "去年", "明年"}
        
        import re
        # 提取所有2-4字的词组
        candidates = set()
        for length in [2, 3, 4]:
            for i in range(len(topic_title) - length + 1):
                chunk = topic_title[i:i+length]
                if chunk not in common_words and not chunk.isdigit():
                    candidates.add(chunk)
        
        # 优先词（高频实体词汇）
        priority_words = [
            '美伊', '中美', '中俄', '伊以', '俄乌',
            '伊朗', '以色列', '俄罗斯', '乌克兰', '美国', '中国', '欧洲', '中东', '亚洲',
            '特朗普', '拜登', '普京', '内塔尼亚胡',
            '协议', '谈判', '对话', '峰会', '制裁', '打击', '袭击', '冲突', '战争',
            '紧张', '升级', '恶化', '缓和', '停火',
            '局势', '经济', '政治', '军事', '外交',
        ]
        
        # 分离优先词和其他词
        priority = []
        others = []
        for c in candidates:
            is_priority = False
            for p in priority_words:
                if p in c or c in p:
                    is_priority = True
                    break
            if is_priority:
                priority.append(c)
            else:
                others.append(c)
        
        # 优先词优先保留更长的
        priority.sort(key=lambda x: -len(x))
        others.sort(key=lambda x: -len(x))
        
        # 合并：优先词取6个，其他取4个
        keywords = priority[:6] + others[:4]
        
        return keywords[:max_keywords]
    
    def expand_keywords(keywords: List[str]) -> List[str]:
        """展开关键词以包含同义词组中的所有词"""
        expanded = set(keywords)
        for kw in keywords:
            for group in SYNONYM_GROUPS:
                for g in group:
                    if kw in g or g in kw:
                        expanded.update(group)
                        break
        return list(expanded)
    
    # 为每个热点事件分配序号和关键词，并收集趋势和天数信息
    event_keywords_map = {}  # event_idx -> keywords (expanded)
    event_info_map = {}  # event_idx -> {trend, days_on_list, category}
    for idx, t in enumerate(sorted_topics, 1):
        t['_event_index'] = idx
        keywords = extract_keywords(t.get("topic", ""))
        # 扩展关键词以包含同义词
        expanded_keywords = expand_keywords(keywords)
        event_keywords_map[idx] = expanded_keywords
        # 保存事件元信息（趋势、天数、类别）
        event_info_map[idx] = {
            "trend": "new",
            "days_on_list": None,
            "category": t.get("category", "其他")
        }

    # 展示热点事件（按热度排序，每个独立一行）
    for t in sorted_topics:
        topic = t.get("topic") or ""
        sentiment = t.get("sentiment") or ""
        comment = t.get("comment") or ""
        heat_score = t.get("heat_score")
        event_idx = t.get("_event_index", 0)
        
        # 获取事件的趋势、天数和类别信息
        event_info = event_info_map.get(event_idx, {})
        event_trend = event_info.get("trend", "new")
        days_on_list = event_info.get("days_on_list")
        category = event_info.get("category", t.get("category", "其他"))
        
        html += f'<div class="topic-item"><div class="topic-name">'
        # 添加序号标签
        html += f'<span class="event-tag">#{event_idx}</span>'
        html += html_escape(topic)
        html += "</div>"
        html += '<div class="topic-meta">'
        # 所属类别
        if category:
            html += f'{html_escape(category)} '
        # 情感和热度
        if sentiment:
            html += f'情感: {html_escape(sentiment)}'
        if heat_score is not None:
            if sentiment:
                html += ' · '
            html += f'舆情热度: {heat_score}'
        # 在榜天数
        if days_on_list is not None:
            html += f' · 在榜{days_on_list}天'
        # 趋势标签
        trend_label = TREND_LABELS.get(event_trend, "保持平稳")
        html += f' · <span class="event-tag event-tag-{event_trend}">{trend_label}</span>'
        html += "</div>"
        html += f'<div class="topic-comment">{html_escape(comment)}</div></div>'
    
    # 更新原始信息列表中的热点事件关联
    # 为每个原始信息匹配热点事件序号
    # 趋势图标映射
    TREND_ICONS = {
        "new": '<span class="trend-badge trend-new" title="新上榜">🆕</span>',
        "up": '<span class="trend-badge trend-up" title="排名上升">↑</span>',
        "down": '<span class="trend-badge trend-down" title="排名下降">↓</span>',
        "stable": '<span class="trend-badge trend-stable" title="排名稳定">→</span>',
    }

    # 获取第一个重大事件作为剖析标题后缀
    major_event_title = ""
    if top_topics and len(top_topics) > 0:
        major_event_title = top_topics[0].get("topic", "")
    event_title_suffix = f": {major_event_title}" if major_event_title else ""

    html += """
            </div>
            <div class="section">
                <div class="section-title">重大事件剖析"""
    html += html_escape(event_title_suffix)
    html += """</div>
                <div class="section-body forum-section-body">"""
    html += format_forum_discussion_html(forum_discussion or "暂无重大事件剖析")
    html += """</div>
            </div>
            <div class="section">
                <div class="section-title">原始信息抓取（来源与链接）</div>
                <div style="margin-bottom: 12px;">
                    <input type="text" id="topicSearchBox" placeholder="🔍 搜索热门话题..." 
                           onkeyup="filterTopics()"
                           style="width: 100%; padding: 10px 14px; border: 1px solid #e5e7eb; border-radius: 8px; font-size: 14px; outline: none; box-sizing: border-box;">
                </div>
                <div class="raw-table-wrapper">
                    <table class="raw-table" id="topicsTable">
                        <thead>
                            <tr>
                                <th>平台</th>
                                <th>排行</th>
                                <th>趋势</th>
                                <th>信息 & 链接</th>
                            </tr>
                        </thead>
                        <tbody>
"""

    # 将所有新闻按平台顺序和平台内排行排序，生成统一列表
    # 平台顺序：微博、百度热搜、抖音、知乎、今日头条、贴吧、澎湃新闻、财联社热门、凤凰网、华尔街见闻，其它平台排在最后
    def _platform_sort_key(item: Dict[str, Any]) -> int:
        sid = item.get("source_id") or item.get("source") or "other"
        return platform_index.get(sid, len(platform_index))

    rows: List[Dict[str, Any]] = []
    for sid, info in platforms.items():
        for n in info["items"]:
            n_copy = dict(n)
            # 确保包含 ID 与展示名
            n_copy["source_id"] = sid
            n_copy["source_name"] = info["name"]
            rows.append(n_copy)

    rows.sort(key=lambda n: (_platform_sort_key(n), n.get("rank") or 9999))

    # 为每个原始信息匹配热点事件序号，并收集最佳的趋势和天数信息
    for n in rows:
        title = n.get("title") or ""
        matched_event_idx = 0
        for event_idx, keywords in event_keywords_map.items():
            for kw in keywords:
                if kw in title:
                    matched_event_idx = event_idx
                    break
            if matched_event_idx:
                break
        n['_matched_event_idx'] = matched_event_idx
        
        # 如果匹配到了事件，更新事件的趋势信息（取最优趋势）
        if matched_event_idx and n.get("trend"):
            current_trend = n.get("trend", "stable")
            current_days = n.get("days_on_list")
            event_info = event_info_map[matched_event_idx]
            
            # 更新趋势：new > up > stable > down（优先级）
            trend_priority = {"new": 0, "up": 1, "stable": 2, "down": 3}
            if trend_priority.get(current_trend, 4) < trend_priority.get(event_info["trend"], 4):
                event_info["trend"] = current_trend
            
            # 更新在榜天数（取最大值）
            if current_days and (not event_info["days_on_list"] or current_days > event_info["days_on_list"]):
                event_info["days_on_list"] = current_days

    # 生成原始信息表格（匹配已完成）
    for n in rows:
        platform_name = n.get("source_name") or n.get("source") or "其他"
        rank = n.get("rank")
        rank_text = str(rank) if rank is not None else "-"
        title = n.get("title") or ""
        original_link_url = n.get("mobile_url") or n.get("url") or ""
        hot_val = n.get("hot_value")
        # 趋势
        trend = n.get("trend", "stable")
        trend_icon = TREND_ICONS.get(trend, TREND_ICONS["stable"])
        # 热点事件序号（如果有匹配的话）
        matched_event_idx = n.get('_matched_event_idx', 0)
        event_tag_html = f'<span class="event-tag" style="margin-right: 6px;">#{matched_event_idx}</span>' if matched_event_idx else ''
        
        # 微博智搜链接：点击后打开微博搜索页面
        weibo_search_url = f"https://s.weibo.com/aisearch?q={quote(title)}&Refer=weibo_aisearch"

        html += "<tr>"
        html += '<td class="platform-cell">' + html_escape(platform_name) + "</td>"
        html += '<td class="rank-cell">' + html_escape(rank_text) + "</td>"
        html += '<td class="trend-cell">' + trend_icon + "</td>"
        html += '<td class="title-cell">'
        # 添加热点事件序号标签
        html += event_tag_html
        # 使用微博智搜链接替代原始链接
        if title:
            html += '<a href="' + html_escape(weibo_search_url) + '" target="_blank" class="news-link" title="点击查看微博智搜结果">' + html_escape(title) + "</a>"
        else:
            html += html_escape(title)
        if hot_val is not None and hot_val != 0:
            html += '<span class="hot-badge">热度 ' + html_escape(str(hot_val)) + "</span>"
        html += "</td></tr>"

    html += """
                        </tbody>
                    </table>
                </div>
            </div>
            </div>
        </div>
        <div class="footer">
            <div class="footer-content">
                由 <span class="project-name">BJTU舆情实验室</span> 生成
            </div>
        </div>
    </div>
    <script>
        async function saveAsImage() {
            const button = event.target;
            const originalText = button.textContent;
            try {
                button.textContent = '生成中...';
                button.disabled = true;
                window.scrollTo(0, 0);
                await new Promise(r => setTimeout(r, 200));
                const buttons = document.querySelector('.save-buttons');
                buttons.style.visibility = 'hidden';
                await new Promise(r => setTimeout(r, 100));
                const container = document.querySelector('.container');
                const canvas = await html2canvas(container, { backgroundColor: '#ffffff', scale: 1.5, useCORS: true, allowTaint: false, imageTimeout: 10000, logging: false });
                buttons.style.visibility = 'visible';
                const link = document.createElement('a');
                const now = new Date();
                link.download = 'BJTUPubClaw_舆情日报_' + now.getFullYear() + String(now.getMonth()+1).padStart(2,'0') + String(now.getDate()).padStart(2,'0') + '_' + String(now.getHours()).padStart(2,'0') + String(now.getMinutes()).padStart(2,'0') + '.png';
                link.href = canvas.toDataURL('image/png', 1.0);
                document.body.appendChild(link);
                link.click();
                document.body.removeChild(link);
                button.textContent = '保存成功!';
                setTimeout(function(){ button.textContent = originalText; button.disabled = false; }, 2000);
            } catch (e) {
                const buttons = document.querySelector('.save-buttons');
                if (buttons) buttons.style.visibility = 'visible';
                button.textContent = '保存失败';
                setTimeout(function(){ button.textContent = originalText; button.disabled = false; }, 2000);
            }
        }
        async function saveAsMultipleImages() {
            const button = event.target;
            const originalText = button.textContent;
            const container = document.querySelector('.container');
            const scale = 1.5;
            const maxHeight = 5000 / scale;
            try {
                button.textContent = '分析中...';
                button.disabled = true;
                const sections = container.querySelectorAll('.section');
                const header = container.querySelector('.header');
                const footer = container.querySelector('.footer');
                const buttons = document.querySelector('.save-buttons');
                buttons.style.visibility = 'hidden';
                const images = [];
                let seg = 0;
                for (let i = 0; i < sections.length; i++) {
                    button.textContent = '生成中 (' + (i+1) + '/' + sections.length + ')...';
                    const temp = document.createElement('div');
                    temp.className = 'container';
                    temp.style.cssText = 'position:absolute;left:-9999px;top:0;width:' + container.offsetWidth + 'px;background:white;';
                    temp.appendChild(header.cloneNode(true));
                    temp.appendChild(sections[i].cloneNode(true));
                    temp.appendChild(footer.cloneNode(true));
                    document.body.appendChild(temp);
                    await new Promise(r => setTimeout(r, 100));
                    const canvas = await html2canvas(temp, { backgroundColor: '#ffffff', scale: scale, useCORS: true, logging: false });
                    document.body.removeChild(temp);
                    images.push(canvas.toDataURL('image/png', 1.0));
                }
                buttons.style.visibility = 'visible';
                const now = new Date();
                const base = 'BJTUPubClaw_舆情日报_' + now.getFullYear() + String(now.getMonth()+1).padStart(2,'0') + String(now.getDate()).padStart(2,'0') + '_' + String(now.getHours()).padStart(2,'0') + String(now.getMinutes()).padStart(2,'0');
                for (let i = 0; i < images.length; i++) {
                    const a = document.createElement('a');
                    a.download = base + '_part' + (i+1) + '.png';
                    a.href = images[i];
                    document.body.appendChild(a);
                    a.click();
                    document.body.removeChild(a);
                    await new Promise(r => setTimeout(r, 100));
                }
                button.textContent = '已保存 ' + images.length + ' 张图片!';
                setTimeout(function(){ button.textContent = originalText; button.disabled = false; }, 2000);
            } catch (e) {
                const buttons = document.querySelector('.save-buttons');
                if (buttons) buttons.style.visibility = 'visible';
                button.textContent = '保存失败';
                setTimeout(function(){ button.textContent = originalText; button.disabled = false; }, 2000);
            }
        }
        document.addEventListener('DOMContentLoaded', function(){ window.scrollTo(0, 0); });

        // 搜索热门话题功能
        function filterTopics() {
            const input = document.getElementById('topicSearchBox');
            const filter = input.value.toLowerCase();
            const table = document.getElementById('topicsTable');
            const rows = table.getElementsByTagName('tr');
            let visibleCount = 0;
            for (let i = 1; i < rows.length; i++) { // 跳过表头
                const row = rows[i];
                const titleCell = row.querySelector('.title-cell');
                if (titleCell) {
                    const titleText = titleCell.textContent || titleCell.innerText;
                    if (titleText.toLowerCase().indexOf(filter) > -1) {
                        row.style.display = '';
                        visibleCount++;
                    } else {
                        row.style.display = 'none';
                    }
                }
            }
            // 显示搜索结果数量
            const searchInfo = document.getElementById('searchInfo');
            if (filter) {
                if (!searchInfo) {
                    const infoDiv = document.createElement('div');
                    infoDiv.id = 'searchInfo';
                    infoDiv.style.cssText = 'padding: 8px 12px; background: #f0f9ff; border-radius: 6px; margin-bottom: 12px; font-size: 13px; color: #0369a1;';
                    input.parentNode.insertBefore(infoDiv, input.nextSibling);
                }
                document.getElementById('searchInfo').textContent = `找到 ${visibleCount} 条相关话题`;
            } else if (searchInfo) {
                searchInfo.remove();
            }
        }
    </script>
</body>
</html>"""
    return html


class ReportNode:
    """报告生成节点：使用与 main.py 一致的 HTML 模板，保留来源与链接，支持保存为图片与浏览器打开"""

    def __init__(self):
        pass

    def __call__(self, state: TrendState) -> TrendState:
        print("--- [ReportNode] 开始生成报告 ---")
        analysis_result = state.get("analysis_result", {})
        news_data = state.get("news_data", {})
        discussion = state.get("forum_discussion", "")
        classification_stats = state.get("classification_stats")
        domain_cluster_stats = state.get("domain_cluster_stats")

        news_list = news_data.get("news_list") or []

        # 修复：即使有错误，只要有新闻数据就生成报告
        if not news_list:
            print("⚠️  警告: 没有新闻数据，生成空报告")
            empty_html = """<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>舆情日报 - 无数据</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
</head>
<body>
    <div class="container mt-5">
        <h1 class="text-center">舆情日报</h1>
        <div class="alert alert-warning mt-4">
            <h4>⚠️ 今日无新闻数据</h4>
            <p>可能的原因：</p>
            <ul>
                <li>API 服务暂时不可用</li>
                <li>网络连接问题</li>
                <li>所有平台均未返回数据</li>
            </ul>
            <p class="mb-0">请检查网络连接或稍后重试。</p>
        </div>
    </div>
</body>
</html>"""
            return {"html_report": empty_html}

        html_content = render_langgraph_html_report(
            news_list,
            analysis_result,
            discussion,
            classification_stats,
            domain_cluster_stats,
        )
        return {"html_report": html_content}


# === 构建图 ===
def build_graph():
    workflow = StateGraph(TrendState)

    # 获取配置中的平台列表；若缺失或为空（例如无 config/config.yaml 时返回 platforms: []），使用内置默认
    # 新增 8 个扩展平台（雪球、36kr、虎扑、V2ex、少数派、快手、IT之家、抽屉新热榜）
    _default_platforms = [
        {"id": "weibo", "name": "微博", "enabled": True},
        {"id": "zhihu", "name": "知乎", "enabled": True},
        {"id": "toutiao", "name": "今日头条", "enabled": True},
        {"id": "baidu", "name": "百度热搜", "enabled": True},
        {"id": "douyin", "name": "抖音", "enabled": True},
        {"id": "bilibili-hot-search", "name": "B站热搜", "enabled": True},
        {"id": "thepaper", "name": "澎湃新闻", "enabled": True},
        {"id": "tieba", "name": "贴吧", "enabled": True},
        {"id": "ifeng", "name": "凤凰网", "enabled": True},
        {"id": "cls-hot", "name": "财联社热门", "enabled": True},
        {"id": "wallstreetcn-hot", "name": "华尔街见闻", "enabled": True},
        # 扩展平台（默认禁用）
        {"id": "xueqiu", "name": "雪球", "enabled": False},
        {"id": "36kr", "name": "36氪", "enabled": False},
        {"id": "hupu", "name": "虎扑", "enabled": False},
        {"id": "v2ex", "name": "V2ex", "enabled": False},
        {"id": "sspai", "name": "少数派", "enabled": False},
        {"id": "kuaishou", "name": "快手", "enabled": False},
        {"id": "ithome", "name": "IT之家", "enabled": False},
        {"id": "chongbuluo", "name": "抽屉新热榜", "enabled": False},
    ]
    platforms: List[Dict[str, Any]] = []
    if CONFIG and "platforms" in CONFIG:
        # 过滤掉 enabled: false 的平台
        platforms = [p for p in (CONFIG["platforms"] or []) if p.get("enabled", True)]
    if not platforms:
        # 如果配置为空，使用默认列表但过滤掉禁用的
        platforms = [p for p in _default_platforms if p.get("enabled", True)]

    # 创建Fetch节点映射（核心平台 + 扩展平台）
    fetch_node_map = {
        "weibo": FetchWeiboNode,
        "zhihu": FetchZhihuNode,
        "toutiao": FetchToutiaoNode,
        "baidu": FetchBaiduNode,
        "douyin": FetchDouyinNode,
        "bilibili-hot-search": FetchBilibiliNode,
        "thepaper": FetchThepaperNode,
        "tieba": FetchTiebaNode,
        "ifeng": FetchIfengNode,
        "cls-hot": FetchClsNode,
        "wallstreetcn-hot": FetchWallstreetcnNode,
        # 扩展平台
        "xueqiu": FetchXueqiuNode,
        "36kr": Fetch36krNode,
        "hupu": FetchHupuNode,
        "v2ex": FetchV2exNode,
        "sspai": FetchSspaiNode,
        "kuaishou": FetchKuaishouNode,
        "ithome": FetchIthomeNode,
        "chongbuluo": FetchChongbuluoNode,
    }

    # 添加入口节点
    workflow.add_node("start_fetch", StartFetchNode())

    # 添加Fetch节点（并行抓取）
    fetch_node_names = []
    for platform in platforms:
        platform_id = platform["id"]
        if platform_id in fetch_node_map:
            node_name = f"fetch_{platform_id}"
            fetch_node_names.append(node_name)
            workflow.add_node(node_name, fetch_node_map[platform_id]())

    # 添加其他节点
    workflow.add_node("merge_fetch", MergeFetchNode())
    workflow.add_node("normalize", NormalizeNewsNode())
    workflow.add_node("cluster_domain", ClusterDomainNode())
    workflow.add_node("insight", InsightNode())
    workflow.add_node("classify", ClassifyNode())
    workflow.add_node("forum", ForumNode())
    workflow.add_node("report", ReportNode())

    # 定义边：所有Fetch节点从start_fetch开始，然后合并
    if fetch_node_names:
        # 设置入口点
        workflow.set_entry_point("start_fetch")

        # 所有Fetch节点都从start_fetch开始（这样可以并行执行）
        for node_name in fetch_node_names:
            workflow.add_edge("start_fetch", node_name)

        # 所有Fetch节点都连接到merge_fetch
        for node_name in fetch_node_names:
            workflow.add_edge(node_name, "merge_fetch")

        # 合并后进入normalize
        workflow.add_edge("merge_fetch", "normalize")
    else:
        # 如果没有配置平台，使用旧的SpiderNode作为后备
        workflow.add_node("spider", SpiderNode())
        workflow.set_entry_point("spider")
        workflow.add_edge("spider", "normalize")

    # 后续流程
    workflow.add_edge("normalize", "cluster_domain")
    workflow.add_edge("cluster_domain", "insight")
    workflow.add_edge("insight", "classify")
    workflow.add_edge("classify", "forum")
    workflow.add_edge("forum", "report")
    workflow.add_edge("report", END)

    return workflow.compile()


def _slim_news_items_for_snapshot(news_list: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """快照中仅保留结构化字段，避免冗余。"""
    out: List[Dict[str, Any]] = []
    for n in news_list:
        out.append(
            {
                "title": n.get("title"),
                "url": n.get("url"),
                "source_id": n.get("source_id"),
                "source_name": n.get("source_name") or n.get("source"),
                "rank": n.get("rank"),
                "hot_value": n.get("hot_value"),
                "trend": n.get("trend"),
                "days_on_list": n.get("days_on_list"),
            }
        )
    return out


def write_hot_run_snapshot_json(final_state: Dict[str, Any], report_html_path: Optional[str]) -> str:
    """将本次热点流程状态写入 ``docs/case candidate/*.json``，返回绝对路径。"""
    global LAST_HOT_SNAPSHOT_JSON
    from utils.path import get_project_root

    root = get_project_root()
    out_dir = root / "docs" / "case candidate"
    ensure_directory_exists(str(out_dir))
    filename = f"hot_snapshot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    output_path = out_dir / filename
    news_data = final_state.get("news_data") or {}
    news_list = news_data.get("news_list") or []
    payload: Dict[str, Any] = {
        "generated_at": get_beijing_time().isoformat(),
        "report_html_path": report_html_path,
        "domain_cluster_stats": final_state.get("domain_cluster_stats"),
        "classification_stats": final_state.get("classification_stats"),
        "analysis_result": final_state.get("analysis_result"),
        "forum_discussion": final_state.get("forum_discussion"),
        "news_list": _slim_news_items_for_snapshot(news_list),
    }
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    resolved = str(output_path.resolve())
    LAST_HOT_SNAPSHOT_JSON = resolved
    return resolved


def run(config_path: Optional[str] = None) -> str:
    """bjtupubclaw 入口：复用 trend_radar_langgraph.py 的完整效果，只做必要的配置路径与品牌名适配。"""
    global CONFIG
    # 与 Sona 主程序共用 .env，并映射 KIMI_APIKEY / OPENAI_APIKEY / DEEPSEEK_APIKEY 等
    try:
        from utils.hot_topics_env import ensure_hot_topics_cwd, prepare_hot_topics_environment

        prepare_hot_topics_environment()
        ensure_hot_topics_cwd()
    except Exception:
        pass

    if config_path:
        os.environ["CONFIG_PATH"] = config_path
    CONFIG = load_config()

    print("=== BJTUPubClaw LangGraph 版启动 ===")

    app = build_graph()

    # 初始状态
    initial_state: TrendState = {
        "messages": [],
        "news_data": {},
        "raw_items": [],
        "analysis_result": {},
        "classification_stats": {},
        "domain_cluster_stats": {},
        "forum_discussion": "",
        "html_report": "",
        "error": None,
    }

    final_state: TrendState
    try:
        final_state = app.invoke(initial_state)
    except Exception as exc:
        raise

    report_path = ""
    if final_state.get("html_report"):
        output_dir = Path("output_langgraph")
        ensure_directory_exists(str(output_dir))
        filename = f"bjtupubclaw_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
        output_path = output_dir / filename

        with open(output_path, "w", encoding="utf-8") as f:
            f.write(final_state["html_report"])

        # 同时写入 index.html，便于直接打开目录时查看最新报告
        index_path = output_dir / "index.html"
        with open(index_path, "w", encoding="utf-8") as f:
            f.write(final_state["html_report"])

        report_path = str(output_path.resolve())
        print(f"\n✅ 流程执行成功！报告已保存: {report_path}")

        # 非 Docker 且交互终端下自动打开浏览器，避免脚本/后台任务噪声
        if not _is_docker_env() and sys.stdout.isatty():
            file_url = "file://" + report_path
            print(f"正在打开报告: {file_url}")
            try:
                webbrowser.open(file_url)
            except Exception as e:
                print(f"自动打开浏览器失败: {e}，请手动打开: {report_path}")
        else:
            print(f"（Docker 环境不自动打开浏览器）报告路径: {report_path}")
    else:
        print("\n❌ 流程执行完成，但未生成报告内容。")
        if final_state.get("error"):
            print(f"错误信息: {final_state['error']}")

    try:
        snap_path = write_hot_run_snapshot_json(final_state, report_path or None)
        print(f"📄 热点快照 JSON: {snap_path}")
    except Exception as e:
        print(f"⚠️  写入热点快照 JSON 失败: {e}")

    return report_path


# === 主程序入口 ===
def main():
    try:
        run()
    except Exception as e:
        print(f"\n❌ 流程执行异常: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
