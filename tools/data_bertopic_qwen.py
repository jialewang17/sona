#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
BERTopic + Qwen 主题聚类工具。

- 舆情流水线：通过 ``analysis_topic_bertopic`` 读取 data_collect 产出的 CSV，
  识别内容列并清洗后做 BERTopic + 大模型主题合并，结果写入任务过程目录。
- 独立 CLI：``python tools/data_bertopic_qwen.py --input-file ...`` 仍可用于离线实验。

依赖（可选安装）: bertopic, umap-learn, hdbscan；停用词见 config/stopwords.txt。
"""
from __future__ import annotations

import re
import json
import argparse
import warnings
import hashlib
import pickle
import os
import sys
import locale
import random
import inspect
import logging
import shutil
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Any

# 严格抑制所有警告（在导入可能产生警告的库之前）
warnings.filterwarnings("ignore")  # 抑制所有警告
warnings.simplefilter("ignore")  # 设置默认过滤器为忽略
# 特别抑制 pkg_resources 相关警告
warnings.filterwarnings("ignore", message=".*pkg_resources.*")
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

import pandas as pd
import numpy as np
import jieba
import yaml
import asyncio
from openai import OpenAI, AsyncOpenAI
from sklearn.feature_extraction.text import CountVectorizer
from collections import defaultdict

BERTopic = None
UMAP = None
HDBSCAN = None
_IMPORT_ERROR = None
try:
    from bertopic import BERTopic
    from umap import UMAP
    from hdbscan import HDBSCAN
except Exception as _e:
    _IMPORT_ERROR = _e


def _patch_hdbscan_sklearn_compat(logger=None) -> None:
    """
    兼容 sklearn 新版 API 变更：
    - hdbscan 旧版本调用 check_array(..., force_all_finite=...)
    - sklearn 新版本参数改名为 ensure_all_finite
    这里做运行时补丁，避免环境轻微不匹配导致主题建模崩溃。
    """
    try:
        import sklearn.utils.validation as sk_validation
        import hdbscan.hdbscan_ as hdbscan_mod
    except Exception:
        return

    try:
        sig = inspect.signature(sk_validation.check_array)
        if "force_all_finite" in sig.parameters:
            return
    except Exception:
        return

    # 已补丁过则跳过
    if getattr(hdbscan_mod.check_array, "__name__", "") == "_compat_check_array":
        return

    def _compat_check_array(*args, **kwargs):
        if "force_all_finite" in kwargs and "ensure_all_finite" not in kwargs:
            kwargs["ensure_all_finite"] = kwargs.pop("force_all_finite")
        return sk_validation.check_array(*args, **kwargs)

    hdbscan_mod.check_array = _compat_check_array
    if logger:
        log_success(logger, "已应用 hdbscan/sklearn 参数兼容补丁", "TopicBertopic")

# 抑制jieba的日志输出（设置为ERROR级别，减少输出）
jieba.setLogLevel(60)  # 60 = ERROR级别，可以抑制"Building prefix dict..."等信息

from langchain_core.tools import tool

from tools._csv_io import read_csv_rows_all
from tools.keyword_stats import CONTENT_COLUMN_KEYWORDS, _identify_content_columns
from utils.content_text import clean_text_like_keyword_stats
from utils.env_loader import get_env_config
from utils.path import ensure_task_dirs, get_config_dir, get_project_root, get_prompt_dir, get_task_process_dir
from utils.task_context import get_task_id


def setup_logger(topic: str, date_range: str) -> logging.Logger:
    """创建模块日志器（CLI 与工具共用）。"""
    logger_name = f"TopicBertopic.{topic}.{date_range}"
    logger = logging.getLogger(logger_name)
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
        logger.addHandler(handler)
    return logger


def log_success(logger: logging.Logger, message: str, _module: Optional[str] = None) -> None:
    logger.info(message)


def log_error(logger: logging.Logger, message: str, _module: Optional[str] = None) -> None:
    logger.error(message)


def log_module_start(logger: logging.Logger, module_name: str) -> None:
    logger.info("模块启动: %s", module_name)


def log_save_success(logger: logging.Logger, message: str, _module: Optional[str] = None) -> None:
    logger.info(message)


def load_env_file() -> None:
    """确保 .env 已加载（兼容模块内旧调用）。"""
    get_env_config()


def get_api_key() -> Optional[str]:
    """与 Sona 其它工具一致：优先 DASHSCOPE / QWEN 密钥。"""
    env = get_env_config()
    for name in ("DASHSCOPE_APIKEY", "QWEN_APIKEY", "OPENAI_APIKEY"):
        key = env.get_api_key(name)
        if key:
            return key
    return (
        os.environ.get("DASHSCOPE_API_KEY")
        or os.environ.get("QWEN_API_KEY")
        or os.environ.get("OPENAI_API_KEY")
        or os.environ.get("API_KEY")
    )


def bucket(*parts: str) -> Path:
    return get_project_root() / "data" / Path(*parts)


def read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


def _env_int(name: str, default: int, low: int, high: int) -> int:
    raw = str(os.environ.get(name, str(default))).strip()
    try:
        val = int(raw)
    except Exception:
        val = default
    return max(low, min(high, val))


_TOPIC_BERTOPIC_MAX_ROWS: int = _env_int("SONA_TOPIC_BERTOPIC_MAX_ROWS", 3000, 100, 50000)
_TOPIC_BERTOPIC_TIMEOUT_SEC: int = _env_int("SONA_TOPIC_BERTOPIC_TIMEOUT_SEC", 1800, 120, 7200)

# CLI 默认路径（舆情流水线请使用 analysis_topic_bertopic + dataFilePath）
_PROJECT_ROOT = get_project_root()
DEFAULT_INPUT_FILE = _PROJECT_ROOT / "data" / "bertopic_input.xlsx"
DEFAULT_OUTPUT_DIR = _PROJECT_ROOT / "data" / "bertopic_output"
DEFAULT_TEXT_COL = "正文"
TARGET_TOPICS = 8  # 建议合并后的议题数（可高于下限）
MIN_TOPIC_COUNT = 3  # 再聚类至少议题数（通过门槛；含程序生成的「无关主题」桶时一并计入）
MAX_TOPIC_LABEL_LEN = 10  # 主题命名最大字数（报告饼图/卡片展示）
_IRRELEVANT_TOPIC_NAME_MARKERS = frozenset(
    {"无关主题", "无关议题", "噪声内容", "噪声主题", "非模因噪声", "非模因噪声主题集"}
)
_PLACEHOLDER_TOPIC_NAME_FRAGMENTS = ("校验占位", "已弃用", "placeholder", "PLACEHOLDER")
LLM_MODEL_NAME = "qwen-plus"
RECLUSTER_PROMPT_YAML = "大模型再聚类.yaml"
RECLUSTER_PROMPT_KEY = "topic_bertopic_recluster"
RECLUSTER_REPAIR_JSON_KEY = "topic_bertopic_recluster_repair_json"
RECLUSTER_REPAIR_PARTITION_KEY = "topic_bertopic_recluster_repair_partition"


class QuotaExhaustedError(RuntimeError):
    """API额度耗尽：用于触发安全退出并保留已完成进度。"""


def _is_quota_exhausted_error(error_text: str) -> bool:
    s = (error_text or "").lower()
    needles = [
        "insufficient_quota",
        "quota",
        "余额不足",
        "额度不足",
        "credit",
        "account balance",
        "bill",
    ]
    return any(n in s for n in needles)

def _ensure_safe_runtime_env(project_root: Path, logger=None) -> None:
    """
    处理 Windows 环境下常见的编码/路径问题：
    - 某些三方库（UMAP/HDBSCAN/joblib/numba 等）可能在含中文的 TEMP 路径下触发 ASCII 编码异常
    - 将 TEMP/TMP 与常见缓存目录切到纯英文路径（不放在中文项目目录下）
    - 尽量强制 stdout/stderr 使用 UTF-8（避免输出阶段编码错误）
    """
    try:
        # 必须使用纯英文路径；若放在中文目录（如 D:\毕设数据处理）仍可能触发 loky ascii 编码报错
        # 允许通过环境变量覆盖，便于用户自定义到更快磁盘。
        safe_tmp_root_env = os.environ.get("BERTOPIC_SAFE_TMP", "").strip()
        if safe_tmp_root_env:
            safe_tmp_root = Path(safe_tmp_root_env)
        else:
            drive = Path(project_root).drive or "D:"
            safe_tmp_root = Path(f"{drive}\\bertopic_tmp")
        safe_tmp_root.mkdir(parents=True, exist_ok=True)

        # 1) 临时目录（避免落到 C:\\Users\\<中文用户名>\\AppData\\Local\\Temp）
        os.environ["TEMP"] = str(safe_tmp_root)
        os.environ["TMP"] = str(safe_tmp_root)

        # 2) 常见并行/缓存组件目录
        joblib_tmp = safe_tmp_root / "joblib"
        numba_cache = safe_tmp_root / "numba"
        joblib_tmp.mkdir(parents=True, exist_ok=True)
        numba_cache.mkdir(parents=True, exist_ok=True)
        os.environ["JOBLIB_TEMP_FOLDER"] = str(joblib_tmp)
        os.environ["NUMBA_CACHE_DIR"] = str(numba_cache)
        os.environ["LOKY_TEMP"] = str(joblib_tmp)

        # 3) 强制 UTF-8（尽量避免被当作 ascii 编码）
        os.environ.setdefault("PYTHONUTF8", "1")
        os.environ.setdefault("PYTHONIOENCODING", "utf-8")

        # 4) 尝试把当前进程 stdout/stderr 切到 utf-8（部分环境支持）
        for stream in (getattr(sys, "stdout", None), getattr(sys, "stderr", None)):
            if stream and hasattr(stream, "reconfigure"):
                try:
                    stream.reconfigure(encoding="utf-8", errors="replace")
                except Exception:
                    pass

        # 5) locale 兜底（不保证有效，但无副作用）
        try:
            locale.setlocale(locale.LC_ALL, "")
        except Exception:
            pass

        if logger:
            log_success(logger, f"已设置安全运行环境 TEMP/TMP -> {safe_tmp_root}", "TopicBertopic")
    except Exception as e:
        if logger:
            log_error(logger, f"设置安全运行环境失败（将继续执行）: {e}", "TopicBertopic")


def _extract_first_json_value(text: str):
    """
    从可能包含 Markdown/表格/解释性文字的 LLM 输出中尽量提取第一个合法 JSON 值（dict/list）。
    支持：
    - ```json ... ``` / ``` ... ``` code fence
    - 输出前后夹杂说明文字（扫描第一个 '{' 或 '[' 并用 raw_decode 解析）
    """
    if text is None:
        raise json.JSONDecodeError("Empty response", doc="", pos=0)

    s = str(text).strip()
    if not s:
        raise json.JSONDecodeError("Empty response", doc=s, pos=0)

    # 优先处理 code fence（只取第一个 fenced block）
    if "```" in s:
        parts = s.split("```")
        if len(parts) >= 3:
            fenced = parts[1]
            fenced_lines = fenced.splitlines()
            if fenced_lines and fenced_lines[0].strip().lower() in ("json", "application/json"):
                fenced = "\n".join(fenced_lines[1:])
            s = fenced.strip()

    # 最快路径：直接 loads
    try:
        return json.loads(s)
    except Exception:
        pass

    # 扫描提取第一个 JSON 对象/数组
    decoder = json.JSONDecoder()
    for i, ch in enumerate(s):
        if ch not in "{[":
            continue
        try:
            obj, _end = decoder.raw_decode(s[i:])
            return obj
        except Exception:
            continue

    # 兼容旧兜底：花括号/方括号片段
    m_obj = re.search(r"\{.*\}", s, re.S)
    if m_obj:
        return json.loads(m_obj.group(0).strip())
    m_arr = re.search(r"\[.*\]", s, re.S)
    if m_arr:
        return json.loads(m_arr.group(0).strip())

    raise json.JSONDecodeError("No JSON object/array found in response", doc=s, pos=0)


def _strip_llm_code_fence(text: str) -> str:
    s = str(text or "").strip()
    if "```" not in s:
        return s
    parts = s.split("```")
    if len(parts) < 3:
        return s
    fenced = parts[1]
    fenced_lines = fenced.splitlines()
    if fenced_lines and fenced_lines[0].strip().lower() in ("json", "application/json"):
        fenced = "\n".join(fenced_lines[1:])
    return fenced.strip()


def _salvage_truncated_merge_json(s: str) -> Optional[Dict]:
    """
    当模型输出在「合并方案」数组中途被 max_tokens 截断时，尝试补齐括号并解析。
    仅接受顶层含「合并方案」且为 list 的对象。
    """
    text = s.strip()
    if '"合并方案"' not in text and "合并方案" not in text:
        return None
    trial = text
    if trial.count('"') % 2 == 1:
        trial += '"'
    open_sq = trial.count("[") - trial.count("]")
    open_cu = trial.count("{") - trial.count("}")
    base_pad = "]" * max(0, open_sq) + "}" * max(0, open_cu)
    for extra in ("", '"}', '"]', '"]}', '"}]}', '"}]}}'):
        candidate = trial + base_pad + extra
        try:
            obj = json.loads(candidate)
            if isinstance(obj, dict) and isinstance(obj.get("合并方案"), list):
                return obj
        except Exception:
            continue
    return None


def _extract_merge_plan_json(text: str) -> Dict:
    """
    专用于解析大模型「合并方案」JSON。
    禁止误取内层单个主题对象（截断响应时 _extract_first_json_value 的常见误判）。
    """
    s = _strip_llm_code_fence(text)
    if not s:
        raise json.JSONDecodeError("Empty response", doc=s, pos=0)

    try:
        obj = json.loads(s)
        if isinstance(obj, dict) and isinstance(obj.get("合并方案"), list):
            return obj
    except Exception:
        salvaged_early = _salvage_truncated_merge_json(s)
        if salvaged_early:
            return salvaged_early

    start = s.find("{")
    if start >= 0:
        try:
            obj, _end = json.JSONDecoder().raw_decode(s, start)
            if isinstance(obj, dict) and isinstance(obj.get("合并方案"), list):
                return obj
        except Exception:
            pass
        salvaged = _salvage_truncated_merge_json(s[start:])
        if salvaged:
            return salvaged

    anchor = s.find('"合并方案"')
    if anchor < 0:
        anchor = s.find("合并方案")
    if anchor >= 0:
        root = s.rfind("{", 0, anchor)
        if root >= 0:
            try:
                obj, _end = json.JSONDecoder().raw_decode(s, root)
                if isinstance(obj, dict) and isinstance(obj.get("合并方案"), list):
                    return obj
            except Exception:
                salvaged = _salvage_truncated_merge_json(s[root:])
                if salvaged:
                    return salvaged

    raise json.JSONDecodeError("No valid merge-plan JSON found in response", doc=s, pos=0)


def _default_paths(topic: str, start_date: str, end_date: str = None) -> Dict[str, Path]:
    """
    获取默认路径，防止路径遍历攻击
    
    Args:
        topic: 专题名称（会进行安全校验）
        start_date: 开始日期
        end_date: 结束日期
    """
    # 安全校验：允许中文、字母、数字、下划线、连字符
    # 中文范围使用基本中日韩统一表意文字块（\u4e00-\u9fff）
    if not re.match(r'^[\w\-\u4e00-\u9fff]+$', topic):
        raise ValueError(f"专题名称包含非法字符: {topic}，只允许中文、字母、数字、下划线、连字符")
    
    project_root = get_project_root()
    configs_root = get_config_dir()
    # 使用fetch目录，与analyze模块一致
    if end_date:
        folder_name = f"{start_date}_{end_date}"
    else:
        folder_name = start_date
    
    # 使用Path.name确保只取文件名部分，防止路径遍历
    safe_topic = Path(topic).name
    fetch_dir = bucket("fetch", safe_topic, folder_name)
    userdict = configs_root / "userdict.txt"  # 可选：用户词典，使用项目统一配置
    stopwords = configs_root / "stopwords.txt"  # 使用项目统一配置
    # 输出路径使用相同的日期范围格式
    out_analyze = bucket("topic", safe_topic, folder_name)  # 输出到data/topic/{topic}/{date_range}/
    return {
        "fetch_dir": fetch_dir,
        "userdict": userdict,
        "stopwords": stopwords,
        "out_analyze": out_analyze,
    }


def _load_and_merge_fetch_data(fetch_dir: Path, logger) -> pd.DataFrame:
    """
    从fetch目录读取所有CSV文件并合并（与analyze模块一致）
    
    Args:
        fetch_dir (Path): fetch目录路径
        logger: 日志记录器
    
    Returns:
        pd.DataFrame: 合并后的数据框
    """
    if not fetch_dir.exists():
        log_error(logger, f"fetch目录不存在: {fetch_dir}", "TopicBertopic")
        return pd.DataFrame()
    
    # 读取总体.csv文件（包含所有渠道数据）
    overall_file = fetch_dir / "总体.csv"
    backup_file = fetch_dir / "总体.csv.raw.bak"
    
    # 优先尝试读取总体.csv
    if overall_file.exists():
        try:
            df = read_csv(overall_file)
            if not df.empty:
                log_success(logger, f"读取总体数据: {len(df)}条", "TopicBertopic")
                # 去重（基于contents字段）
                before_count = len(df)
                df = df.drop_duplicates(subset=['contents'], keep='last')
                after_count = len(df)
                if before_count != after_count:
                    log_success(logger, f"去重: {before_count} -> {after_count}", "TopicBertopic")
                log_success(logger, f"合并完成，共{len(df)}条数据", "TopicBertopic")
                return df
        except Exception as e:
            log_error(logger, f"读取总体数据失败: {e}", "TopicBertopic")
            # 如果总体.csv读取失败，尝试使用errors='replace'作为兜底
            try:
                encodings = ['utf-8', 'gb18030', 'gbk', 'utf-8-sig']
                for enc in encodings:
                    try:
                        with open(overall_file, 'r', encoding=enc, errors='replace') as f:
                            df = pd.read_csv(f, low_memory=False)
                        if not df.empty:
                            log_success(logger, f"使用编码{enc}和errors='replace'成功读取总体数据: {len(df)}条", "TopicBertopic")
                            # 去重（基于contents字段）
                            before_count = len(df)
                            df = df.drop_duplicates(subset=['contents'], keep='last')
                            after_count = len(df)
                            if before_count != after_count:
                                log_success(logger, f"去重: {before_count} -> {after_count}", "TopicBertopic")
                            log_success(logger, f"合并完成，共{len(df)}条数据", "TopicBertopic")
                            return df
                    except Exception:
                        continue
            except Exception as e2:
                log_error(logger, f"使用errors='replace'读取总体.csv也失败: {e2}", "TopicBertopic")
    
    # 如果总体.csv读取失败，尝试读取备份文件
    if backup_file.exists():
        try:
            df = read_csv(backup_file)
            if not df.empty:
                log_success(logger, f"读取备份文件: {len(df)}条", "TopicBertopic")
                # 去重（基于contents字段）
                before_count = len(df)
                df = df.drop_duplicates(subset=['contents'], keep='last')
                after_count = len(df)
                if before_count != after_count:
                    log_success(logger, f"去重: {before_count} -> {after_count}", "TopicBertopic")
                log_success(logger, f"合并完成，共{len(df)}条数据", "TopicBertopic")
                return df
        except Exception as e:
            log_error(logger, f"读取备份文件失败: {e}", "TopicBertopic")
            # 如果备份文件读取失败，尝试使用errors='replace'作为兜底
            try:
                encodings = ['utf-8', 'gb18030', 'gbk', 'utf-8-sig']
                for enc in encodings:
                    try:
                        with open(backup_file, 'r', encoding=enc, errors='replace') as f:
                            df = pd.read_csv(f, low_memory=False)
                        if not df.empty:
                            log_success(logger, f"使用编码{enc}和errors='replace'成功读取备份文件: {len(df)}条", "TopicBertopic")
                            # 去重（基于contents字段）
                            before_count = len(df)
                            df = df.drop_duplicates(subset=['contents'], keep='last')
                            after_count = len(df)
                            if before_count != after_count:
                                log_success(logger, f"去重: {before_count} -> {after_count}", "TopicBertopic")
                            log_success(logger, f"合并完成，共{len(df)}条数据", "TopicBertopic")
                            return df
                    except Exception:
                        continue
            except Exception as e2:
                log_error(logger, f"使用errors='replace'读取备份文件也失败: {e2}", "TopicBertopic")
    
    # 如果没有总体.csv，则读取各渠道CSV文件并合并
    csv_files = sorted([f for f in fetch_dir.glob("*.csv") if f.name != "总体.csv"])
    if not csv_files:
        log_error(logger, f"未找到CSV文件: {fetch_dir}", "TopicBertopic")
        return pd.DataFrame()
    
    log_success(logger, f"找到{len(csv_files)}个渠道CSV文件", "TopicBertopic")
    
    # 读取并合并所有文件
    all_data = []
    for file_path in csv_files:
        try:
            df = read_csv(file_path)
            if not df.empty:
                all_data.append(df)
                log_success(logger, f"读取: {file_path.name} - {len(df)}条", "TopicBertopic")
        except Exception as e:
            log_error(logger, f"读取失败 {file_path.name}: {e}", "TopicBertopic")
            continue
    
    if not all_data:
        log_error(logger, "没有读取到任何数据", "TopicBertopic")
        return pd.DataFrame()
    
    # 合并所有数据
    merged_df = pd.concat(all_data, ignore_index=True)
    
    # 去重（基于contents字段）
    before_count = len(merged_df)
    merged_df = merged_df.drop_duplicates(subset=['contents'], keep='last')
    after_count = len(merged_df)
    
    if before_count != after_count:
        log_success(logger, f"合并后去重: {before_count} -> {after_count}", "TopicBertopic")
    
    log_success(logger, f"合并完成，共{len(merged_df)}条数据", "TopicBertopic")
    return merged_df


def _resolve_text_column(df: pd.DataFrame, text_col_preferred: Optional[str] = None) -> Optional[str]:
    """解析建模用文本列；指定列名时严格使用该列（用于对比实验固定「正文」）。"""
    if text_col_preferred:
        preferred = str(text_col_preferred).strip()
        if preferred in df.columns:
            return preferred
        lower_map = {str(c).strip().lower(): c for c in df.columns}
        key = preferred.lower()
        if key in lower_map:
            return lower_map[key]
        return None

    candidate_cols = [
        "向量化正文",
        "contents",
        "正文",
        "effective_text",
        "llm_fixed_5w",
        "llm_summary",
    ]
    lower_map = {str(c).strip().lower(): c for c in df.columns}
    for c in candidate_cols:
        if c in df.columns:
            return c
        c_lower = c.strip().lower()
        if c_lower in lower_map:
            return lower_map[c_lower]
    for c in df.columns:
        if "contents" in str(c).lower():
            return c
    return None


def _load_input_table(input_file: Path, logger) -> pd.DataFrame:
    """
    直接加载单个输入文件（xlsx/csv）。
    用于绕过 fetch 目录，直接对成品表做 BERTopic。
    """
    if not input_file.exists():
        log_error(logger, f"输入文件不存在: {input_file}", "TopicBertopic")
        return pd.DataFrame()
    try:
        suffix = input_file.suffix.lower()
        if suffix in {".xlsx", ".xls"}:
            df = pd.read_excel(input_file)
        elif suffix == ".csv":
            df = read_csv(input_file)
        else:
            log_error(logger, f"不支持的输入格式: {input_file.suffix}", "TopicBertopic")
            return pd.DataFrame()
        if logger:
            log_success(logger, f"读取输入文件: {input_file.name} | {len(df)}条", "TopicBertopic")
        return df
    except Exception as e:
        log_error(logger, f"读取输入文件失败: {e}", "TopicBertopic")
        return pd.DataFrame()


def _clean_text(text: str) -> str:
    if not text:
        return ""
    text = re.sub(r'[^\u4e00-\u9fa5\u3000-\u303f0-9，。！？；：、（）《》【】""''\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def _clean_batch_with_indices(
    texts: List[str], source_indices: List[Any]
) -> Tuple[List[str], List[Any], Dict[str, int], Dict[str, int]]:
    """清洗并去重；保留每行对应的上游索引；ct_to_seg_idx 将清洗后文本映射到 seg 下标（首个）。"""
    cleaned: List[str] = []
    kept_sources: List[Any] = []
    ct_to_seg_idx: Dict[str, int] = {}
    seen_ct = set()
    stats = {"total": 0, "duplicates": 0, "final": 0}
    for t, src_i in zip(texts, source_indices):
        stats["total"] += 1
        ct = _clean_text(t)
        if not ct:
            continue
        if ct in seen_ct:
            stats["duplicates"] += 1
            continue
        seen_ct.add(ct)
        ct_to_seg_idx[ct] = len(cleaned)
        cleaned.append(ct)
        kept_sources.append(src_i)
    stats["final"] = len(cleaned)
    return cleaned, kept_sources, ct_to_seg_idx, stats


def _clean_batch(texts: List[str]) -> Tuple[List[str], Dict[str, int]]:
    dummy = list(range(len(texts)))
    cleaned, _, _, stats = _clean_batch_with_indices(texts, dummy)
    return cleaned, stats


def _pick_document_id_column(df: pd.DataFrame) -> Optional[str]:
    """优先匹配常见「文档编号」列名；找不到则返回 None（导出时用 DataFrame 行标签）。"""
    candidates = [
        "文档原有编号",
        "文档编号",
        "编号",
        "序号",
        "稿件编号",
        "原始编号",
        "id",
        "ID",
        "doc_id",
        "Doc_ID",
    ]
    lower_map = {str(c).strip().lower(): c for c in df.columns}
    for c in candidates:
        if c in df.columns:
            return c
        cl = c.strip().lower()
        if cl in lower_map:
            return lower_map[cl]
    return None


def _build_doc_index_to_final_mapping(final_result: Dict) -> Dict[int, Tuple[str, str]]:
    """seg 文档下标 -> (最终归属主题键, 最终主题名)"""
    out: Dict[int, Tuple[str, str]] = {}
    for new_key, info in final_result.items():
        if not isinstance(info, dict):
            continue
        naming = str(info.get("主题命名", "") or "")
        for di in info.get("文档ID", []) or []:
            try:
                out[int(di)] = (str(new_key), naming)
            except (TypeError, ValueError):
                continue
    return out


def _write_document_topic_table_csv(
    out_dir: Path,
    final_result: Dict,
    topic_stats: Dict,
    df: pd.DataFrame,
    text_col: str,
    doc_id_col: Optional[str],
    ct_to_seg_idx: Dict[str, int],
    logger,
) -> None:
    """
    导出四列：文档原有编号、原文列（与建模列同名）、最终归属主题、最终主题名。
    与建模一致的文本行参与导出；清洗后重复的文档映射到同一 seg 下标与最终主题。
    """
    coords = topic_stats.get("文档2D坐标") or []
    doc_final = _build_doc_index_to_final_mapping(final_result)

    rows_out: List[Dict[str, Any]] = []
    for idx in df.index:
        raw = df.at[idx, text_col]
        s = str(raw).strip()
        if not s or s.lower() in ("nan", "none", ""):
            continue
        ct = _clean_text(s)
        if not ct:
            continue
        seg_idx = ct_to_seg_idx.get(ct)
        if seg_idx is None:
            continue
        if seg_idx >= len(coords):
            log_error(logger, f"文档行对齐异常: seg_idx={seg_idx} 超出坐标表长度", "TopicBertopic")
            continue
        tid = int(coords[seg_idx].get("topic_id", -999))
        if tid == -1:
            final_key, final_name = "噪声", "未归入BERTopic簇"
        elif seg_idx in doc_final:
            final_key, final_name = doc_final[seg_idx]
        else:
            final_key, final_name = "未映射", "未找到再聚类归属"

        if doc_id_col and doc_id_col in df.columns:
            orig_id = df.at[idx, doc_id_col]
        else:
            orig_id = idx
        if pd.isna(orig_id):
            orig_id = ""
        elif not isinstance(orig_id, str):
            orig_id = str(orig_id)

        rows_out.append(
            {
                "文档原有编号": orig_id,
                text_col: s,
                "最终归属主题": final_key,
                "最终主题名": final_name,
            }
        )

    out_path = out_dir / "6文档主题归属表.csv"
    pd.DataFrame(rows_out).to_csv(out_path, index=False, encoding="utf-8-sig")
    log_success(logger, f"已写入文档主题归属表: {out_path}（共 {len(rows_out)} 行）", "TopicBertopic")


def _load_stopwords(path: Path) -> List[str]:
    if path.exists():
        return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return []


def _segment(texts: List[str], stopwords: List[str], userdict: Optional[Path]) -> List[str]:
    if userdict and userdict.exists():
        jieba.load_userdict(str(userdict))
    stopset = set(stopwords)
    result: List[str] = []
    for t in texts:
        words = [w for w in jieba.cut(t) if len(w) >= 2 and not w.isdigit() and w not in stopset]
        result.append(" ".join(words))
    return result


def _get_text_hash(text: str) -> str:
    """生成文本的MD5哈希值作为唯一标识"""
    return hashlib.md5(text.encode('utf-8')).hexdigest()


def _embedding_checkpoint_path(cache_file: Optional[Path]) -> Optional[Path]:
    if not cache_file:
        return None
    return cache_file.with_suffix(".embedding_ckpt.json")


def _save_run_embedding_matrix(
    out_dir: Path,
    embeddings: np.ndarray,
    *,
    texts: Optional[List[str]] = None,
    logger: Optional[logging.Logger] = None,
    basename: str = "text_embeddings",
    model: str = "text-embedding-v4",
) -> Tuple[Path, Path]:
    """保存本次聚类对齐的向量矩阵（.npy）及元数据（.json），供后续再分析复用。"""
    out_dir.mkdir(parents=True, exist_ok=True)
    npy_path = out_dir / f"{basename}.npy"
    meta_path = out_dir / f"{basename}.json"

    arr = np.asarray(embeddings, dtype=np.float32)
    npy_tmp = npy_path.with_suffix(".npy.tmp")
    with open(npy_tmp, "wb") as f:
        np.save(f, arr)
    os.replace(npy_tmp, npy_path)

    meta: Dict[str, Any] = {
        "format": "sona_text_embeddings_v1",
        "npy_file": npy_path.name,
        "shape": list(arr.shape),
        "dtype": "float32",
        "model": model,
        "dimensions": int(arr.shape[1]) if arr.ndim == 2 else 0,
        "text_count": int(arr.shape[0]) if arr.ndim >= 1 else 0,
        "generated_at": datetime.now().isoformat(sep=" "),
    }
    if texts is not None:
        meta["text_count"] = len(texts)
    meta_tmp = meta_path.with_suffix(".json.tmp")
    with open(meta_tmp, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    os.replace(meta_tmp, meta_path)

    if logger:
        log_success(
            logger,
            f"已保存向量化矩阵: {npy_path} shape={arr.shape}",
            "TopicBertopic",
        )
    return npy_path, meta_path


def _sync_topic_embeddings_to_process_dir(
    process_dir: Path,
    artifact_dir: Path,
    *,
    run_stamp: str,
    logger: Optional[logging.Logger] = None,
) -> Dict[str, str]:
    """将 artifact 子目录中的 .npy 向量复制到过程目录（latest + 时间戳副本）。"""
    paths: Dict[str, str] = {}
    pairs = (
        ("text_embeddings.npy", "text_embeddings.json"),
        ("embedding_cache.npy", "embedding_cache.json"),
    )
    process_dir.mkdir(parents=True, exist_ok=True)

    for npy_name, json_name in pairs:
        src_npy = artifact_dir / npy_name
        src_meta = artifact_dir / json_name
        if not src_npy.exists():
            continue
        prefix = "topic_bertopic_embeddings" if npy_name.startswith("text_") else "topic_bertopic_embedding_cache"
        latest_npy = process_dir / f"{prefix}_latest.npy"
        latest_meta = process_dir / f"{prefix}_latest.json"
        stamped_npy = process_dir / f"{prefix}_{run_stamp}.npy"
        stamped_meta = process_dir / f"{prefix}_{run_stamp}.json"
        for src, dst in (
            (src_npy, latest_npy),
            (src_meta, latest_meta),
            (src_npy, stamped_npy),
            (src_meta, stamped_meta),
        ):
            if src.exists():
                shutil.copy2(src, dst)
        paths[f"{prefix}_latest_npy"] = str(latest_npy)
        paths[f"{prefix}_latest_meta"] = str(latest_meta)
        if logger:
            log_success(logger, f"向量已同步至过程目录: {latest_npy}", "TopicBertopic")

    return paths


def _load_embedding_checkpoint(cache_file: Optional[Path], logger=None) -> Dict:
    ckpt = _embedding_checkpoint_path(cache_file)
    if not ckpt or not ckpt.exists():
        return {}
    try:
        with open(ckpt, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            return data
        return {}
    except Exception as e:
        if logger:
            log_error(logger, f"加载 embedding checkpoint 失败: {e}", "TopicBertopic")
        return {}


def _save_embedding_checkpoint(cache_file: Optional[Path], payload: Dict, logger=None) -> None:
    ckpt = _embedding_checkpoint_path(cache_file)
    if not ckpt:
        return
    try:
        ckpt.parent.mkdir(parents=True, exist_ok=True)
        tmp = ckpt.with_suffix(".embedding_ckpt.json.tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        os.replace(tmp, ckpt)  # 原子替换，避免中断导致半文件
    except Exception as e:
        if logger:
            log_error(logger, f"保存 embedding checkpoint 失败: {e}", "TopicBertopic")


def _load_embedding_cache(cache_file: Path, logger=None) -> Dict[str, np.ndarray]:
    """
    加载向量缓存（使用安全的JSON+NPY格式，避免pickle反序列化漏洞）
    
    缓存格式：
    - cache_file.json: 存储 {text_hash: index} 映射
    - cache_file.npy: 存储所有向量（按index顺序）
    
    兼容性：如果存在旧的.pkl文件，会自动迁移到新格式
    """
    json_file = cache_file.with_suffix('.json')
    npy_file = cache_file.with_suffix('.npy')
    old_pkl_file = cache_file.with_suffix('.pkl')
    
    # 如果存在旧的pickle文件，尝试迁移
    if old_pkl_file.exists() and (not json_file.exists() or not npy_file.exists()):
        if logger:
            log_success(logger, f"检测到旧格式缓存文件，正在迁移到安全格式...", "TopicBertopic")
        try:
            # 加载旧格式
            with open(old_pkl_file, 'rb') as f:
                old_cache = pickle.load(f)
            
            # 转换为新格式并保存
            if old_cache:
                _save_embedding_cache(old_cache, cache_file, logger)
                if logger:
                    log_success(logger, f"缓存迁移完成，已删除旧文件", "TopicBertopic")
                # 删除旧文件
                old_pkl_file.unlink(missing_ok=True)
        except Exception as e:
            if logger:
                log_error(logger, f"缓存迁移失败: {e}，将使用新格式", "TopicBertopic")
    
    if not json_file.exists() or not npy_file.exists():
        return {}
    
    try:
        # 加载JSON映射
        with open(json_file, 'r', encoding='utf-8') as f:
            hash_to_index = json.load(f)
        
        # 加载向量数组（使用内存映射，节省内存）
        vectors = np.load(npy_file, mmap_mode='r')
        
        # 重建字典
        # 注意：这里将所有向量复制到内存字典中，对于30万条数据会占用约1.2GB内存
        # 如果内存紧张（<16GB），可以考虑延迟加载：只保留hash_to_index映射，
        # 在需要时直接从vectors memmap中按索引读取（需要修改_embed函数的逻辑）
        cache = {}
        for text_hash, idx in hash_to_index.items():
            if 0 <= idx < len(vectors):
                cache[text_hash] = vectors[idx].copy()  # 复制到内存（避免mmap问题）
        
        if logger:
            log_success(logger, f"加载向量缓存: {len(cache)}条已向量化的文本", "TopicBertopic")
        return cache
    except Exception as e:
        if logger:
            log_error(logger, f"加载向量缓存失败: {e}", "TopicBertopic")
        return {}


def _save_embedding_cache(cache: Dict[str, np.ndarray], cache_file: Path, logger=None):
    """
    保存向量缓存（使用安全的JSON+NPY格式，避免pickle反序列化漏洞）
    """
    if not cache:
        return
    
    try:
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        
        # 设置文件权限：仅当前用户可读写（Unix/Linux）
        if hasattr(os, 'chmod'):
            os.chmod(cache_file.parent, 0o700)
        
        json_file = cache_file.with_suffix('.json')
        npy_file = cache_file.with_suffix('.npy')
        
        # 构建向量数组和索引映射
        vectors = []
        hash_to_index = {}
        
        for idx, (text_hash, vec) in enumerate(cache.items()):
            hash_to_index[text_hash] = idx
            vectors.append(vec)
        
        if vectors:
            vectors_array = np.array(vectors, dtype=np.float32)

            # 原子写：先写 tmp，再 replace，避免中断时 json/npy 不一致
            json_tmp = json_file.with_suffix(".json.tmp")
            npy_tmp = npy_file.with_suffix(".npy.tmp")

            with open(json_tmp, 'w', encoding='utf-8') as f:
                json.dump(hash_to_index, f, ensure_ascii=False)

            with open(npy_tmp, "wb") as f:
                np.save(f, vectors_array)

            os.replace(json_tmp, json_file)
            os.replace(npy_tmp, npy_file)
            
            # 设置文件权限
            if hasattr(os, 'chmod'):
                os.chmod(json_file, 0o600)
                os.chmod(npy_file, 0o600)
        
        if logger:
            log_success(logger, f"保存向量缓存: {len(cache)}条向量", "TopicBertopic")
    except Exception as e:
        if logger:
            log_error(logger, f"保存向量缓存失败: {e}", "TopicBertopic")


async def _embed_async_batches(
    texts_to_embed: List[Tuple[int, str, str]], api_key: str, base_url: str,
    model: str, dimensions: int, batch_size: int, MAX_INPUT_LENGTH: int,
    cache: Dict[str, np.ndarray], new_vecs_dict: Dict[str, np.ndarray],
    cache_file: Optional[Path], total_batches: int, logger,
    semaphore_limit: int = 3,
    batch_max_retries: int = 6,
    save_interval: int = 100,
    window_size: int = 200,
    request_timeout_s: int = 120,
):
    """
    异步并发处理向量化批次，带指数退避重试机制
    
    优化：
    1. 使用滑动窗口控制并发，避免一次性创建3万个协程导致内存爆炸
    2. 添加指数退避重试，应对429限流
    3. 批量处理，减少内存占用
    """
    client = AsyncOpenAI(api_key=api_key, base_url=base_url)
    semaphore = asyncio.Semaphore(max(1, int(semaphore_limit)))  # 控制并发数量，避免触发API限流
    completed_batches = 0
    lock = asyncio.Lock()
    failed_batches: List[Dict] = []
    SAVE_INTERVAL = max(1, int(save_interval))
    
    async def process_batch_with_retry(batch_items: List[Tuple[int, str, str]], batch_num: int, max_retries: int = 5):
        """
        处理单个批次，带指数退避重试
        
        Args:
            batch_items: 批次数据
            batch_num: 批次编号
            max_retries: 最大重试次数
        """
        nonlocal completed_batches
        sub = [item[1] for item in batch_items]  # 提取文本
        batch_hashes = [item[2] for item in batch_items]  # 提取哈希值
        
        # 防御性检查
        max_len_in_batch = max(len(t) for t in sub) if sub else 0
        if max_len_in_batch > MAX_INPUT_LENGTH:
            if logger:
                log_error(logger, f"批次{batch_num}中发现超长文本: {max_len_in_batch}字符，已截断", "TopicBertopic")
            sub = [text[:MAX_INPUT_LENGTH] if len(text) > MAX_INPUT_LENGTH else text for text in sub]
        
        # 指数退避重试
        max_retries = max(1, int(batch_max_retries))
        for attempt in range(max_retries):
            try:
                async with semaphore:  # 控制并发数量
                    resp = await asyncio.wait_for(
                        client.embeddings.create(
                            model=model, input=sub, dimensions=dimensions, encoding_format="float"
                        ),
                        timeout=max(30, int(request_timeout_s)),
                    )
                    
                    # 保存新向量到缓存和结果字典（需要加锁）
                    async with lock:
                        for j, embedding_item in enumerate(resp.data):
                            text_hash = batch_hashes[j]
                            vec = np.array(embedding_item.embedding, dtype=np.float32)
                            cache[text_hash] = vec
                            new_vecs_dict[text_hash] = vec
                        
                        completed_batches += 1
                        
                        if cache_file and completed_batches % SAVE_INTERVAL == 0:
                            if logger:
                                log_success(logger, f"正在执行定期检查点保存（已完成{completed_batches}批次）...", "TopicBertopic")
                            _save_embedding_cache(cache, cache_file, logger)
                            _save_embedding_checkpoint(
                                cache_file,
                                {
                                    "status": "running",
                                    "completed_batches": completed_batches,
                                    "total_batches": total_batches,
                                    "failed_batches": failed_batches,
                                },
                                logger,
                            )
                            if logger:
                                progress_pct = (completed_batches / total_batches * 100)
                                log_success(logger, f"检查点保存完成，向量化进度: {completed_batches}/{total_batches}批次 ({progress_pct:.1f}%)", "TopicBertopic")
                        
                        # 进度日志保持高频（每100批次），方便实时监控，不影响性能
                        elif logger and completed_batches % 100 == 0:
                            progress_pct = (completed_batches / total_batches * 100)
                            log_success(logger, f"向量化进度: {completed_batches}/{total_batches}批次 ({progress_pct:.1f}%)", "TopicBertopic")
                    
                    return  # 成功，退出重试循环
                    
            except Exception as e:
                error_str = str(e)
                if _is_quota_exhausted_error(error_str):
                    async with lock:
                        failed_batches.append(
                            {
                                "batch_num": batch_num,
                                "hashes": batch_hashes,
                                "reason": "quota_exhausted",
                                "error": error_str[:500],
                            }
                        )
                        if cache_file and cache:
                            _save_embedding_cache(cache, cache_file, logger)
                        _save_embedding_checkpoint(
                            cache_file,
                            {
                                "status": "quota_exhausted",
                                "completed_batches": completed_batches,
                                "total_batches": total_batches,
                                "failed_batches": failed_batches,
                            },
                            logger,
                        )
                    raise QuotaExhaustedError(f"额度耗尽，停止于批次{batch_num}: {error_str}")
                is_rate_limit = '429' in error_str or 'rate limit' in error_str.lower() or 'too many requests' in error_str.lower()
                
                if attempt < max_retries - 1:
                    # 指数退避：2^attempt 秒
                    wait_time = min((2 ** attempt) + random.uniform(0.0, 1.0), 60)  # 最多等待60秒
                    if logger:
                        if is_rate_limit:
                            log_error(logger, f"批次{batch_num}触发限流，{wait_time}秒后重试 (尝试 {attempt+1}/{max_retries})", "TopicBertopic")
                        else:
                            log_error(logger, f"批次{batch_num}失败，{wait_time}秒后重试 (尝试 {attempt+1}/{max_retries}): {type(e).__name__}", "TopicBertopic")
                    await asyncio.sleep(wait_time)
                else:
                    # 最后一次重试失败
                    if logger:
                        max_len = max(len(t) for t in sub) if sub else 0
                        log_error(logger, f"向量化批次{batch_num}最终失败（已重试{max_retries}次）: {e}", "TopicBertopic")
                        log_error(logger, f"批次大小: {len(sub)}, 最大文本长度: {max_len}字符", "TopicBertopic")
                    # 异常时保存已完成的缓存
                    async with lock:
                        failed_batches.append(
                            {
                                "batch_num": batch_num,
                                "hashes": batch_hashes,
                                "reason": "retries_exhausted",
                                "error": error_str[:500],
                            }
                        )
                        if cache_file and cache:
                            _save_embedding_cache(cache, cache_file, logger)
                        _save_embedding_checkpoint(
                            cache_file,
                            {
                                "status": "failed",
                                "completed_batches": completed_batches,
                                "total_batches": total_batches,
                                "failed_batches": failed_batches,
                            },
                            logger,
                        )
                    raise  # 重试耗尽，抛出异常
    
    # 使用滑动窗口批量处理，避免一次性创建3万个协程导致内存爆炸
    # 每次只处理1000个批次，处理完一批再处理下一批
    window_size = max(1, int(window_size))
    all_batches = []
    for i in range(0, len(texts_to_embed), batch_size):
        batch_items = texts_to_embed[i:i + batch_size]
        batch_num = i // batch_size + 1
        all_batches.append((batch_items, batch_num))
    
    # 分批处理，避免内存爆炸
    for window_start in range(0, len(all_batches), window_size):
        window_batches = all_batches[window_start:window_start + window_size]
        tasks = [process_batch_with_retry(batch_items, batch_num) for batch_items, batch_num in window_batches]
        
        # 并发执行当前窗口的批次（严格模式：任一失败立即抛出，避免静默丢向量）
        await asyncio.gather(*tasks)

    # 全部成功后写最终 checkpoint
    _save_embedding_checkpoint(
        cache_file,
        {
            "status": "completed",
            "completed_batches": total_batches,
            "total_batches": total_batches,
            "failed_batches": [],
        },
        logger,
    )
    return {
        "completed_batches": total_batches,
        "failed_batches": failed_batches,
    }


def _embed(batch_texts: List[str], api_key: str, logger=None, base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1",
           model: str = "text-embedding-v4", dimensions: int = 1024, batch_size: int = 10,
           cache_file: Optional[Path] = None,
           semaphore_limit: int = 3,
           batch_max_retries: int = 6,
           save_interval: int = 100,
           window_size: int = 200,
           request_timeout_s: int = 120) -> np.ndarray:
    """
    生成文本向量，支持断点续传
    
    Args:
        batch_texts: 文本列表
        api_key: API密钥
        logger: 日志记录器
        base_url: API基础URL
        model: 模型名称
        dimensions: 向量维度
        batch_size: 批次大小
        cache_file: 缓存文件路径（可选）
    
    Returns:
        np.ndarray: 向量数组
    """
    if not batch_texts:
        return np.array([])
    
    # 处理超长文本，保留原始索引
    # API限制是[1, 8192]，但实际可能按token计算，为安全起见截断到8000字符
    MAX_INPUT_LENGTH = 4096
    processed_texts = []
    valid_indices = []
    truncated_count = 0
    
    for idx, text in enumerate(batch_texts):
        if len(text) > MAX_INPUT_LENGTH:
            processed_texts.append(text[:MAX_INPUT_LENGTH])
            valid_indices.append(idx)
            truncated_count += 1
        elif len(text) == 0:
            # 跳过空文本，但记录索引
            continue
        else:
            processed_texts.append(text)
            valid_indices.append(idx)
    
    if truncated_count > 0:
        if logger:
            log_success(logger, f"{truncated_count}条文本因超长被截断", "TopicBertopic")
    
    if not processed_texts:
        return np.array([])
    
    # 加载缓存
    cache = {}
    if cache_file:
        cache = _load_embedding_cache(cache_file, logger)
    
    # 分离需要向量化的文本和已缓存的文本
    texts_to_embed = []
    cached_vecs = {}
    text_hash_to_index = {}  # 哈希值到processed_texts索引的映射
    
    for idx, text in enumerate(processed_texts):
        text_hash = _get_text_hash(text)
        text_hash_to_index[text_hash] = idx
        if text_hash in cache:
            cached_vecs[text_hash] = cache[text_hash]
        else:
            texts_to_embed.append((idx, text, text_hash))

    # 续跑优化：若上次 checkpoint 记录了失败批次哈希，本次优先重试这些文本
    ck = _load_embedding_checkpoint(cache_file, logger)
    failed_hashes = []
    try:
        for fb in ck.get("failed_batches", []):
            failed_hashes.extend(fb.get("hashes", []))
    except Exception:
        failed_hashes = []
    if failed_hashes and texts_to_embed:
        failed_set = set(failed_hashes)
        pri = [x for x in texts_to_embed if x[2] in failed_set]
        rest = [x for x in texts_to_embed if x[2] not in failed_set]
        if pri:
            texts_to_embed = pri + rest
            if logger:
                log_success(logger, f"检测到上次失败批次，优先重试 {len(pri)} 条文本", "TopicBertopic")
    
    cached_count = len(cached_vecs)
    new_count = len(texts_to_embed)
    
    if logger:
        log_success(logger, f"向量化统计: 缓存命中{cached_count}条，需新向量化{new_count}条", "TopicBertopic")
    
    # 只对未缓存的文本进行向量化
    new_vecs_dict = {}  # 存储新向量化的结果：text_hash -> vec
    if texts_to_embed:
        total_batches = (len(texts_to_embed) + batch_size - 1) // batch_size
        if logger:
            log_success(logger, f"开始生成向量，共{total_batches}个批次，每批{batch_size}条（并发处理）", "TopicBertopic")
        _save_embedding_checkpoint(
            cache_file,
            {
                "status": "running",
                "completed_batches": 0,
                "total_batches": total_batches,
                "failed_batches": [],
            },
            logger,
        )
        
        # 使用异步并发处理提升速度
        try:
            run_info = asyncio.run(_embed_async_batches(
                texts_to_embed, api_key, base_url, model, dimensions, batch_size,
                MAX_INPUT_LENGTH, cache, new_vecs_dict, cache_file, total_batches, logger,
                semaphore_limit=semaphore_limit,
                batch_max_retries=batch_max_retries,
                save_interval=save_interval,
                window_size=window_size,
                request_timeout_s=request_timeout_s,
            ))
            _save_embedding_checkpoint(
                cache_file,
                {
                    "status": "running",
                    "completed_batches": int((run_info or {}).get("completed_batches", 0)),
                    "total_batches": total_batches,
                    "failed_batches": (run_info or {}).get("failed_batches", []),
                },
                logger,
            )
        except QuotaExhaustedError:
            if logger:
                log_error(logger, "检测到API额度耗尽：已强制保存缓存与checkpoint，请充值后重跑续传。", "TopicBertopic")
            if cache_file and cache:
                _save_embedding_cache(cache, cache_file, logger)
            raise
        except KeyboardInterrupt:
            # 手动中断（Ctrl+C）时保存已完成的缓存
            if logger:
                log_error(logger, "向量化被用户中断，正在保存已完成的缓存...", "TopicBertopic")
            if cache_file and cache:
                _save_embedding_cache(cache, cache_file, logger)
                if logger:
                    log_success(logger, f"已保存缓存: {len(cache)}条向量，下次运行将自动续传", "TopicBertopic")
            _save_embedding_checkpoint(
                cache_file,
                {
                    "status": "interrupted",
                    "completed_batches": 0,
                    "total_batches": total_batches,
                    "failed_batches": [],
                },
                logger,
            )
            raise  # 重新抛出异常，让外层处理
        except Exception as e:
            if cache_file and cache:
                _save_embedding_cache(cache, cache_file, logger)
            _save_embedding_checkpoint(
                cache_file,
                {
                    "status": "failed",
                    "completed_batches": 0,
                    "total_batches": total_batches,
                    "failed_batches": [{"batch_num": -1, "hashes": [], "reason": "unexpected", "error": str(e)[:500]}],
                },
                logger,
            )
            raise
        
        # 循环结束后保存最终缓存，确保所有数据都已保存
        if cache_file:
            _save_embedding_cache(cache, cache_file, logger)
    
    # 构建结果向量数组（按processed_texts的顺序）
    # 对于超大数据集（>20万条），使用memmap减少内存占用
    use_memmap = len(processed_texts) > 200000
    if use_memmap and cache_file:
        # 使用临时memmap文件存储向量
        memmap_file = cache_file.with_suffix('.memmap.npy')
        result_vecs = np.memmap(memmap_file, dtype=np.float32, mode='w+', 
                                shape=(len(processed_texts), dimensions))
        if logger:
            log_success(logger, f"使用内存映射文件存储向量（节省内存）: {memmap_file}", "TopicBertopic")
    else:
        result_vecs = np.zeros((len(processed_texts), dimensions), dtype=np.float32)
    
    # 填充所有向量（缓存的和新向量化的）
    for idx, text in enumerate(processed_texts):
        text_hash = _get_text_hash(text)
        if text_hash in cached_vecs:
            result_vecs[idx] = cached_vecs[text_hash]
        elif text_hash in new_vecs_dict:
            result_vecs[idx] = new_vecs_dict[text_hash]
    
    # 如果有空文本，需要调整结果向量的维度
    if len(valid_indices) < len(batch_texts):
        final_vecs = np.zeros((len(batch_texts), dimensions), dtype=np.float32)
        for i, vec_idx in enumerate(valid_indices):
            final_vecs[vec_idx] = result_vecs[i]
        # 清理memmap文件
        if use_memmap and cache_file:
            del result_vecs
            memmap_file.unlink(missing_ok=True)
        return final_vecs
    
    # 如果使用memmap，需要转换为普通数组供BERTopic使用（BERTopic需要内存数组）
    if use_memmap and cache_file:
        # 将memmap转换为普通数组
        result_array = np.array(result_vecs, dtype=np.float32)
        del result_vecs  # 释放memmap
        memmap_file.unlink(missing_ok=True)  # 删除临时文件
        return result_array
    
    return result_vecs


def _build_bertopic() -> BERTopic:
    if BERTopic is None or UMAP is None or HDBSCAN is None:
        raise RuntimeError(
            "BERTopic 依赖导入失败。常见原因是 numba 与 numpy 版本不兼容。"
            "请执行: pip install \"numpy<=2.3\" \"numba>=0.60\" \"umap-learn\" \"hdbscan\" \"bertopic\""
            f" | 原始错误: {_IMPORT_ERROR}"
        )
    # 添加 low_memory=True 避免内存溢出（20万条数据需要大量内存）
    umap_model = UMAP(n_neighbors=15, n_components=5, min_dist=0.0, metric='cosine', random_state=42)
    hdbscan_model = HDBSCAN(min_cluster_size=15, min_samples=5, metric='euclidean')
    vectorizer_model = CountVectorizer(stop_words=['控烟', '吸烟'])
    return BERTopic(
        nr_topics=30,
        top_n_words=20,
        vectorizer_model=vectorizer_model,
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        language="multilingual",
        calculate_probabilities=False,
        verbose=False # 关闭BERTopic的详细日志输出
    )


def _generate_jsons(topic_model: BERTopic, documents: List[str], embeddings: np.ndarray,
                    out_dir: Path, logger) -> Dict:
    if UMAP is None:
        raise RuntimeError(
            "UMAP 依赖不可用，请先修复环境。建议执行: pip install \"numpy<=2.3\" \"numba>=0.60\" \"umap-learn\""
        )
    out_dir.mkdir(parents=True, exist_ok=True)
    topic_info = topic_model.get_topic_info()
    topics = topic_model.topics_

    umap_2d = UMAP(n_neighbors=15, n_components=2, min_dist=0.0, metric='cosine', random_state=42)
    coords = umap_2d.fit_transform(embeddings)

    # 主题文档统计
    topic_docs: Dict[str, Dict] = {}
    valid_topic_count = 0
    noise_doc_count = 0
    for tid in topic_info['Topic']:
        if tid != -1:
            idxs = [i for i, t in enumerate(topics) if t == tid]
            topic_docs[f"主题{tid}"] = {"文档数": len(idxs), "文档ID": idxs}
            valid_topic_count += 1
        else:
            # 统计噪声主题文档数
            noise_doc_count = len([i for i, t in enumerate(topics) if t == -1])
    
    # 记录初步聚类主题数量
    log_success(logger, f"初步聚类完成: 有效主题数={valid_topic_count}, 噪声文档数={noise_doc_count}", "TopicBertopic")

    # 主题关键词
    topic_keywords: Dict[str, Dict] = {}
    for tid in topic_info['Topic']:
        if tid != -1:
            kws = topic_model.get_topic(tid)
            top20 = kws[:20] if len(kws) >= 20 else kws
            topic_keywords[f"主题{tid}"] = {"关键词": [[w, float(s)] for w, s in top20]}

    # 文档2D坐标
    doc_coords = [
        {"doc_id": i, "topic_id": int(topics[i]), "x": float(coords[i][0]), "y": float(coords[i][1])}
        for i in range(len(documents))
    ]

    stats_result = {
        "主题文档统计": topic_docs,
        "主题关键词": topic_keywords,
        "文档2D坐标": doc_coords,
    }

    p1 = out_dir / "1主题统计结果.json"
    p2 = out_dir / "2主题关键词.json"
    p3 = out_dir / "3文档2D坐标.json"
    p1.write_text(json.dumps(stats_result, ensure_ascii=False, indent=2), encoding="utf-8")
    p2.write_text(json.dumps(topic_keywords, ensure_ascii=False, indent=2), encoding="utf-8")
    p3.write_text(json.dumps(doc_coords, ensure_ascii=False, indent=2), encoding="utf-8")

    return stats_result


def _default_classify_prompt(event_introduction: str) -> Dict[str, str]:
    intro = (event_introduction or "未提供事件背景").strip()[:4000]
    return {
        "system": (
            "你是舆情主题分析专家。结合事件背景，判断 BERTopic 聚类主题是否与该舆情事件相关，"
            "并为相关主题生成简洁中文命名与描述。"
        ),
        "user": (
            f"【事件背景】\n{intro}\n\n"
            "【聚类主题信息】\n{input_data}\n\n"
            "请输出 JSON，包含字段「主题分类结果」（数组，每项含：原始主题名称、是否与事件相关、"
            "主题命名、主题描述、判断理由）与「分析说明」。只输出 JSON。"
        ),
    }


def _recluster_prompt_yaml_path() -> Path:
    """大模型再聚类提示词唯一来源：config/大模型再聚类.yaml。"""
    return get_config_dir() / RECLUSTER_PROMPT_YAML


def _load_recluster_prompt_section(
    prompt_key: str,
    logger: Optional[logging.Logger] = None,
) -> Dict[str, str]:
    """从 config/大模型再聚类.yaml 加载指定 prompts 节点。"""
    prompt_file = _recluster_prompt_yaml_path()
    if not prompt_file.is_file():
        msg = f"未找到再聚类提示词文件: {prompt_file}"
        if logger:
            log_error(logger, msg, "TopicBertopic")
        raise FileNotFoundError(msg)
    try:
        with open(prompt_file, "r", encoding="utf-8") as f:
            prompt_config = yaml.safe_load(f) or {}
    except Exception as e:
        msg = f"读取再聚类提示词失败: {prompt_file} ({e})"
        if logger:
            log_error(logger, msg, "TopicBertopic")
        raise RuntimeError(msg) from e
    prompts = prompt_config.get("prompts")
    if not isinstance(prompts, dict):
        msg = f"提示词文件格式错误，缺少 prompts 字段: {prompt_file}"
        if logger:
            log_error(logger, msg, "TopicBertopic")
        raise ValueError(msg)
    section = prompts.get(prompt_key)
    if not isinstance(section, dict):
        msg = f"提示词文件中未找到 {prompt_key}: {prompt_file}"
        if logger:
            log_error(logger, msg, "TopicBertopic")
        raise KeyError(msg)
    system = str(section.get("system", "") or "").strip()
    user = str(section.get("user", "") or "").strip()
    if not system or not user:
        msg = f"提示词 {prompt_key} 的 system/user 不能为空: {prompt_file}"
        if logger:
            log_error(logger, msg, "TopicBertopic")
        raise ValueError(msg)
    return {"system": system, "user": user}


_GENERIC_TOPIC_LABEL_FRAGMENTS = frozenset(
    {
        "舆情响应",
        "情绪极化",
        "制作执行",
        "跨圈延伸",
        "核心录制",
        "同框实证",
        "信息传播",
        "议题迁移",
        "语义聚合",
        "模因传播",
        "话语场",
        "传播机制",
        "公共情绪",
        "风险联想",
        "内容生产",
        "噪声剥离",
        "弱语义",
        "非模因噪声",
        "情感共振",
        "议程迁移",
    }
)
_GENERIC_TOPIC_LABEL_ONLY = frozenset(
    {"舆论", "网友", "事件", "讨论", "热点", "争议", "舆情", "传播", "情绪", "话题"}
)


_TOPIC_LABEL_ACTION_WORDS = (
    "同框",
    "造型",
    "录制",
    "排名",
    "微指",
    "争议",
    "互动",
    "玩梗",
    "热搜",
    "回应",
    "联动",
    "超话",
    "开车",
    "话题",
)


def _keyword_texts(keywords: Optional[List[Any]], limit: int = 8) -> List[str]:
    out: List[str] = []
    for kw in keywords or []:
        if len(out) >= limit:
            break
        if isinstance(kw, (list, tuple)) and kw:
            text = str(kw[0]).strip()
        else:
            text = str(kw).strip()
        if len(text) >= 2 and text not in out:
            out.append(text)
    return out


def _normalize_topic_label(
    label: str,
    keywords: Optional[List[Any]] = None,
    *,
    max_len: int = MAX_TOPIC_LABEL_LEN,
    logger: Optional[logging.Logger] = None,
) -> str:
    """将主题命名压缩为不超过 max_len 字，保留核心对象与讨论点。"""
    raw = re.sub(r"[《》【】\[\]（）()]", "", str(label or "").strip())
    if not raw or raw == "无关主题":
        return raw
    if len(raw) <= max_len:
        return raw

    kw_texts = _keyword_texts(keywords, limit=8)

    for action in _TOPIC_LABEL_ACTION_WORDS:
        if action not in raw:
            continue
        for idx, entity in enumerate(kw_texts[:6]):
            if entity not in raw:
                continue
            for other in kw_texts[idx + 1 : 6]:
                if other in raw and other != entity:
                    cand = f"{entity}{other}{action}"
                    if len(cand) <= max_len:
                        if logger:
                            log_success(logger, f"主题命名已压缩：「{raw}」→「{cand}」", "TopicBertopic")
                        return cand
            cand = f"{entity}{action}"
            if len(cand) <= max_len:
                if logger:
                    log_success(logger, f"主题命名已压缩：「{raw}」→「{cand}」", "TopicBertopic")
                return cand

    for conn in ("及粉丝", "以及", "及平台", "及", "与", "、", "和"):
        if conn in raw:
            head = raw.split(conn, 1)[0].strip()
            if 4 <= len(head) <= max_len:
                if logger:
                    log_success(logger, f"主题命名已压缩：「{raw}」→「{head}」", "TopicBertopic")
                return head

    if len(kw_texts) >= 2:
        cand = f"{kw_texts[0]}{kw_texts[1]}"
        if len(cand) <= max_len:
            if logger:
                log_success(logger, f"主题命名已压缩：「{raw}」→「{cand}」", "TopicBertopic")
            return cand

    compact = raw[:max_len]
    if logger:
        log_success(logger, f"主题命名已截断：「{raw}」→「{compact}」", "TopicBertopic")
    return compact


def _collect_cluster_keyword_hints(input_data: Dict[str, Any], limit: int = 24) -> List[str]:
    """从各原始主题关键词中汇总高频词，供命名锚点提示。"""
    freq: Dict[str, int] = defaultdict(int)
    topics = input_data.get("主题信息") if isinstance(input_data.get("主题信息"), dict) else {}
    for info in topics.values():
        if not isinstance(info, dict):
            continue
        doc_count = max(1, int(info.get("文档数", 0) or 0))
        kws = info.get("关键词") or []
        if not isinstance(kws, list):
            continue
        for kw in kws[:12]:
            if isinstance(kw, (list, tuple)) and kw:
                text = str(kw[0]).strip()
            else:
                text = str(kw).strip()
            if len(text) < 2 or text.isdigit():
                continue
            freq[text] += doc_count
    ranked = sorted(freq.items(), key=lambda x: (-x[1], -len(x[0]), x[0]))
    return [w for w, _ in ranked[:limit]]


def _build_event_naming_hints(event_introduction: str, input_data: Dict[str, Any]) -> str:
    """生成注入再聚类 prompt 的命名锚点说明。"""
    intro = (event_introduction or "未提供事件背景").strip()
    intro_short = intro[:360] + ("..." if len(intro) > 360 else "")
    top_kws = _collect_cluster_keyword_hints(input_data, limit=20)
    kw_line = "、".join(top_kws[:14]) if top_kws else "（关键词不足，请从事件背景提取实体）"
    return (
        f"- 事件背景摘要：{intro_short}\n"
        f"- 数据侧高频词/实体（命名应优先从中选取，并与事件背景交叉验证）：{kw_line}\n"
        f"- 每个「主题命名」**不超过 {MAX_TOPIC_LABEL_LEN} 个字**，只保留 1 个核心讨论点（对象+动作/争议），勿堆砌多个子话题。\n"
        "- 每个主题名须写出「谁/什么 + 在讨论什么」，勿写纯分析框架标签。"
    )


def _is_generic_topic_label(label: str) -> bool:
    """判断主题命名是否过于抽象（用于日志告警）。"""
    text = str(label or "").strip()
    if not text or text in _GENERIC_TOPIC_LABEL_ONLY:
        return True
    if len(text) <= 4 and any(ch in text for ch in "舆情传播情绪讨论热点"):
        return True
    hits = sum(1 for frag in _GENERIC_TOPIC_LABEL_FRAGMENTS if frag in text)
    if hits >= 2:
        return True
    if hits >= 1 and not re.search(r"[\u4e00-\u9fff]{2,}(?:哥|姐|节目|品牌|公司|回应|造型|同框|代言|录制|通报|处罚)", text):
        return True
    return False


def _warn_generic_topic_labels(merge_plan: List[Any], logger: Optional[logging.Logger]) -> None:
    if not logger:
        return
    for group in merge_plan or []:
        if not isinstance(group, dict):
            continue
        naming = str(group.get("主题命名", "") or "").strip()
        if naming and _is_generic_topic_label(naming):
            log_error(
                logger,
                f"主题命名可能过于抽象，建议结合事件重写：「{naming}」",
                "TopicBertopic",
            )


def _format_recluster_main_user_prompt(
    user_template: str,
    *,
    event_introduction: str,
    input_data: Dict[str, Any],
    min_topics: List[str],
    total_topic_count: int,
    emphasize_min_topics: bool,
) -> str:
    """填充主再聚类 user 模板占位符。"""
    min_a = min_topics[0] if len(min_topics) > 0 else "主题0"
    min_b = min_topics[1] if len(min_topics) > 1 else min_a
    retry_reminder = ""
    if emphasize_min_topics:
        retry_reminder = (
            f"【重试提醒】上次输出议题数不足 {MIN_TOPIC_COUNT} 个，或遗漏了文档数最低的原始主题，"
            "或主题命名过于抽象/超过字数上限；请重新分析并满足全部硬性约束。\n\n"
        )
    naming_hints = _build_event_naming_hints(event_introduction, input_data)
    return user_template.format(
        retry_reminder=retry_reminder,
        event_introduction=(event_introduction or "未提供事件背景").strip()[:4000],
        naming_hints=naming_hints,
        TARGET_TOPICS=TARGET_TOPICS,
        MIN_TOPIC_COUNT=MIN_TOPIC_COUNT,
        input_data=json.dumps(input_data, ensure_ascii=False, indent=2),
        min_topic_low_doc_a=min_a,
        min_topic_low_doc_b=min_b,
        total_topic_count=total_topic_count,
    )

async def _call_llm_classify_topics(
    topic_stats: Dict,
    topic: str,
    logger: logging.Logger,
    *,
    event_introduction: str = "",
) -> Optional[Dict]:
    """调用大模型判断每个原始主题是否与事件相关，并给出命名。"""
    try:
        client = OpenAI(api_key=get_api_key(), base_url="https://dashscope.aliyuncs.com/compatible-mode/v1")
        
        # 控制输入规模：保留所有主题，每个主题前 10 个关键词
        input_data = {"主题信息": {}}
        max_keywords = 10
        for topic_key, topic_info in topic_stats["主题文档统计"].items():
            keywords = topic_stats["主题关键词"][topic_key]["关键词"][:max_keywords]
            input_data["主题信息"][topic_key] = {
                "文档数": topic_info["文档数"],
                "关键词": keywords
            }
        
        prompt_config = _load_prompt("topic_bertopic_classify.yaml", "topic_bertopic_classify", logger)
        if not prompt_config:
            prompt_config = _load_prompt(f"topic_bertopic/{topic}.yaml", "topic_bertopic_classify", logger)
        if not prompt_config:
            prompt_config = _default_classify_prompt(event_introduction)

        user_template = prompt_config["user"]
        if "{event_introduction}" in user_template:
            user_prompt = user_template.format(
                event_introduction=(event_introduction or "").strip()[:4000],
                input_data=json.dumps(input_data, ensure_ascii=False, indent=2),
            )
        else:
            user_prompt = user_template.format(
                input_data=json.dumps(input_data, ensure_ascii=False, indent=2)
            )
        max_prompt_len = 7000
        if len(user_prompt) > max_prompt_len:
            head = user_prompt[:4800]
            tail = user_prompt[-2000:]
            user_prompt = head + "\n\n[内容过长已截断；以下为输出格式与约束，请严格遵守]\n\n" + tail
        
        response = client.chat.completions.create(
            model=LLM_MODEL_NAME,
            messages=[
                {"role": "system", "content": prompt_config['system']},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.1,
            max_tokens=4000
        )
        
        result_text = response.choices[0].message.content.strip()
        
        try:
            classify_result = _extract_first_json_value(result_text)
            if isinstance(classify_result, dict) and "主题分类结果" in classify_result:
                log_success(logger, "大模型主题分类结果获取成功", "TopicBertopic")
                return classify_result
            raise ValueError("JSON结构不符合预期（缺少'主题分类结果'）")
        except Exception as e:
            log_error(logger, f"JSON解析失败: {e}", "TopicBertopic")
            log_error(logger, f"响应长度: {len(result_text)}, 前200字符: {result_text[:200]}", "TopicBertopic")
            
            try:
                repair_system = "你是一个严格的JSON转换器。你的输出必须是单个JSON对象，且不包含任何解释、Markdown、代码块标记。"
                repair_user = (
                    "请把下面内容转换为严格JSON，并满足以下schema：\n"
                    "{\n"
                    '  "主题分类结果": [\n'
                    "    {\n"
                    '      "原始主题名称": "主题0",\n'
                    '      "是否控烟相关": true,\n'
                    '      "主题命名": "6-8个字中文主题名",\n'
                    '      "主题描述": "...",\n'
                    '      "判断理由": "..."\n'
                    "    }\n"
                    "  ],\n"
                    '  "分析说明": "..."\n'
                    "}\n\n"
                    "注意：只输出JSON本体，必须以 { 开头，以 } 结尾。\n\n"
                    "需要转换的原始内容如下：\n"
                    f"{result_text}"
                )
                
                repair_resp = client.chat.completions.create(
                    model=LLM_MODEL_NAME,
                    messages=[
                        {"role": "system", "content": repair_system},
                        {"role": "user", "content": repair_user},
                    ],
                    temperature=0.0,
                    max_tokens=2000,
                )
                repaired_text = repair_resp.choices[0].message.content.strip()
                classify_result = _extract_first_json_value(repaired_text)
                if isinstance(classify_result, dict) and "主题分类结果" in classify_result:
                    log_success(logger, "大模型主题分类结果获取成功（repair JSON）", "TopicBertopic")
                    return classify_result
                log_error(logger, "repair返回JSON结构仍不符合预期", "TopicBertopic")
                return None
            except Exception as repair_e:
                log_error(logger, f"repair JSON 失败: {repair_e}", "TopicBertopic")
                return None
        
    except Exception as e:
        log_error(logger, f"大模型主题分类调用失败: {e}", "TopicBertopic")
        import traceback
        log_error(logger, f"完整堆栈: {traceback.format_exc()}", "TopicBertopic")
        return None


def _topic_sort_key_for_merge(name: str) -> Tuple:
    s = str(name).strip()
    m = re.match(r"^主题(\d+)$", s)
    if m:
        return (0, int(m.group(1)))
    m2 = re.match(r"^主题(-?\d+)$", s)
    if m2:
        return (0, int(m2.group(1)))
    return (1, s)


def _canonical_topic_id(raw, valid_topics: set) -> Optional[str]:
    """将模型输出的主题 id 规范为 valid_topics 中的键。"""
    if raw is None:
        return None
    s = str(raw).strip()
    if not s:
        return None
    if s in valid_topics:
        return s
    for pattern in (
        r"^主题\s*(\d+)$",
        r"^主题\s*(-?\d+)$",
        r"^[Tt]opic\s*(\d+)$",
        r"^#?(\d+)$",
    ):
        m = re.match(pattern, s)
        if m:
            cand = f"主题{int(m.group(1))}"
            if cand in valid_topics:
                return cand
    return None


def _canonical_ids_in_merge_group(group: Dict[str, Any], valid_topics: set) -> List[str]:
    """从合并组提取合法且去重后的原始主题 id 列表。"""
    ot = group.get("原始主题集合", [])
    if not isinstance(ot, list):
        return []
    out: List[str] = []
    seen: set = set()
    for raw in ot:
        cid = _canonical_topic_id(raw, valid_topics)
        if cid and cid not in seen:
            seen.add(cid)
            out.append(cid)
    return out


def _is_placeholder_merge_group(group: Dict[str, Any], valid_topics: set) -> bool:
    naming = str(group.get("主题命名", "") or "").strip()
    if any(frag in naming for frag in _PLACEHOLDER_TOPIC_NAME_FRAGMENTS):
        return True
    ot = group.get("原始主题集合", [])
    if not isinstance(ot, list) or len(ot) == 0:
        return True
    return len(_canonical_ids_in_merge_group(group, valid_topics)) == 0


def _is_llm_irrelevant_merge_group(group: Dict[str, Any]) -> bool:
    if bool(group.get("是否无关主题")):
        return True
    naming = str(group.get("主题命名", "") or "").strip()
    if naming in _IRRELEVANT_TOPIC_NAME_MARKERS:
        return True
    for marker in ("无关议题", "噪声内容", "噪声主题", "非模因噪声", "弱关联", "噪声集", "噪声主题集"):
        if marker in naming:
            return True
    return False


def _partition_merge_plan_for_pipeline(
    merge_plan: List[Any],
    valid_topics: set,
    logger: Optional[logging.Logger] = None,
) -> Tuple[List[Dict[str, Any]], set]:
    """
    拆分合并方案：事件相关组 / 待并入程序「无关主题」的低相关组；剔除占位与空组。
    """
    relevant: List[Dict[str, Any]] = []
    deferred: set = set()
    skipped_placeholders = 0
    skipped_irrelevant_groups = 0

    for group in merge_plan or []:
        if not isinstance(group, dict):
            continue
        topic_ids = _canonical_ids_in_merge_group(group, valid_topics)
        if _is_placeholder_merge_group(group, valid_topics) or not topic_ids:
            skipped_placeholders += 1
            continue
        if _is_llm_irrelevant_merge_group(group):
            deferred.update(topic_ids)
            skipped_irrelevant_groups += 1
            continue
        group_copy = dict(group)
        group_copy["原始主题集合"] = topic_ids
        relevant.append(group_copy)

    if logger and (skipped_placeholders or skipped_irrelevant_groups):
        log_success(
            logger,
            f"合并方案整理：有效议题{len(relevant)}个，低相关/噪声组{skipped_irrelevant_groups}个"
            f"（{len(deferred)}个原始主题将并入程序无关桶），跳过占位/空组{skipped_placeholders}个",
            "TopicBertopic",
        )
    return relevant, deferred


def _projected_topic_bucket_count(relevant_plan: List[Dict[str, Any]], deferred_topics: set) -> int:
    """预计最终报告可见的议题桶数（相关议题 + 可选的程序无关桶）。"""
    count = len(relevant_plan)
    if deferred_topics:
        count += 1
    return count


def _sanitize_merge_plan_partition(
    merge_result: Dict,
    valid_topics: set,
    logger: Optional[logging.Logger] = None,
) -> Dict:
    """
    程序兜底：剔除非法 id、去除重复分配、补齐遗漏主题，使合并方案成为合法划分。
    保留各组中首次出现的主题归属，遗漏主题归入「其他相关议题」组。
    """
    valid_sorted = sorted(valid_topics, key=_topic_sort_key_for_merge)
    assigned: set = set()
    plan_in = merge_result.get("合并方案", []) or []
    cleaned_plan: List[Dict[str, Any]] = []

    for group in plan_in:
        if not isinstance(group, dict):
            continue
        group_copy = dict(group)
        ot = group_copy.get("原始主题集合", [])
        if not isinstance(ot, list):
            ot = []
        new_ot: List[str] = []
        for raw in ot:
            cid = _canonical_topic_id(raw, valid_topics)
            if cid is None or cid in assigned:
                continue
            assigned.add(cid)
            new_ot.append(cid)
        group_copy["原始主题集合"] = new_ot
        if new_ot:
            cleaned_plan.append(group_copy)

    missing = [t for t in valid_sorted if t not in assigned]
    if missing:
        cleaned_plan.append(
            {
                "新主题名称": f"新主题{len(cleaned_plan)}",
                "原始主题集合": missing,
                "主题命名": "其他相关议题",
                "主题描述": "程序补全：模型合并方案中遗漏或重复剔除后未覆盖的原始主题。",
                "合并理由": "分区校验程序兜底",
                "是否无关主题": False,
                "合并后关键词": [],
            }
        )

    out = dict(merge_result)
    out["合并方案"] = cleaned_plan
    if logger:
        log_success(
            logger,
            f"已对合并方案做程序分区兜底：{len(cleaned_plan)} 组，覆盖 {len(valid_topics)} 个原始主题",
            "TopicBertopic",
        )
    return out


def _validate_merge_plan_partition(
    merge_plan: list, valid_topics: set, logger
) -> Tuple[bool, Dict]:
    """
    硬校验：每个合法原始主题恰好出现在某一个「原始主题集合」中；
    集合内 id 必须属于 valid_topics；无重复、无遗漏、无非法 id。
    """
    seen_count: Dict[str, int] = defaultdict(int)
    invalid_ids: List[str] = []
    group_hits: Dict[str, List[int]] = defaultdict(list)

    for gi, group in enumerate(merge_plan or []):
        if not isinstance(group, dict):
            continue
        ot = group.get("原始主题集合", [])
        if not isinstance(ot, list):
            continue
        for raw in ot:
            cid = _canonical_topic_id(raw, valid_topics)
            if cid is None:
                invalid_ids.append(str(raw))
            else:
                seen_count[cid] += 1
                group_hits[cid].append(gi)

    duplicates = sorted([t for t, c in seen_count.items() if c > 1], key=_topic_sort_key_for_merge)
    covered = set(seen_count.keys())
    missing = sorted(valid_topics - covered, key=_topic_sort_key_for_merge)
    invalid_ids = sorted(set(invalid_ids), key=lambda x: (len(str(x)), str(x)))

    dup_detail = {t: group_hits[t] for t in duplicates}
    ok = (
        not invalid_ids
        and not duplicates
        and not missing
        and len(covered) == len(valid_topics)
    )
    report = {
        "invalid_ids": invalid_ids,
        "duplicates": duplicates,
        "duplicate_group_indices": dup_detail,
        "missing": missing,
        "covered_count": len(covered),
        "expected_count": len(valid_topics),
    }
    if not ok and logger:
        log_error(logger, f"合并方案分区校验失败: missing={len(missing)} dup={len(duplicates)} invalid={len(invalid_ids)}", "TopicBertopic")
    return ok, report


def _repair_merge_plan_partition_call(
    client: OpenAI,
    logger: logging.Logger,
    merge_result: Dict,
    report: Dict,
    input_data: Dict,
    min_topics: List[str],
    valid_topics_sorted: List[str],
    *,
    event_introduction: str = "",
) -> Optional[Dict]:
    """校验失败时：二次调用模型重写合并方案（完整分区）。"""
    prev_plan = merge_result.get("合并方案", [])
    try:
        prompt_config = _load_recluster_prompt_section(RECLUSTER_REPAIR_PARTITION_KEY, logger)
    except (FileNotFoundError, KeyError, ValueError, RuntimeError) as e:
        log_error(logger, f"加载分区 repair 提示词失败: {e}", "TopicBertopic")
        return None
    repair_user = prompt_config["user"].format(
        total_topic_count=len(valid_topics_sorted),
        valid_topics_json=json.dumps(valid_topics_sorted, ensure_ascii=False),
        validation_report_json=json.dumps(report, ensure_ascii=False, indent=2),
        min_topics_json=json.dumps(min_topics, ensure_ascii=False),
        input_data_json=json.dumps(input_data, ensure_ascii=False, indent=2)[:6500],
        prev_plan_json=json.dumps(prev_plan, ensure_ascii=False)[:4000],
        MIN_TOPIC_COUNT=MIN_TOPIC_COUNT,
        event_introduction=(event_introduction or "未提供事件背景").strip()[:2000],
        naming_hints=_build_event_naming_hints(event_introduction, input_data),
    )
    try:
        repair_resp = client.chat.completions.create(
            model=LLM_MODEL_NAME,
            messages=[
                {"role": "system", "content": prompt_config["system"]},
                {"role": "user", "content": repair_user},
            ],
            temperature=0.0,
            max_tokens=8000,
        )
        repaired_text = repair_resp.choices[0].message.content.strip()
        fixed = _extract_merge_plan_json(repaired_text)
        if isinstance(fixed, dict) and "合并方案" in fixed:
            log_success(logger, "合并方案分区 repair 调用成功，已重新解析 JSON", "TopicBertopic")
            return fixed
        log_error(logger, "分区 repair 返回 JSON 缺少「合并方案」", "TopicBertopic")
        return None
    except Exception as e:
        log_error(logger, f"分区 repair 调用失败: {e}", "TopicBertopic")
        return None


def _ensure_merge_plan_partition(
    merge_result: Dict,
    valid_topics: set,
    client: OpenAI,
    logger,
    input_data: Dict,
    min_topics: List[str],
    max_repairs: int = 2,
    *,
    event_introduction: str = "",
) -> Optional[Dict]:
    """若分区不合法，自动触发 repair；仍失败时用程序兜底去重/补全。"""
    valid_sorted = sorted(valid_topics, key=_topic_sort_key_for_merge)
    current = merge_result

    def _try_sanitize(candidate: Dict) -> Optional[Dict]:
        sanitized = _sanitize_merge_plan_partition(candidate, valid_topics, logger)
        ok, _report = _validate_merge_plan_partition(
            sanitized.get("合并方案", []), valid_topics, None
        )
        if ok:
            log_success(logger, "合并方案分区校验通过（程序兜底去重/补全）", "TopicBertopic")
            _warn_generic_topic_labels(sanitized.get("合并方案", []), logger)
            return sanitized
        return None

    for attempt in range(max_repairs + 1):
        plan = current.get("合并方案", [])
        ok, report = _validate_merge_plan_partition(plan, valid_topics, logger)
        if ok:
            if attempt > 0:
                log_success(logger, f"合并方案分区校验通过（经 {attempt} 次 repair）", "TopicBertopic")
            _warn_generic_topic_labels(current.get("合并方案", []), logger)
            return current
        if attempt >= max_repairs:
            log_error(logger, f"合并方案分区校验仍失败，已用尽 repair 次数({max_repairs})", "TopicBertopic")
            return _try_sanitize(current)
        log_error(logger, f"触发合并方案分区 repair，第 {attempt + 1}/{max_repairs} 次", "TopicBertopic")
        repaired = _repair_merge_plan_partition_call(
            client,
            logger,
            current,
            report,
            input_data,
            min_topics,
            valid_sorted,
            event_introduction=event_introduction,
        )
        if not repaired:
            fallback = _try_sanitize(current)
            if fallback:
                return fallback
            return None
        current = repaired
    return None


async def _call_llm_recluster(
    topic_stats: Dict,
    topic: str,
    logger: logging.Logger,
    emphasize_min_topics: bool = False,
    *,
    event_introduction: str = "",
) -> Optional[Dict]:
    """调用大模型进行主题合并。

    Args:
        topic_stats: 主题统计信息
        topic: 专题/领域标识（用于加载可选 YAML 提示词）
        logger: 日志记录器
        emphasize_min_topics: 是否强调必须生成至少 MIN_TOPIC_COUNT 个主题（用于重试时）
        event_introduction: 事件背景（舆情流水线传入）
    """
    try:
        client = OpenAI(api_key=get_api_key(), base_url="https://dashscope.aliyuncs.com/compatible-mode/v1")
        
        # 控制输入规模：保留所有主题，每个主题前 10 个关键词
        input_data = {"主题信息": {}}
        max_keywords = 10
        all_topic_keys = []
        topic_doc_counts = []  # 用于找出文档数最低的两个主题
        
        for topic_key, topic_info in topic_stats["主题文档统计"].items():
            all_topic_keys.append(topic_key)
            doc_count = topic_info["文档数"]
            topic_doc_counts.append((topic_key, doc_count))
            keywords_raw = topic_stats["主题关键词"][topic_key]["关键词"][:max_keywords]
            # 将关键词从嵌套数组格式转换为易读格式：提取关键词列表
            keywords_list = [kw[0] for kw in keywords_raw]  # 只提取关键词文本
            
            input_data["主题信息"][topic_key] = {
                "文档数": doc_count,
                "关键词": keywords_list  # 使用关键词列表，更易读
            }
        
        # 找出文档数最低的两个主题（主题0和主题1）
        topic_doc_counts.sort(key=lambda x: x[1])  # 按文档数升序排序
        min_topics = [t[0] for t in topic_doc_counts[:2]]  # 取前两个
        valid_topics = set(topic_stats["主题文档统计"].keys())
        log_success(logger, f"文档数最低的两个主题: {min_topics[0]}({topic_doc_counts[0][1]}文档), {min_topics[1]}({topic_doc_counts[1][1]}文档)", "TopicBertopic")
        
        # 记录输入数据统计，用于调试
        log_success(logger, f"准备输入给LLM的主题数量: {len(all_topic_keys)}", "TopicBertopic")
        log_success(logger, f"主题列表: {', '.join(all_topic_keys[:10])}{'...' if len(all_topic_keys) > 10 else ''}", "TopicBertopic")

        try:
            prompt_config = _load_recluster_prompt_section(RECLUSTER_PROMPT_KEY, logger)
        except (FileNotFoundError, KeyError, ValueError, RuntimeError) as e:
            log_error(logger, f"加载再聚类提示词失败: {e}", "TopicBertopic")
            return None

        user_prompt = _format_recluster_main_user_prompt(
            prompt_config["user"],
            event_introduction=event_introduction,
            input_data=input_data,
            min_topics=min_topics,
            total_topic_count=len(all_topic_keys),
            emphasize_min_topics=emphasize_min_topics,
        )
        # 进一步收紧，留出 system+元数据空间，避免 8192 上限
        max_prompt_len = 7000
        if len(user_prompt) > max_prompt_len:
            # 不能简单从尾部截断，否则可能截掉“只输出JSON”约束与 schema 示例，导致模型更容易输出 Markdown
            head = user_prompt[:4800]
            tail = user_prompt[-2000:]
            user_prompt = head + "\n\n[内容过长已截断；以下为输出格式与约束，请严格遵守]\n\n" + tail
        
        response = client.chat.completions.create(
            model=LLM_MODEL_NAME,
            messages=[
                {"role": "system", "content": prompt_config['system']},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.1,
            max_tokens=8192,
        )
        
        result_text = response.choices[0].message.content.strip()
        
        try:
            merge_result = _extract_merge_plan_json(result_text)
            if isinstance(merge_result, dict) and "合并方案" in merge_result:
                merge_plan = merge_result.get("合并方案", [])
                log_success(logger, f"大模型合并建议获取成功，返回{len(merge_plan)}个合并方案", "TopicBertopic")

                merged_ok = _ensure_merge_plan_partition(
                    merge_result,
                    valid_topics,
                    client,
                    logger,
                    input_data,
                    min_topics,
                    event_introduction=event_introduction,
                )
                if not merged_ok:
                    return None
                merge_result = merged_ok
                merge_plan = merge_result.get("合并方案", [])

                required_topics = set(min_topics)
                all_canonical = set()
                for group in merge_plan:
                    if not isinstance(group, dict):
                        continue
                    ot = group.get("原始主题集合", [])
                    if not isinstance(ot, list):
                        continue
                    for raw in ot:
                        cid = _canonical_topic_id(raw, valid_topics)
                        if cid:
                            all_canonical.add(cid)
                if required_topics - all_canonical:
                    log_error(
                        logger,
                        f"分区合法但文档数最低主题未全部出现: {sorted(required_topics - all_canonical)}",
                        "TopicBertopic",
                    )

                for idx, group in enumerate(merge_plan):
                    if isinstance(group, dict):
                        original_topics = group.get("原始主题集合", [])
                        topic_naming = group.get("主题命名", "")
                        log_success(logger, f"合并方案{idx}: {topic_naming} <- {len(original_topics) if isinstance(original_topics, list) else 0}个原始主题", "TopicBertopic")

                return merge_result
            # 解析出了 JSON，但结构不符合预期（多为截断后误解析到内层对象）
            parsed_type = type(merge_result).__name__
            parsed_keys = list(merge_result.keys())[:8] if isinstance(merge_result, dict) else []
            raise ValueError(
                f"JSON结构不符合预期（缺少'合并方案'），解析类型={parsed_type}，顶层键={parsed_keys}"
            )
        except Exception as e:
            # 常见原因：输出被截断、Markdown 包裹、或误解析内层对象
            log_error(logger, f"JSON解析失败: {e}", "TopicBertopic")
            log_error(logger, f"响应长度: {len(result_text)}, 前200字符: {result_text[:200]}", "TopicBertopic")
            if len(result_text) > 200:
                log_error(logger, f"响应末尾200字符: {result_text[-200:]}", "TopicBertopic")

            try:
                repair_prompt = _load_recluster_prompt_section(RECLUSTER_REPAIR_JSON_KEY, logger)
            except (FileNotFoundError, KeyError, ValueError, RuntimeError) as load_err:
                log_error(logger, f"加载 JSON repair 提示词失败: {load_err}", "TopicBertopic")
                return None

            try:
                repair_user = repair_prompt["user"].format(raw_model_output=result_text)
                repair_resp = client.chat.completions.create(
                    model=LLM_MODEL_NAME,
                    messages=[
                        {"role": "system", "content": repair_prompt["system"]},
                        {"role": "user", "content": repair_user},
                    ],
                    temperature=0.0,
                    max_tokens=8192,
                )
                repaired_text = repair_resp.choices[0].message.content.strip()
                merge_result = _extract_merge_plan_json(repaired_text)
                if isinstance(merge_result, dict) and "合并方案" in merge_result:
                    log_success(logger, "大模型合并建议获取成功（repair JSON）", "TopicBertopic")
                    merged_ok = _ensure_merge_plan_partition(
                        merge_result,
                        valid_topics,
                        client,
                        logger,
                        input_data,
                        min_topics,
                        event_introduction=event_introduction,
                    )
                    return merged_ok
                log_error(logger, "repair返回JSON结构仍不符合预期", "TopicBertopic")
                return None
            except Exception as repair_e:
                log_error(logger, f"repair JSON 失败: {repair_e}", "TopicBertopic")
                return None
        
    except Exception as e:
        log_error(logger, f"大模型再聚类调用失败: {e}", "TopicBertopic")
        import traceback
        log_error(logger, f"完整堆栈: {traceback.format_exc()}", "TopicBertopic")
        return None


def _calculate_reclustered_keywords(topic_stats: Dict, merge_result: Dict, naming_results: Dict, logger=None) -> Dict:
    """根据合并方案重新计算关键词权重"""
    reclustered_topics = {}
    
    for merge_group in merge_result["合并方案"]:
        new_topic_name = merge_group.get("新主题名称", f"新主题{len(reclustered_topics)}")
        topic_naming = naming_results.get(new_topic_name, merge_group.get("主题命名", f"主题{len(reclustered_topics)}"))
        original_topics = merge_group.get("原始主题集合", [])
        topic_description = merge_group.get("主题描述", "")
        
        # 确保original_topics是列表
        if not isinstance(original_topics, list):
            if logger:
                log_error(logger, f"警告：合并方案中的'原始主题集合'不是列表格式: {original_topics}", "TopicBertopic")
            continue
        
        if not re.match(r'^新主题\d+$', new_topic_name):
            new_topic_name = f"新主题{len(reclustered_topics)}"
        
        all_doc_ids = []
        keyword_weights = defaultdict(float)
        total_original_docs = 0
        matched_topics = []
        unmatched_topics = []
        
        for original_topic in original_topics:
            if original_topic in topic_stats["主题文档统计"]:
                matched_topics.append(original_topic)
                doc_count = topic_stats["主题文档统计"][original_topic]["文档数"]
                doc_ids = topic_stats["主题文档统计"][original_topic]["文档ID"]
                keywords = topic_stats["主题关键词"][original_topic]["关键词"]
                
                all_doc_ids.extend(doc_ids)
                total_original_docs += doc_count
                
                for keyword, weight in keywords:
                    keyword_weights[keyword] += weight * doc_count
            else:
                unmatched_topics.append(original_topic)
        
        # 记录匹配情况
        if logger:
            if unmatched_topics:
                # 获取实际存在的主题范围，帮助调试
                all_actual_topics = list(topic_stats["主题文档统计"].keys())
                if all_actual_topics:
                    # 提取主题编号
                    topic_numbers = []
                    for t in all_actual_topics:
                        match = re.match(r'主题(\d+)', t)
                        if match:
                            topic_numbers.append(int(match.group(1)))
                    if topic_numbers:
                        max_topic_num = max(topic_numbers)
                        min_topic_num = min(topic_numbers)
                        log_error(logger, f"警告：以下主题在合并方案中但不在原始主题统计中: {', '.join(unmatched_topics)}。实际主题范围：主题{min_topic_num} 到 主题{max_topic_num}（共{len(all_actual_topics)}个主题）", "TopicBertopic")
                    else:
                        log_error(logger, f"警告：以下主题在合并方案中但不在原始主题统计中: {', '.join(unmatched_topics)}。实际主题列表: {', '.join(all_actual_topics[:10])}{'...' if len(all_actual_topics) > 10 else ''}", "TopicBertopic")
                else:
                    log_error(logger, f"警告：以下主题在合并方案中但不在原始主题统计中: {', '.join(unmatched_topics)}", "TopicBertopic")
            if matched_topics:
                log_success(logger, f"合并方案'{topic_naming}'成功匹配{len(matched_topics)}个原始主题: {', '.join(matched_topics[:5])}{'...' if len(matched_topics) > 5 else ''}", "TopicBertopic")
        
        new_doc_count = len(all_doc_ids)
        if new_doc_count == 0:
            continue
        
        recalculated_keywords = []
        for keyword, total_weight in keyword_weights.items():
            if total_original_docs > 0:
                new_weight = (total_weight / total_original_docs) * (total_original_docs / new_doc_count)
                recalculated_keywords.append([keyword, new_weight])
        
        recalculated_keywords.sort(key=lambda x: x[1], reverse=True)
        top_20_keywords = recalculated_keywords[:20]
        merge_kw = merge_group.get("合并后关键词") or []
        topic_naming = _normalize_topic_label(
            str(topic_naming or ""),
            merge_kw or top_20_keywords,
            logger=logger,
        )

        reclustered_topics[new_topic_name] = {
            "主题命名": topic_naming,
            "原始主题集合": original_topics,
            "文档数": new_doc_count,
            "主题描述": topic_description,
            "文档ID": all_doc_ids,
            "关键词": top_20_keywords
        }
    
    return reclustered_topics


async def _generate_reclustered_json(
    topic_stats: Dict,
    topic: str,
    out_dir: Path,
    logger: logging.Logger,
    *,
    df: Optional[pd.DataFrame] = None,
    text_col: Optional[str] = None,
    doc_id_col: Optional[str] = None,
    ct_to_seg_idx: Optional[Dict[str, int]] = None,
    event_introduction: str = "",
) -> Optional[Dict]:
    """生成大模型再聚类结果JSON"""
    n_raw_topics = len(topic_stats.get("主题文档统计") or {})
    min_topic_required = min(n_raw_topics, max(MIN_TOPIC_COUNT, 1))
    MAX_RETRIES = 3  # 最多重试3次
    log_success(
        logger,
        f"开始调用大模型对所有主题进行合并，目标生成至少{min_topic_required}个主题（建议{TARGET_TOPICS}个左右）",
        "TopicBertopic",
    )
    
    merge_result = None
    filtered_merge_plan: List[Dict[str, Any]] = []
    deferred_irrelevant_topics: set = set()
    
    # 重试逻辑：如果主题数不足，重新调用 LLM
    for retry_count in range(MAX_RETRIES):
        if retry_count > 0:
            log_success(logger, f"主题数量不足，进行第{retry_count}次重试（共{MAX_RETRIES}次）", "TopicBertopic")

        # 如果是重试，强调必须生成至少 MIN_TOPIC_COUNT 个主题
        emphasize = retry_count > 0
        merge_result = await _call_llm_recluster(
            topic_stats,
            topic,
            logger,
            emphasize_min_topics=emphasize,
            event_introduction=event_introduction,
        )

        # 如果本次调用没有返回结果
        if not merge_result:
            log_error(logger, "无法获取大模型合并建议", "TopicBertopic")
            if retry_count < MAX_RETRIES - 1:
                # 还有重试机会，继续下一轮
                continue
            # 没有更多重试机会，直接失败
            return None
    
        valid_topics = set(topic_stats.get("主题文档统计") or {})
        merge_plan = merge_result.get("合并方案", [])
        filtered_merge_plan, deferred_irrelevant_topics = _partition_merge_plan_for_pipeline(
            merge_plan, valid_topics, logger
        )
        projected_buckets = _projected_topic_bucket_count(filtered_merge_plan, deferred_irrelevant_topics)

        if len(filtered_merge_plan) == 0 and not deferred_irrelevant_topics:
            log_error(logger, "合并结果中没有有效主题", "TopicBertopic")
            if retry_count < MAX_RETRIES - 1:
                continue
            return None

        if projected_buckets >= min_topic_required:
            log_success(
                logger,
                f"议题桶数量达标：事件相关{len(filtered_merge_plan)}个"
                f"+低相关并入无关{1 if deferred_irrelevant_topics else 0}个"
                f"=合计{projected_buckets}个（要求≥{min_topic_required}）",
                "TopicBertopic",
            )
            break

        log_error(
            logger,
            f"合并结果预计仅{projected_buckets}个议题桶（事件相关{len(filtered_merge_plan)}个），"
            f"少于最低要求{min_topic_required}个",
            "TopicBertopic",
        )
        if retry_count < MAX_RETRIES - 1:
            log_success(
                logger,
                f"将在提示词中更强调必须生成至少{min_topic_required}个事件相关议题",
                "TopicBertopic",
            )
            continue
        log_error(
            logger,
            f"重试{MAX_RETRIES}次后议题桶仍不足，当前预计{projected_buckets}个，将使用当前结果",
            "TopicBertopic",
        )
        break

    if not merge_result:
        return None

    projected_buckets = _projected_topic_bucket_count(filtered_merge_plan, deferred_irrelevant_topics)
    if projected_buckets < min_topic_required:
        log_error(
            logger,
            f"警告：预计议题桶{projected_buckets}个（相关{len(filtered_merge_plan)}个），"
            f"少于最低要求{min_topic_required}个",
            "TopicBertopic",
        )
    else:
        log_success(
            logger,
            f"最终采用{len(filtered_merge_plan)}个事件相关议题"
            f"{'' if not deferred_irrelevant_topics else ' + 1个程序无关桶'}，"
            f"合计{projected_buckets}个议题桶",
            "TopicBertopic",
        )

    merge_result["合并方案"] = filtered_merge_plan
    merge_result["_deferred_irrelevant_topics"] = sorted(deferred_irrelevant_topics)
    log_success(
        logger,
        f"最终生成{len(filtered_merge_plan)}个事件相关合并方案"
        f"{f'，{len(deferred_irrelevant_topics)}个原始主题待并入无关桶' if deferred_irrelevant_topics else ''}",
        "TopicBertopic",
    )
    
    # ========== 根据最终合并方案计算再聚类关键词 ==========
    naming_results = {}
    for i, merge_group in enumerate(merge_result["合并方案"]):
        new_topic_name = merge_group.get("新主题名称", f"新主题{i}")
        topic_naming = merge_group.get("主题命名", "")
        if topic_naming:
            naming_results[new_topic_name] = topic_naming
    
    reclustered_topics = _calculate_reclustered_keywords(topic_stats, merge_result, naming_results, logger)
    
    # ========== 检查未包含的主题，合并为"无关主题" ==========
    all_original_topics = set(topic_stats["主题文档统计"].keys())
    included_original_topics = set()
    
    # 收集所有已包含在已定好主题中的原始主题
    for topic_name, topic_info in reclustered_topics.items():
        original_topics = topic_info.get("原始主题集合", [])
        if isinstance(original_topics, list):
            included_original_topics.update(original_topics)
    
    # 找出未包含的原始主题（含 LLM 低相关组中已标记、待并入无关桶的主题）
    excluded_original_topics = all_original_topics - included_original_topics
    pre_deferred = set(merge_result.get("_deferred_irrelevant_topics") or [])
    if pre_deferred:
        excluded_original_topics |= pre_deferred
        log_success(
            logger,
            f"LLM低相关/噪声组中{len(pre_deferred)}个原始主题将并入程序「无关主题」",
            "TopicBertopic",
        )

    # 如果有未包含的主题，将它们合并为一个"无关主题"
    if excluded_original_topics:
        log_success(logger, f"发现{len(excluded_original_topics)}个未包含在已定好主题中的原始主题，将合并为'无关主题'", "TopicBertopic")
        log_success(logger, f"未包含的主题: {', '.join(sorted(excluded_original_topics)[:10])}{'...' if len(excluded_original_topics) > 10 else ''}", "TopicBertopic")
        
        # 计算"无关主题"的文档和关键词
        irrelevant_doc_ids = []
        irrelevant_keyword_weights = defaultdict(float)
        irrelevant_total_docs = 0
        
        for excluded_topic in excluded_original_topics:
            if excluded_topic in topic_stats["主题文档统计"]:
                doc_count = topic_stats["主题文档统计"][excluded_topic]["文档数"]
                doc_ids = topic_stats["主题文档统计"][excluded_topic]["文档ID"]
                keywords = topic_stats["主题关键词"][excluded_topic]["关键词"]
                
                irrelevant_doc_ids.extend(doc_ids)
                irrelevant_total_docs += doc_count
                
                for keyword, weight in keywords:
                    irrelevant_keyword_weights[keyword] += weight * doc_count
        
        if len(irrelevant_doc_ids) > 0:
            # 计算关键词权重
            irrelevant_recalculated_keywords = []
            for keyword, total_weight in irrelevant_keyword_weights.items():
                if irrelevant_total_docs > 0:
                    new_weight = (total_weight / irrelevant_total_docs) * (irrelevant_total_docs / len(irrelevant_doc_ids))
                    irrelevant_recalculated_keywords.append([keyword, new_weight])
            
            irrelevant_recalculated_keywords.sort(key=lambda x: x[1], reverse=True)
            irrelevant_top_20_keywords = irrelevant_recalculated_keywords[:20]
            
            # 创建"无关主题"
            irrelevant_topic_name = f"新主题{len(reclustered_topics)}"
            reclustered_topics[irrelevant_topic_name] = {
                "主题命名": "无关主题",
                "原始主题集合": sorted(list(excluded_original_topics)),
                "文档数": len(irrelevant_doc_ids),
                "主题描述": (
                    f"此主题包含{len(excluded_original_topics)}个未被归类到已定好议题中的原始主题，"
                    "与当前事件相关度较低或属于噪声内容。"
                ),
                "文档ID": irrelevant_doc_ids,
                "关键词": irrelevant_top_20_keywords
            }
            
            log_success(logger, f"已创建'无关主题'，包含{len(excluded_original_topics)}个原始主题，{len(irrelevant_doc_ids)}个文档", "TopicBertopic")
    else:
        log_success(logger, "所有原始主题都已包含在已定好的主题中，无需创建'无关主题'", "TopicBertopic")
    
    final_result = {}
    reclustered_keywords_data = {}
    
    for topic_name, topic_info in reclustered_topics.items():
        if topic_info["文档数"] == 0:
            continue
        
        final_result[topic_name] = {
            "主题命名": topic_info["主题命名"],
            "原始主题集合": topic_info["原始主题集合"],
            "文档数": topic_info["文档数"],
            "主题描述": topic_info["主题描述"],
            "文档ID": topic_info["文档ID"],
            "关键词": topic_info["关键词"]
        }
        
        reclustered_keywords_data[topic_name] = {
            "主题命名": topic_info["主题命名"],
            "关键词": topic_info["关键词"]
        }
    
    # 保存JSON文件
    p4 = out_dir / "4大模型再聚类结果.json"
    p5 = out_dir / "5大模型主题关键词.json"
    p4.write_text(json.dumps(final_result, ensure_ascii=False, indent=2), encoding="utf-8")
    p5.write_text(json.dumps(reclustered_keywords_data, ensure_ascii=False, indent=2), encoding="utf-8")

    if df is not None and text_col and ct_to_seg_idx is not None:
        try:
            _write_document_topic_table_csv(
                out_dir, final_result, topic_stats, df, text_col, doc_id_col, ct_to_seg_idx, logger
            )
        except Exception as e:
            log_error(logger, f"写入 6文档主题归属表.csv 失败: {e}", "TopicBertopic")
    else:
        log_success(logger, "跳过 6文档主题归属表.csv（未传入 DataFrame / 文本列 / 清洗对齐映射）", "TopicBertopic")

    return final_result


def _load_prompt(file_path: str, prompt_key: str, logger: logging.Logger) -> Optional[Dict[str, str]]:
    """加载提示词配置（config/prompt、prompt/、项目根）。"""
    try:
        project_root = get_project_root()
        candidates = [
            get_config_dir() / file_path,
            get_config_dir() / "prompt" / file_path,
            get_prompt_dir() / file_path,
            project_root / "configs" / "prompt" / file_path,
            project_root / file_path,
            project_root / Path(file_path).name,
        ]
        prompt_file = None
        for c in candidates:
            if c.exists():
                prompt_file = c
                break
        if prompt_file is None:
            # 专用 topic yaml 常不存在，后续仍会回退到没出息/控烟等通用模板，不必用 ERROR
            log_success(
                logger,
                f"提示词文件不存在，尝试下一回退路径: {candidates[0]}",
                "TopicBertopic",
            )
            return None
        
        with open(prompt_file, 'r', encoding='utf-8') as f:
            prompt_config = yaml.safe_load(f)
        
        if 'prompts' not in prompt_config:
            log_error(logger, "提示词文件格式错误，缺少prompts字段", "TopicBertopic")
            return None
        
        prompts = prompt_config['prompts']
        if prompt_key not in prompts:
            log_error(logger, f"未找到{prompt_key}的提示词配置", "TopicBertopic")
            return None
        
        return prompts[prompt_key]
        
    except Exception as e:
        log_error(logger, f"加载提示词失败: {e}", "TopicBertopic")
        return None


def _row_cleaned_content(row: Dict[str, Any], content_columns: List[str]) -> str:
  parts: List[str] = []
  for col in content_columns:
    val = row.get(col)
    if val is None:
      continue
    s = str(val).strip()
    if s:
      parts.append(s)
  return clean_text_like_keyword_stats("\n".join(parts))


def _build_dataframe_from_csv_rows(
    rows: List[Dict[str, Any]],
    content_columns: List[str],
    *,
    max_rows: int,
) -> Tuple[pd.DataFrame, List[str], Dict[str, Any]]:
  """将 CSV 行转为 DataFrame，并写入 ``_sona_bertopic_text`` 列供建模。"""
  meta: Dict[str, Any] = {"total_rows": len(rows), "max_rows_cap": max_rows}
  if not rows:
    return pd.DataFrame(), [], meta

  capped = rows[:max_rows] if len(rows) > max_rows else rows
  meta["used_rows"] = len(capped)
  meta["truncated"] = len(rows) > len(capped)

  records: List[Dict[str, Any]] = []
  for i, row in enumerate(capped):
    cleaned = _row_cleaned_content(row, content_columns)
    rec = dict(row)
    rec["_sona_row_index"] = i
    rec["_sona_bertopic_text"] = cleaned
    records.append(rec)

  df = pd.DataFrame(records)
  return df, content_columns, meta


def _generate_topic_result_filename(retry_context: Optional[str] = None) -> str:
  timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
  base_name = f"topic_bertopic_{timestamp}"
  if not retry_context:
    return f"{base_name}.json"
  task_id = get_task_id()
  if not task_id:
    return f"{base_name}.json"
  process_dir = get_task_process_dir(task_id)
  if not process_dir.exists():
    return f"{base_name}_1.json"
  existing = list(process_dir.glob("topic_bertopic_*.json"))
  suffix_nums: List[int] = []
  for file in existing:
    match = re.search(r"topic_bertopic_\d{8}_\d{6}_(\d+)\.json", file.name)
    if match:
      suffix_nums.append(int(match.group(1)))
  if not suffix_nums:
    return f"{base_name}_1.json"
  return f"{base_name}_{max(suffix_nums) + 1}.json"


def _format_topics_for_sona(final_result: Dict[str, Any]) -> List[Dict[str, Any]]:
  topics: List[Dict[str, Any]] = []
  for topic_key, info in final_result.items():
    if not isinstance(info, dict):
      continue
    keywords_raw = info.get("关键词") or []
    keywords = [str(kw[0]) for kw in keywords_raw[:15] if isinstance(kw, (list, tuple)) and kw]
    label = _normalize_topic_label(str(info.get("主题命名", "") or ""), keywords)
    topics.append(
      {
        "topic_id": str(topic_key),
        "label": label,
        "description": str(info.get("主题描述", "") or ""),
        "doc_count": int(info.get("文档数", 0) or 0),
        "keywords": keywords,
        "original_cluster_ids": list(info.get("原始主题集合") or []),
      }
    )
  topics.sort(key=lambda x: x.get("doc_count", 0), reverse=True)
  return topics


def _run_bertopic_core(
    *,
    df: pd.DataFrame,
    text_col: str,
    out_dir: Path,
    logger: logging.Logger,
    event_introduction: str,
    domain_topic: str,
    stopwords_path: Path,
    userdict_path: Optional[Path] = None,
) -> Tuple[bool, Dict[str, Any]]:
  """执行 BERTopic + 大模型合并，返回 (成功与否, 结果载荷)。"""
  payload: Dict[str, Any] = {
    "topics": [],
    "statistics": {},
    "artifacts": {},
    "error": "",
  }
  if df.empty:
    payload["error"] = "数据为空"
    return False, payload

  out_dir.mkdir(parents=True, exist_ok=True)
  doc_id_col = _pick_document_id_column(df)

  source_indices: List[Any] = []
  texts: List[str] = []
  for idx in df.index:
    x = str(df.at[idx, text_col])
    if x.strip() and x.lower() not in ("nan", "none", ""):
      source_indices.append(idx)
      texts.append(x)

  if not texts:
    payload["error"] = "无有效文本可供聚类"
    return False, payload

  cleaned, _, ct_to_seg_idx, clean_stats = _clean_batch_with_indices(texts, source_indices)
  sw = _load_stopwords(stopwords_path)
  userdict = userdict_path if userdict_path and userdict_path.exists() else None
  seg = _segment(cleaned, sw, userdict)

  load_env_file()
  api_key = get_api_key()
  if not api_key:
    payload["error"] = "未配置 API 密钥（DASHSCOPE_APIKEY / QWEN_APIKEY）"
    return False, payload

  cache_file = out_dir / "embedding_cache"
  log_success(logger, f"开始向量化，文本数量: {len(seg)}", "TopicBertopic")
  vecs = _embed(
    seg,
    api_key,
    logger,
    cache_file=cache_file,
    batch_size=8,
    semaphore_limit=3,
    batch_max_retries=6,
    save_interval=80,
    window_size=120,
    request_timeout_s=120,
  )
  if vecs.size == 0:
    payload["error"] = "向量化失败"
    return False, payload

  log_success(logger, f"向量化完成，向量维度: {vecs.shape}", "TopicBertopic")
  emb_npy, emb_meta = _save_run_embedding_matrix(out_dir, vecs, texts=seg, logger=logger)
  try:
    _patch_hdbscan_sklearn_compat(logger)
    model = _build_bertopic()
    model.fit_transform(seg, embeddings=vecs)
    log_success(logger, "主题建模完成", "TopicBertopic")
  except Exception as e:
    payload["error"] = f"主题建模失败: {e}"
    log_error(logger, payload["error"], "TopicBertopic")
    return False, payload

  stats_json = _generate_jsons(model, seg, vecs, out_dir, logger)
  final_result = asyncio.run(
    _generate_reclustered_json(
      stats_json,
      domain_topic,
      out_dir,
      logger,
      df=df,
      text_col=text_col,
      doc_id_col=doc_id_col,
      ct_to_seg_idx=ct_to_seg_idx,
      event_introduction=event_introduction,
    )
  )
  if not final_result:
    payload["error"] = "大模型主题合并失败"
    payload["artifacts"] = {
      "output_dir": str(out_dir),
      "preliminary_stats": str(out_dir / "1主题统计结果.json"),
    }
    return False, payload

  topics = _format_topics_for_sona(final_result)
  irrelevant = sum(1 for t in topics if t.get("label") == "无关主题")
  payload["topics"] = topics
  payload["statistics"] = {
    "input_rows": len(df),
    "valid_text_rows": len(texts),
    "clustered_unique_texts": clean_stats.get("final", 0),
    "duplicate_dropped": clean_stats.get("duplicates", 0),
    "topic_count": len(topics),
    "relevant_topic_count": max(0, len(topics) - irrelevant),
    "irrelevant_topic_count": irrelevant,
    "content_column": text_col,
    "domain_topic": domain_topic,
  }
  payload["artifacts"] = {
    "output_dir": str(out_dir),
    "recluster_json": str(out_dir / "4大模型再聚类结果.json"),
    "recluster_keywords_json": str(out_dir / "5大模型主题关键词.json"),
    "doc_topic_csv": str(out_dir / "6文档主题归属表.csv"),
    "embedding_npy": str(emb_npy),
    "embedding_meta": str(emb_meta),
    "embedding_cache_json": str(cache_file.with_suffix(".json")),
    "embedding_cache_npy": str(cache_file.with_suffix(".npy")),
    "embedding_cache": str(cache_file.with_suffix(".json")),
  }
  return True, payload


@tool
def analysis_topic_bertopic(
  eventIntroduction: str,
  dataFilePath: str,
  retryContext: Optional[str] = None,
  contentColumns: Optional[List[str]] = None,
  domainTopic: Optional[str] = None,
) -> str:
  """
  描述：对舆情 CSV 做 BERTopic 主题聚类，并用通义大模型将细粒度簇合并为可解读的议题列表。
  输入与 keyword_stats / analysis_sentiment 一致：自动识别内容列并做相同规则清洗。
  输出：JSON 字符串，含 topics、statistics、result_file_path；详细产物保存在任务过程目录的 topic_bertopic_* 子目录。
  可通过环境变量 SONA_TOPIC_BERTOPIC_MAX_ROWS 限制参与聚类的最大行数（默认 3000）。
  """
  import json as json_module

  if retryContext:
    try:
      json_module.loads(retryContext) if isinstance(retryContext, str) else retryContext
    except Exception:
      pass

  try:
    all_data = read_csv_rows_all(dataFilePath)
  except Exception as e:
    return json_module.dumps(
      {"error": f"读取数据文件失败: {e}", "topics": [], "statistics": {}, "result_file_path": ""},
      ensure_ascii=False,
    )

  if not all_data:
    return json_module.dumps(
      {"error": "数据文件为空", "topics": [], "statistics": {}, "result_file_path": ""},
      ensure_ascii=False,
    )

  fieldnames = list(all_data[0].keys())
  forced_cols: List[str] = []
  if contentColumns:
    normalized = [str(c or "").strip() for c in contentColumns if str(c or "").strip()]
    header_set = {str(h) for h in fieldnames}
    forced_cols = [c for c in normalized if c in header_set]
  content_columns = forced_cols if forced_cols else _identify_content_columns(fieldnames)
  if not content_columns:
    return json_module.dumps(
      {
        "error": "无法识别内容列，请确保列名包含: " + ", ".join(CONTENT_COLUMN_KEYWORDS),
        "topics": [],
        "statistics": {},
        "result_file_path": "",
      },
      ensure_ascii=False,
    )

  domain_topic = (domainTopic or "sona_event").strip() or "sona_event"
  logger = setup_logger(domain_topic, datetime.now().strftime("%Y%m%d"))
  log_module_start(logger, "analysis_topic_bertopic")

  try:
    _ensure_safe_runtime_env(get_project_root(), logger)
  except Exception:
    pass

  df, _, row_meta = _build_dataframe_from_csv_rows(
    all_data,
    content_columns,
    max_rows=_TOPIC_BERTOPIC_MAX_ROWS,
  )
  text_col = "_sona_bertopic_text"
  if text_col not in df.columns:
    return json_module.dumps(
      {"error": "构建建模文本列失败", "topics": [], "statistics": {}, "result_file_path": ""},
      ensure_ascii=False,
    )

  task_id = get_task_id()
  artifact_parent = (
    ensure_task_dirs(task_id) / f"topic_bertopic_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    if task_id
    else Path(dataFilePath).resolve().parent / "topic_bertopic_artifacts"
  )
  stopwords_path = get_config_dir() / "stopwords.txt"
  userdict_path = get_config_dir() / "userdict.txt"

  ok, core_payload = _run_bertopic_core(
    df=df,
    text_col=text_col,
    out_dir=artifact_parent,
    logger=logger,
    event_introduction=eventIntroduction,
    domain_topic=domain_topic,
    stopwords_path=stopwords_path,
    userdict_path=userdict_path if userdict_path.exists() else None,
  )

  statistics = dict(core_payload.get("statistics") or {})
  statistics["content_columns"] = content_columns
  statistics["csv_row_meta"] = row_meta
  statistics["data_file_path"] = dataFilePath

  full_result: Dict[str, Any] = {
    "topics": core_payload.get("topics") or [],
    "statistics": statistics,
    "artifacts": core_payload.get("artifacts") or {},
    "event_introduction": (eventIntroduction or "")[:500],
    "error": core_payload.get("error", "") if not ok else "",
  }

  if task_id and ok:
    try:
      run_stamp = artifact_parent.name.replace("topic_bertopic_", "", 1)
      process_dir = ensure_task_dirs(task_id)
      synced = _sync_topic_embeddings_to_process_dir(
        process_dir,
        artifact_parent,
        run_stamp=run_stamp,
        logger=logger,
      )
      if synced:
        arts = full_result.setdefault("artifacts", {})
        arts.update(synced)
    except Exception as e:
      full_result["embedding_sync_error"] = f"同步向量到过程目录失败: {e}"

  result_file_path = ""
  if task_id:
    try:
      process_dir = ensure_task_dirs(task_id)
      filename = _generate_topic_result_filename(retryContext)
      result_file = process_dir / filename
      full_result["result_file_path"] = str(result_file)
      with open(result_file, "w", encoding="utf-8", errors="replace") as f:
        json_module.dump(full_result, f, ensure_ascii=False, indent=2)
      result_file_path = str(result_file)
    except Exception as e:
      full_result["save_error"] = f"保存结果文件失败: {e}"
      full_result["result_file_path"] = ""
  else:
    full_result["save_error"] = "未找到任务ID，无法保存结果文件"
    full_result["result_file_path"] = ""

  summary: Dict[str, Any] = {
    "message": (
      f"主题聚类完成：议题 {statistics.get('topic_count', 0)} 个"
      f"（相关 {statistics.get('relevant_topic_count', 0)} 个）"
      if ok
      else f"主题聚类失败: {full_result.get('error', '')}"
    ),
    "topics": full_result.get("topics", [])[:12],
    "statistics": statistics,
    "result_file_path": result_file_path,
    "artifacts": full_result.get("artifacts", {}),
  }
  if full_result.get("error"):
    summary["error"] = full_result["error"]
  if full_result.get("save_error"):
    summary["save_error"] = full_result["save_error"]
  return json_module.dumps(summary, ensure_ascii=False)


def run_topic_bertopic(topic: str, start_date: str, end_date: str = None,
                       fetch_dir: Optional[str] = None,
                       input_file: Optional[str] = None,
                       output_dir: Optional[str] = None,
                       text_col: Optional[str] = None,
                       userdict: Optional[str] = None, stopwords: Optional[str] = None) -> bool:
    # 使用日期范围格式作为日志标识
    date_range = f"{start_date}_{end_date}" if end_date else start_date
    logger = setup_logger(topic, date_range)
    log_module_start(logger, "TopicBertopic")

    # Windows 兼容性：避免第三方库在含中文的 TEMP 路径下触发 ascii 编码错误
    try:
        _ensure_safe_runtime_env(get_project_root(), logger)
    except Exception:
        # 安全兜底：不影响主流程
        pass

    paths = _default_paths(topic, start_date, end_date)
    userdict_path = Path(userdict) if userdict else paths["userdict"]
    stopwords_path = Path(stopwords) if stopwords else paths["stopwords"]
    out_analyze = Path(output_dir) if output_dir else DEFAULT_OUTPUT_DIR
    out_analyze.mkdir(parents=True, exist_ok=True)

    preferred_text_col = (text_col or DEFAULT_TEXT_COL).strip()
    in_path = Path(input_file) if input_file else DEFAULT_INPUT_FILE

    try:
        if in_path.exists():
            log_success(logger, f"读取输入: {in_path}", "TopicBertopic")
            df = _load_input_table(in_path, logger)
        elif fetch_dir:
            log_success(logger, f"输入文件不存在，回退 fetch 目录: {fetch_dir}", "TopicBertopic")
            df = _load_and_merge_fetch_data(Path(fetch_dir), logger)
        else:
            fetch_path = paths["fetch_dir"]
            if fetch_path.exists():
                log_success(logger, f"输入文件不存在，回退 fetch 目录: {fetch_path}", "TopicBertopic")
                df = _load_and_merge_fetch_data(fetch_path, logger)
            else:
                log_error(logger, f"输入文件不存在: {in_path}", "TopicBertopic")
                return False
        if df.empty:
            log_error(logger, "未读取到任何数据", "TopicBertopic")
            return False

        resolved_text_col = _resolve_text_column(df, preferred_text_col)
        if not resolved_text_col:
            log_error(
                logger,
                f"输入数据中未找到文本列「{preferred_text_col}」，当前列: {list(df.columns)}",
                "TopicBertopic",
            )
            return False
        text_col = resolved_text_col
        log_success(logger, f"建模文本列: {text_col} | 输出目录: {out_analyze}", "TopicBertopic")
        ud_path = userdict_path if isinstance(userdict_path, Path) and userdict_path.exists() else None
        ok, payload = _run_bertopic_core(
            df=df,
            text_col=text_col,
            out_dir=out_analyze,
            logger=logger,
            event_introduction="",
            domain_topic=topic,
            stopwords_path=stopwords_path,
            userdict_path=ud_path,
        )
        if not ok:
            log_error(logger, payload.get("error", "主题分析失败"), "TopicBertopic")
            return False
        log_success(logger, "主题分析完成", "TopicBertopic")
        return True
    except Exception as e:
        log_error(logger, f"异常: {e}", "TopicBertopic")
        return False


def main() -> int:
    parser = argparse.ArgumentParser(description="BERTopic + Qwen 主题分析（默认：正式处理文件.xlsx 正文列）")
    parser.add_argument("--topic", default="没出息", help="专题名称（默认：没出息）")
    parser.add_argument("--start-date", default="0501", help="开始日期标识（默认：0501）")
    parser.add_argument("--end-date", default=None, help="结束日期，如 2026-04-30")
    parser.add_argument("--fetch-dir", default=None, help="可选：输入文件缺失时回退的 fetch 数据目录")
    parser.add_argument(
        "--input-file",
        default=str(DEFAULT_INPUT_FILE),
        help=f"输入表路径（默认：{DEFAULT_INPUT_FILE.name}）",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help=f"输出目录（默认：{DEFAULT_OUTPUT_DIR.name}）",
    )
    parser.add_argument(
        "--text-col",
        default=DEFAULT_TEXT_COL,
        help=f"建模用文本列名（默认：{DEFAULT_TEXT_COL}）",
    )
    parser.add_argument("--userdict", default=None, help="可选：用户词典路径")
    parser.add_argument("--stopwords", default=None, help="可选：停用词路径")
    args = parser.parse_args()

    ok = run_topic_bertopic(
        topic=args.topic,
        start_date=args.start_date,
        end_date=args.end_date,
        fetch_dir=args.fetch_dir,
        input_file=args.input_file,
        output_dir=args.output_dir,
        text_col=args.text_col,
        userdict=args.userdict,
        stopwords=args.stopwords,
    )
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())


