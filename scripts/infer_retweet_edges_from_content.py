#!/usr/bin/env python3
"""
从微博汇总 CSV 推断「谁引用/转发谁」的有向边，并在「作者」列（及可选 URL→uid）中解析目标。

除「@昵称」类信号外，还支持：
- 无 @：`转发自/转自/源自/via/来源：` 等后的纯昵称；
- 全角 `＠`、全角斜杠 `／／@`；
- `[某某的微博视频]` 等卡片残留文案；
- 可选：`weibo.com/uid` 与表内 URL 映射；
- 可选：同标题多作者 → 指向发布时间最早的一条（弱信号）；
- 可选：长正文包含另一条更短的全文且时间更晚 → 指向短帖作者（洗稿/搬运启发式，噪声较大）。

用法见 `python scripts/infer_retweet_edges_from_content.py --help`。
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


# 半角 / 全角 斜杠 + @ / ＠
RE_WEIBO_RETWEET_HANDLE = re.compile(
    r"(?:/|／){2}\s*(?:@|＠)([^:/：\s　]{1,40})[:：]?",
    re.UNICODE,
)
RE_ATTRIBUTION = re.compile(
    r"(?:转自|来自)\s*(?:@|＠)([\w\u4e00-\u9fff.\-·]{1,40})(?=[\s)）\]、，,;<]|http|$)",
    re.UNICODE | re.IGNORECASE,
)
RE_SHARE_FROM = re.compile(
    r"分享自\s*(?:@|＠)([\w\u4e00-\u9fff.\-·]{1,40})(?=[\s)）\]、，,;<]|http|$)",
    re.UNICODE,
)
RE_FORWARD_NEAR_AT = re.compile(
    r"(?:转发|转发了|转自)[:：\s]*(?:@|＠)([^\s,，.。]{2,30})",
    re.UNICODE,
)
# 无 @：须带明显「出处」引导词，避免匹配正文里的「源自真实故事」等
RE_PLAIN_ATTRIBUTION = re.compile(
    r"(?:转发自|轉載自)[:：\s]+([\w\u4e00-\u9fff.\-·]{2,24})(?=[\s,，。#]|$)|"
    r"转自[:：\s]+([\w\u4e00-\u9fff.\-·]{2,24})(?=[\s,，。#]|$)|"
    r"出处[:：\s]+([\w\u4e00-\u9fff.\-·]{2,24})(?=[\s,，。#]|$)|"
    r"(?:来源|via|VIA)[:：\s]+([\w\u4e00-\u9fff.\-·]{2,24})(?=[\s,，。#]|$)",
    re.UNICODE | re.IGNORECASE,
)
# [用户名的微博视频]
RE_BRACKET_WEIBO_CARD = re.compile(
    r"\[([\w\u4e00-\u9fff.\-·]{2,20})的微博(?:视频|直播|图片)\]",
    re.UNICODE,
)
RE_WEIBO_UID = re.compile(
    r"(?:https?://)?(?:www\.)?weibo\.com/(?:u/)?(\d{6,})",
    re.IGNORECASE,
)

_PLAIN_ATTR_STOPWORDS = frozenset(
    {
        "微博",
        "新浪",
        "网络",
        "网友",
        "平台",
        "媒体",
        "豆瓣",
        "抖音",
        "快手",
        "小红书",
        "微信",
        "公众号",
    }
)


def _norm(s: str) -> str:
    return re.sub(r"\s+", "", (s or "").strip())


def _norm_title(s: str) -> str:
    return _norm(s)[:500]


def _row_time_ms(r: Dict[str, str], *, time_key: str = "发布时间", ts_key: str = "发布时间戳") -> int:
    ts = str(r.get(ts_key) or "").strip()
    if ts.isdigit():
        v = int(ts)
        if v > 10_000_000_000_000:  # 已是毫秒
            return v
        if v > 1_000_000_000_000:  # 微秒级少见，当毫秒
            return v
        return int(v * 1000) if v < 10_000_000_000 else v
    raw = (r.get(time_key) or "").strip()
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y/%m/%d %H:%M:%S", "%Y-%m-%d"):
        try:
            s = raw[:19] if len(raw) >= 19 else raw
            return int(datetime.strptime(s, fmt).timestamp() * 1000)
        except ValueError:
            continue
    return 0


@dataclass(frozen=True)
class Edge:
    """一条推断边：当前帖作者 -> 目标作者。"""

    row_id: str
    from_author: str
    to_author: str
    raw_handle: str
    match_kind: str  # exact | fuzzy | uid_lookup
    pattern: str


@dataclass
class MatchStats:
    rows_total: int
    rows_with_handle_pattern: int
    handles_found: int
    handles_matched_exact: int
    handles_matched_fuzzy: int
    handles_unmatched: int
    edges_from_handles: int
    rows_with_uid_in_content: int = 0
    uid_tokens_matched: int = 0
    uid_tokens_unmapped: int = 0
    edges_from_uids: int = 0
    same_title_clusters: int = 0
    edges_from_same_title: int = 0
    embed_candidates_scanned: int = 0
    edges_from_content_embed: int = 0


def _load_authors(rows: Sequence[Dict[str, str]], author_key: str) -> Tuple[List[str], Dict[str, List[str]]]:
    norm_to_canon: Dict[str, List[str]] = defaultdict(list)
    for r in rows:
        a = (r.get(author_key) or "").strip()
        if not a:
            continue
        k = _norm(a)
        if k and a not in norm_to_canon[k]:
            norm_to_canon[k].append(a)
    authors = sorted({(r.get(author_key) or "").strip() for r in rows if (r.get(author_key) or "").strip()})
    return authors, dict(norm_to_canon)


def _extract_name_candidates(text: str) -> List[Tuple[str, str]]:
    """从正文提取 (昵称片段, pattern)，按出现顺序、去重。"""
    if not text:
        return []
    seen: set[str] = set()
    out: List[Tuple[str, str]] = []

    def push(h: str, pat: str) -> None:
        h = h.strip()
        if len(h) < 1:
            return
        key = _norm(h)
        if key in seen:
            return
        seen.add(key)
        out.append((h, pat))

    for m in RE_WEIBO_RETWEET_HANDLE.finditer(text):
        push(m.group(1), "retweet_chain")
    for m in RE_ATTRIBUTION.finditer(text):
        push(m.group(1), "attribution_at")
    for m in RE_SHARE_FROM.finditer(text):
        push(m.group(1), "share_from")
    for m in RE_FORWARD_NEAR_AT.finditer(text):
        push(m.group(1), "forward_at")
    for m in RE_PLAIN_ATTRIBUTION.finditer(text):
        name = next((m.group(i) for i in range(1, 5) if m.group(i)), None)
        if name and _norm(name) not in _PLAIN_ATTR_STOPWORDS and len(name) >= 2:
            push(name, "plain_attribution")
    for m in RE_BRACKET_WEIBO_CARD.finditer(text):
        push(m.group(1), "weibo_card_bracket")
    return out


def _match_handle_to_author(
    handle: str,
    norm_to_canon: Dict[str, List[str]],
    *,
    min_fuzzy_len: int,
) -> Tuple[Optional[str], str]:
    hn = _norm(handle)
    if not hn:
        return None, "none"
    if hn in norm_to_canon:
        return norm_to_canon[hn][0], "exact"
    best: Optional[str] = None
    if len(hn) >= min_fuzzy_len:
        for anorm, variants in norm_to_canon.items():
            if len(anorm) < min_fuzzy_len:
                continue
            if hn in anorm or anorm in hn:
                cand = variants[0]
                if best is None:
                    best = cand
                else:
                    if abs(len(_norm(cand)) - len(hn)) < abs(len(_norm(best)) - len(hn)):
                        best = cand
    if best:
        return best, "fuzzy"
    return None, "none"


def _uid_from_weibo_url(url: str) -> Optional[str]:
    m = RE_WEIBO_UID.search(url or "")
    return m.group(1) if m else None


def _build_uid_to_author(
    rows: Sequence[Dict[str, str]],
    *,
    url_key: str = "URL",
    author_key: str = "作者",
) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for r in rows:
        uid = _uid_from_weibo_url(r.get(url_key) or "")
        if not uid:
            continue
        au = (r.get(author_key) or "").strip()
        if au and uid not in out:
            out[uid] = au
    return out


def _infer_edges_weibo_uid(
    rows: Sequence[Dict[str, str]],
    uid_to_author: Dict[str, str],
    *,
    content_key: str = "内容",
    author_key: str = "作者",
    id_key: str = "id",
    url_key: str = "URL",
) -> Tuple[List[Edge], Tuple[int, int, int, int]]:
    edges: List[Edge] = []
    rows_with_uid = 0
    tok_map = 0
    tok_miss = 0

    for r in rows:
        content = r.get(content_key) or ""
        author = (r.get(author_key) or "").strip() or "（空作者）"
        rid = str(r.get(id_key) or "").strip() or "（空id）"
        self_uid = _uid_from_weibo_url(r.get(url_key) or "") or ""

        uids = list(dict.fromkeys(RE_WEIBO_UID.findall(content)))
        if not uids:
            continue
        local_row_hit = False
        for uid in uids:
            if uid == self_uid:
                continue
            to_a = uid_to_author.get(uid)
            if not to_a:
                tok_miss += 1
                continue
            if to_a == author:
                continue
            tok_map += 1
            local_row_hit = True
            edges.append(
                Edge(
                    row_id=rid,
                    from_author=author,
                    to_author=to_a,
                    raw_handle=uid,
                    match_kind="uid_lookup",
                    pattern="weibo_uid",
                )
            )
        if local_row_hit:
            rows_with_uid += 1

    return edges, (rows_with_uid, tok_map, tok_miss, len(edges))


def _infer_same_title_edges(
    rows: Sequence[Dict[str, str]],
    *,
    title_key: str = "标题",
    author_key: str = "作者",
    id_key: str = "id",
    min_title_len: int = 10,
) -> Tuple[List[Edge], int, int]:
    """同一归一化标题下，非本人作者指向该簇内时间最早的一条（弱：通稿/话题重复）。"""
    clusters: Dict[str, List[Tuple[int, str, str]]] = defaultdict(list)
    for r in rows:
        title = (r.get(title_key) or "").strip()
        if len(_norm(title)) < min_title_len:
            continue
        author = (r.get(author_key) or "").strip()
        if not author:
            continue
        rid = str(r.get(id_key) or "").strip() or "（空id）"
        t = _row_time_ms(r)
        clusters[_norm_title(title)].append((t, author, rid))

    edges: List[Edge] = []
    n_clu = 0
    for _title, lst in clusters.items():
        if len(lst) < 2:
            continue
        authors_set = {x[1] for x in lst}
        if len(authors_set) < 2:
            continue
        n_clu += 1
        lst.sort(key=lambda x: x[0])
        _, hub_author, _ = lst[0]
        for t, au, rid in lst[1:]:
            if au == hub_author:
                continue
            edges.append(
                Edge(
                    row_id=rid,
                    from_author=au,
                    to_author=hub_author,
                    raw_handle=_title[:80],
                    match_kind="fuzzy",
                    pattern="same_title_earliest",
                )
            )
    return edges, n_clu, len(edges)


def _infer_content_embed_edges(
    rows: Sequence[Dict[str, str]],
    *,
    content_key: str = "内容",
    author_key: str = "作者",
    id_key: str = "id",
    min_sub_len: int,
    max_ratio: float,
    max_pairs: int,
) -> Tuple[List[Edge], int, int]:
    """
    若 row A 的正文包含 row B 的完整正文（B 更短），且 A 时间不早于 B，则 A -> B。
    按正文长度升序枚举短帖，对每条长帖在更短候选中扫描（有对数上限）。
    """
    packed: List[Tuple[int, int, str, str, str]] = []
    for r in rows:
        c = (r.get(content_key) or "").strip()
        if len(c) < min_sub_len:
            continue
        author = (r.get(author_key) or "").strip() or "（空作者）"
        rid = str(r.get(id_key) or "").strip() or "（空id）"
        t = _row_time_ms(r)
        packed.append((len(c), t, author, rid, c))

    by_len_asc = sorted(packed, key=lambda x: x[0])
    edges: List[Edge] = []
    scanned = 0

    for ln, t_long, au_long, rid_long, c_long in sorted(packed, key=lambda x: -x[0]):
        if ln < min_sub_len * 2:
            continue
        for ln_s, t_s, au_s, rid_s, c_short in by_len_asc:
            if ln_s >= ln:
                break
            if ln_s < min_sub_len:
                continue
            if ln_s / max(ln, 1) > max_ratio:
                continue
            scanned += 1
            if scanned > max_pairs:
                return edges, scanned, len(edges)
            if au_s == au_long:
                continue
            if t_s > t_long:
                continue
            if c_short not in c_long:
                continue
            edges.append(
                Edge(
                    row_id=rid_long,
                    from_author=au_long,
                    to_author=au_s,
                    raw_handle=f"embed_len={ln_s}",
                    match_kind="fuzzy",
                    pattern="content_contains",
                )
            )
            break  # 每条长帖只连一条最短的「被包含」帖，减少爆炸

    return edges, scanned, len(edges)


def infer_edges(
    rows: Sequence[Dict[str, str]],
    *,
    content_key: str = "内容",
    author_key: str = "作者",
    id_key: str = "id",
    min_fuzzy_len: int = 3,
    only_first_handle: bool = False,
    include_uid_links: bool = False,
    url_key: str = "URL",
    same_title_edges: bool = False,
    content_embed_edges: bool = False,
    embed_min_len: int = 80,
    embed_max_ratio: float = 0.92,
    embed_max_pairs: int = 400_000,
    title_key: str = "标题",
) -> Tuple[List[Edge], MatchStats]:
    _, norm_to_canon = _load_authors(rows, author_key)
    edges: List[Edge] = []
    st = MatchStats(
        rows_total=len(rows),
        rows_with_handle_pattern=0,
        handles_found=0,
        handles_matched_exact=0,
        handles_matched_fuzzy=0,
        handles_unmatched=0,
        edges_from_handles=0,
    )

    for r in rows:
        content = r.get(content_key) or ""
        author = (r.get(author_key) or "").strip() or "（空作者）"
        rid = str(r.get(id_key) or "").strip() or "（空id）"
        pairs = _extract_name_candidates(content)
        if not pairs:
            continue
        st.rows_with_handle_pattern += 1
        iterable: Iterable[Tuple[str, str]] = pairs[:1] if only_first_handle else pairs
        for handle, pat in iterable:
            st.handles_found += 1
            to_a, mk = _match_handle_to_author(handle, norm_to_canon, min_fuzzy_len=min_fuzzy_len)
            if mk == "exact":
                st.handles_matched_exact += 1
            elif mk == "fuzzy":
                st.handles_matched_fuzzy += 1
            else:
                st.handles_unmatched += 1
                continue
            if to_a == author:
                continue
            edges.append(
                Edge(
                    row_id=rid,
                    from_author=author,
                    to_author=to_a,
                    raw_handle=handle,
                    match_kind=mk,
                    pattern=pat,
                )
            )
            st.edges_from_handles += 1

    if include_uid_links:
        uid_map = _build_uid_to_author(rows, url_key=url_key, author_key=author_key)
        uid_edges, (ru, tm, tmiss, ec) = _infer_edges_weibo_uid(
            rows, uid_map, content_key=content_key, author_key=author_key, id_key=id_key, url_key=url_key
        )
        edges.extend(uid_edges)
        st.rows_with_uid_in_content = ru
        st.uid_tokens_matched = tm
        st.uid_tokens_unmapped = tmiss
        st.edges_from_uids = ec

    if same_title_edges:
        ste, ncl, ne = _infer_same_title_edges(
            rows,
            title_key=title_key,
            author_key=author_key,
            id_key=id_key,
        )
        edges.extend(ste)
        st.same_title_clusters = ncl
        st.edges_from_same_title = ne

    if content_embed_edges:
        ee, scanned, ne = _infer_content_embed_edges(
            rows,
            content_key=content_key,
            author_key=author_key,
            id_key=id_key,
            min_sub_len=embed_min_len,
            max_ratio=embed_max_ratio,
            max_pairs=embed_max_pairs,
        )
        edges.extend(ee)
        st.embed_candidates_scanned = scanned
        st.edges_from_content_embed = ne

    return edges, st


def _read_csv_rows(path: Path, limit: Optional[int]) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="", errors="replace") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            raise ValueError("CSV 无表头")
        rows: List[Dict[str, str]] = []
        for i, row in enumerate(reader):
            if limit is not None and i >= limit:
                break
            rows.append({k: (v if v is not None else "") for k, v in row.items()})
    return rows


def main() -> int:
    project_root = Path(__file__).resolve().parents[1]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    ap = argparse.ArgumentParser(description="从「内容」等推断传播边并在「作者」列匹配")
    ap.add_argument(
        "--csv",
        type=Path,
        default=project_root
        / "sandbox/a1a55342-ba70-4774-9d4e-4a65aeabf433/过程文件/netinsight_微博_汇总_20260515_162607.csv",
        help="输入 CSV 路径",
    )
    ap.add_argument("--out", type=Path, default=None, help="输出 JSON")
    ap.add_argument("--limit", type=int, default=None, help="只读前 N 行")
    ap.add_argument("--min-fuzzy-len", type=int, default=3, help="昵称模糊匹配最短长度")
    ap.add_argument("--only-first-handle", action="store_true", help="每条正文只取第一个昵称候选")
    ap.add_argument("--include-uid-links", action="store_true", help="正文 weibo uid + 表内 URL 映射")
    ap.add_argument("--same-title-edges", action="store_true", help="同标题多作者 → 指向时间最早的一条（弱）")
    ap.add_argument("--content-embed-edges", action="store_true", help="长帖含短帖全文 → 指向短帖（噪声大，见参数）")
    ap.add_argument("--embed-min-len", type=int, default=80, help="短帖最短字数才参与 embed 推断")
    ap.add_argument("--embed-max-ratio", type=float, default=0.92, help="短/长 长度比上限（避免整帖复制）")
    ap.add_argument("--embed-max-pairs", type=int, default=400_000, help="embed 扫描对数上限（防卡死）")
    ap.add_argument("--sample-unmatched", type=int, default=8, help="未匹配昵称示例条数，0 关闭")
    args = ap.parse_args()

    csv_path: Path = args.csv.resolve()
    if not csv_path.is_file():
        print(f"文件不存在: {csv_path}", file=sys.stderr)
        return 1

    rows = _read_csv_rows(csv_path, args.limit)
    edges, stats = infer_edges(
        rows,
        min_fuzzy_len=max(2, args.min_fuzzy_len),
        only_first_handle=args.only_first_handle,
        include_uid_links=args.include_uid_links,
        same_title_edges=args.same_title_edges,
        content_embed_edges=args.content_embed_edges,
        embed_min_len=max(30, args.embed_min_len),
        embed_max_ratio=min(0.99, max(0.1, args.embed_max_ratio)),
        embed_max_pairs=max(1000, args.embed_max_pairs),
    )

    print("=== 传播边推断 ===")
    print(f"CSV: {csv_path}")
    print(f"行数: {stats.rows_total}")
    print("\n[昵称 / 无@ / 卡片 等文本通道]")
    print(f"  命中至少一种模式的行: {stats.rows_with_handle_pattern}")
    print(f"  候选昵称数: {stats.handles_found}")
    print(f"    精确→作者: {stats.handles_matched_exact}")
    print(f"    模糊→作者: {stats.handles_matched_fuzzy}")
    print(f"    未匹配作者表: {stats.handles_unmatched}")
    print(f"  边数: {stats.edges_from_handles}")
    if args.include_uid_links:
        print("\n[uid 链接]")
        print(f"  有 uid 边的行: {stats.rows_with_uid_in_content}")
        print(f"  uid 映射命中: {stats.uid_tokens_matched} / 未映射: {stats.uid_tokens_unmapped}")
        print(f"  边数: {stats.edges_from_uids}")
    if args.same_title_edges:
        print("\n[同标题]")
        print(f"  多作者标题簇数: {stats.same_title_clusters}")
        print(f"  边数: {stats.edges_from_same_title}")
    if args.content_embed_edges:
        print("\n[正文包含]")
        print(f"  扫描对数(上限内): {stats.embed_candidates_scanned}")
        print(f"  边数: {stats.edges_from_content_embed}")

    print(f"\n合计边数: {len(edges)}")
    out_d = Counter(e.from_author for e in edges)
    in_d = Counter(e.to_author for e in edges)
    print("\n--- 出度 Top10 ---")
    for name, c in out_d.most_common(10):
        print(f"  {c:4d}  {name}")
    print("\n--- 入度 Top10 ---")
    for name, c in in_d.most_common(10):
        print(f"  {c:4d}  {name}")

    if args.sample_unmatched > 0 and stats.handles_unmatched:
        print("\n--- 未匹配昵称示例 ---")
        _, n2c = _load_authors(rows, "作者")
        shown = 0
        for r in rows:
            if shown >= args.sample_unmatched:
                break
            content = r.get("内容") or ""
            author = (r.get("作者") or "").strip()
            for handle, pat in _extract_name_candidates(content):
                to_a, mk = _match_handle_to_author(handle, n2c, min_fuzzy_len=max(2, args.min_fuzzy_len))
                if mk == "none":
                    snippet = content[:120].replace("\n", " ")
                    print(f"  [{pat}] {handle} <- 作者={author} | {snippet}...")
                    shown += 1
                    break

    if args.out:
        payload = {
            "source_csv": str(csv_path),
            "stats": asdict(stats),
            "edges": [asdict(e) for e in edges],
        }
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\n已写入: {args.out.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
