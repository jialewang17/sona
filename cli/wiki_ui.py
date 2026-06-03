"""CLI bridge for /wiki command."""

from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from rich import box
from rich.console import Group
from rich.panel import Panel
from rich.prompt import Prompt
from rich.table import Table
from rich.text import Text

from cli.display import console
from tools.oprag import build_reference_wiki
from utils.env_loader import get_env_config, reload_env_config
from utils.path import get_opinion_analysis_kb_root
from workflow.wiki_cli import answer_wiki_query


def _wiki_panel_width() -> int:
    """统一 Wiki 各模块 Panel 宽度，与终端可视宽度对齐。"""
    try:
        return max(72, int(console.size.width))
    except Exception:
        return 100


def _wiki_panel(
    renderable: Any,
    *,
    title: str,
    border_style: str,
    padding: tuple[int, int] = (1, 2),
) -> Panel:
    return Panel(
        renderable,
        title=title,
        border_style=border_style,
        padding=padding,
        expand=True,
        width=_wiki_panel_width(),
    )


def _strip_markdown_for_display(text: str) -> str:
    """去掉模型输出的 Markdown 标记，避免终端原样显示 ** 等符号。"""
    normalized = str(text or "").replace("\r\n", "\n").replace("\r", "\n")
    normalized = re.sub(r"^\s{0,3}#{1,6}\s*", "", normalized, flags=re.MULTILINE)
    normalized = normalized.replace("**", "").replace("__", "")
    normalized = re.sub(r"\*([^*\n]+)\*", r"\1", normalized)
    normalized = normalized.replace("`", "")
    normalized = re.sub(r"\n{3,}", "\n\n", normalized)
    return normalized.strip()


def _strip_template_answer_sections(answer: str) -> str:
    """去掉模型常输出的编号模板与重复的来源块（来源由下方单独展示）。"""
    text = _strip_markdown_for_display(answer)
    if not text:
        return text
    for pat in (
        r"\n\s*2[)\)]\s*依据来源[\s\S]*$",
        r"\n\s*##?\s*依据来源[\s\S]*$",
        r"\n\s*依据来源[：:][\s\S]*$",
    ):
        text = re.sub(pat, "", text, flags=re.IGNORECASE).strip()
    text = re.sub(r"^1[)\)]\s*全面回答[：:\s]*", "", text).strip()
    text = re.sub(r"^✅\s*", "", text, flags=re.MULTILINE)
    return text


_SECTION_TITLE_RE = re.compile(
    r"^(?:(先说|另外|还有|此外|有趣的是|简单来说|最后|总结一下|首先)[^。\n]{0,36}[：:]|(.{2,22}[：:]))\s*",
    re.DOTALL,
)


def _split_answer_into_sections(answer: str) -> List[tuple[str, str]]:
    """将回答拆成 (小标题, 正文) 块，便于分块展示。"""
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", str(answer or "").strip()) if p.strip()]
    if not paragraphs:
        return [("", str(answer or "").strip())]

    raw_sections: List[tuple[str, str]] = []
    for para in paragraphs:
        m = _SECTION_TITLE_RE.match(para)
        if m:
            colon = para.find("：") if "：" in para[:42] else para.find(":")
            if 0 <= colon <= 42:
                raw_sections.append((para[:colon].strip(), para[colon + 1 :].strip()))
                continue
        raw_sections.append(("", para))

    merged: List[tuple[str, str]] = []
    intro_parts: List[str] = []
    for title, body in raw_sections:
        if not title:
            intro_parts.append(body)
            continue
        if intro_parts:
            merged.append(("", "\n\n".join(intro_parts)))
            intro_parts = []
        merged.append((title, body))
    if intro_parts:
        merged.insert(0, ("", "\n\n".join(intro_parts)))
    return merged or [("", str(answer or "").strip())]


def _build_answer_renderable(sections: List[tuple[str, str]]) -> Text:
    """组装带小标题层级的回答正文。"""
    content = Text()
    for idx, (title, body) in enumerate(sections):
        if idx > 0:
            content.append("\n\n")
        if title:
            content.append(f"{title}\n", style="bold cyan")
        content.append(body.strip(), style="white")
    return content


def _wiki_route_label(meta: Dict[str, Any]) -> tuple[str, str]:
    if meta.get("wiki_route") == "neo4j_qa_only":
        nq = meta.get("neo4j_qa") if isinstance(meta.get("neo4j_qa"), dict) else {}
        rc = nq.get("rows_count", 0)
        return "熊猫知识库", f"图谱 {rc} 条线索"
    if meta.get("llm_used"):
        rc = meta.get("retrieved_count", 0)
        return "本地 Wiki", f"摘录 {rc} 条 · LLM 合成"
    rc = meta.get("retrieved_count", 0)
    return "本地 Wiki", f"摘录 {rc} 条"


def _render_wiki_header(query: str, meta: Dict[str, Any], *, source_count: int) -> None:
    route_name, route_detail = _wiki_route_label(meta)
    overview = Table(show_header=False, box=box.SIMPLE, padding=(0, 1), expand=True)
    overview.add_column(style="dim", width=10, no_wrap=True)
    overview.add_column(ratio=1)
    overview.add_row("问题", Text(query, style="bold white"))
    overview.add_row("知识来源", route_name)
    overview.add_row("检索", route_detail)
    if source_count:
        overview.add_row("延伸阅读", f"{source_count} 篇可用")

    err = str(
        meta.get("neo4j_qa_error")
        or meta.get("neo4j_qa_invoke_error")
        or meta.get("llm_error")
        or ""
    ).strip()
    if err:
        overview.add_row("提示", Text(err[:160], style="yellow"))

    console.print()
    console.print(
        _wiki_panel(
            overview,
            title="[bold cyan]Wiki 问答[/bold cyan]",
            border_style="cyan",
            padding=(0, 1),
        )
    )


def _render_wiki_answer(query: str, answer: str) -> None:
    sections = _split_answer_into_sections(answer)
    body = _build_answer_renderable(sections)
    subtitle = query if len(query) <= 56 else query[:55].rstrip() + "…"
    console.print(
        _wiki_panel(
            body,
            title=f"[bold green]科普回答[/bold green] [dim]· {subtitle}[/dim]",
            border_style="green",
            padding=(1, 2),
        )
    )


def _render_wiki_sources(sources: List[Dict[str, Any]], *, graph_route: bool) -> None:
    if not sources:
        console.print(
            _wiki_panel(
                Text("暂无延伸阅读材料。", style="dim"),
                title="[bold]延伸阅读[/bold]",
                border_style="dim",
                padding=(1, 2),
            )
        )
        return

    heading = "延伸阅读" if graph_route else "参考摘录"
    table = Table(
        box=box.ROUNDED,
        show_header=True,
        header_style="bold",
        expand=True,
        show_lines=False,
        padding=(0, 1),
    )
    table.add_column("#", style="dim", width=3, justify="right")
    table.add_column("资料", style="cyan", ratio=2, no_wrap=False)
    table.add_column("摘录", style="white", ratio=5, no_wrap=False)

    seen_docs: set[str] = set()
    shown = 0
    for item in sources:
        if not isinstance(item, dict):
            continue
        label = _friendly_source_label(item)
        snippet = _source_reading_snippet(item)
        doc_key = str(item.get("path") or label)
        if doc_key in seen_docs and graph_route:
            continue
        seen_docs.add(doc_key)
        shown += 1
        if shown > 4:
            break
        preview = snippet if len(snippet) <= 160 else snippet[:159].rstrip() + "…"
        doc_name = _source_doc_subtitle(item)
        label_cell = Text.assemble(
            (label, "bold cyan"),
            (f"\n{doc_name}", "dim italic") if doc_name and doc_name != label else "",
        )
        table.add_row(str(shown), label_cell, preview or "—")

    panel_content: Any = table
    if len(sources) > shown:
        panel_content = Group(
            table,
            Text(f"另有 {len(sources) - shown} 条相关材料未展开。", style="dim"),
        )

    console.print(
        _wiki_panel(
            panel_content,
            title=f"[bold yellow]{heading}[/bold yellow]",
            border_style="yellow",
            padding=(0, 1),
        )
    )


def _friendly_source_label(item: Dict[str, Any]) -> str:
    path = str(item.get("path") or "").strip()
    if path and not path.startswith("neo4j://"):
        stem = Path(path).stem
        stem = re.sub(r"^【[^】]+】", "", stem)
        if " - " in stem:
            tail = stem.rsplit(" - ", 1)[-1].strip()
            if tail:
                return tail
        if stem.strip():
            return stem.strip()
    title = str(item.get("title") or "").strip()
    if title.startswith("图谱："):
        return title[3:].strip()
    return title or "参考资料"


def _source_doc_subtitle(item: Dict[str, Any]) -> str:
    """资料条目的文档名（不含完整路径）。"""
    path = str(item.get("path") or "").strip()
    if path and not path.startswith("neo4j://"):
        stem = Path(path).stem
        stem = re.sub(r"^【[^】]+】", "", stem).strip()
        if stem:
            return stem
    return ""


def _source_reading_snippet(item: Dict[str, Any]) -> str:
    snippet = str(item.get("snippet") or "").strip()
    if "证据摘录：" in snippet:
        return snippet.split("证据摘录：", 1)[1].strip()
    if "—[" in snippet and "→" in snippet:
        lines = [ln.strip() for ln in snippet.splitlines() if ln.strip() and not ln.startswith("证据摘录")]
        if lines:
            return lines[0]
    return snippet


def _slugify_filename(text: str, max_len: int = 72) -> str:
    s = re.sub(r"[^\w\u4e00-\u9fff-]+", "_", str(text or "").strip()).strip("_")
    if not s:
        s = "approved_output"
    return s[:max_len]


def _wiki_root(project_root: Path) -> Path:
    return get_opinion_analysis_kb_root(project_root) / "references" / "wiki"


def _candidate_dir(project_root: Path) -> Path:
    return _wiki_root(project_root) / "output" / "_candidates"


def _list_candidate_files(project_root: Path) -> List[Path]:
    cdir = _candidate_dir(project_root)
    if not cdir.is_dir():
        return []
    files = [p for p in cdir.glob("*.md") if p.is_file()]
    files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return files


def _pick_candidate(project_root: Path, selector: str | None = None) -> Optional[Path]:
    files = _list_candidate_files(project_root)
    if not files:
        return None
    key = str(selector or "").strip().lower()
    if not key:
        return files[0]
    # 先精确后模糊
    for p in files:
        if p.name.lower() == key or p.stem.lower() == key:
            return p
    for p in files:
        if key in p.name.lower():
            return p
    return None


def _extract_question_from_candidate(md_text: str) -> str:
    marker = "## 原始问题"
    idx = md_text.find(marker)
    if idx < 0:
        return ""
    rest = md_text[idx + len(marker) :]
    lines = [ln.strip() for ln in rest.splitlines() if ln.strip()]
    if not lines:
        return ""
    return lines[0][:120]


def _approve_candidate_to_output(project_root: Path, candidate_file: Path) -> Dict[str, Any]:
    wiki_root = _wiki_root(project_root)
    out_dir = wiki_root / "output"
    out_dir.mkdir(parents=True, exist_ok=True)
    raw = candidate_file.read_text(encoding="utf-8", errors="replace")
    question = _extract_question_from_candidate(raw)
    stem = _slugify_filename(question or candidate_file.stem.replace("候选沉淀_", ""))
    target = out_dir / f"{stem}_回答沉淀.md"
    if target.exists():
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        target = out_dir / f"{stem}_回答沉淀_{ts}.md"

    rows: List[str] = []
    rows.append("---")
    rows.append(f"title: {question or '高价值回答沉淀'}")
    rows.append(f"approved_at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    rows.append("approved_by: human")
    rel_candidate = candidate_file.relative_to(wiki_root).as_posix()
    rows.append(f"source_candidate: {rel_candidate}")
    rows.append("status: approved")
    rows.append("---")
    rows.append("")
    rows.append(raw.strip())
    rows.append("")
    target.write_text("\n".join(rows), encoding="utf-8")

    # 触发增量编译，刷新 index/log/entities/concepts
    compile_result = build_reference_wiki.invoke({"limit": 120, "force": False})
    return {
        "ok": True,
        "candidate": rel_candidate,
        "output_path": target.relative_to(wiki_root).as_posix(),
        "compile_result": compile_result,
    }


def run_wiki_command(raw_query: str | None = None) -> None:
    query = str(raw_query or "").strip()
    if not query:
        query = Prompt.ask("请输入 wiki 问题").strip()
    if not query:
        console.print("[yellow]/wiki 需要问题文本，例如：/wiki 什么是舆情反转？[/yellow]")
        return

    root = Path(__file__).resolve().parents[1]
    env_file = root / ".env"
    if env_file.is_file():
        load_dotenv(env_file, override=True)
    reload_env_config()
    get_env_config()
    result = answer_wiki_query(query, topk=6, style="teach", project_root=Path(__file__).resolve().parents[1])
    meta = result.get("_wiki_meta") if isinstance(result.get("_wiki_meta"), dict) else {}
    sources = result.get("sources") if isinstance(result.get("sources"), list) else []
    graph_route = meta.get("wiki_route") == "neo4j_qa_only"

    _render_wiki_header(query, meta, source_count=len(sources))
    answer_text = _strip_template_answer_sections(str(result.get("answer", "")))
    _render_wiki_answer(query, answer_text)
    _render_wiki_sources(sources, graph_route=graph_route)
    console.print()


def run_wiki_approve_command(raw_selector: str | None = None) -> None:
    """
    审核并回流候选沉淀到正式 output。

    用法：
    - /wiki-approve            -> 默认审核最新候选
    - /wiki-approve 关键词      -> 按文件名模糊匹配候选
    """
    root = Path(__file__).resolve().parents[1]
    env_file = root / ".env"
    if env_file.is_file():
        load_dotenv(env_file, override=True)
    reload_env_config()
    get_env_config()

    selector = str(raw_selector or "").strip()
    candidate = _pick_candidate(root, selector=selector if selector else None)
    if candidate is None:
        console.print("[yellow]未找到可审批候选。请先通过 /wiki 生成高价值候选。[/yellow]")
        return

    console.print(f"[cyan]待审批候选: {candidate.name}[/cyan]")
    result = _approve_candidate_to_output(root, candidate)
    console.print("[green]✓ 已完成候选审批并回流到正式 output[/green]")
    console.print(f"[green]  候选: {result.get('candidate','')}[/green]")
    console.print(f"[green]  正式页: {result.get('output_path','')}[/green]")

