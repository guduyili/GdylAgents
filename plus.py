"""Tools for the software testing agent (tester_agent)."""

from __future__ import annotations

import csv
import io
import re
from datetime import datetime
from typing import Annotated, Any

from langchain_core.tools import tool
from langchain_core.tools.base import InjectedToolCallId
from langgraph.types import Command


def _make_file_data(content: str) -> dict:
    """构造 LangGraph FileData 格式（与 StateBackend.write 返回格式一致）。"""
    now = datetime.utcnow().isoformat() + "Z"
    return {
        "content": content.splitlines(keepends=True),
        "created_at": now,
        "modified_at": now,
    }


@tool
def format_test_cases(
    test_cases_markdown: str,
    project_name: str,
    requirements_source: str = "用户提供的需求文档",
) -> str:
    """将测试用例内容格式化为标准 Markdown 文档并返回完整文档字符串。

    Args:
        test_cases_markdown: 原始测试用例 Markdown 内容（含各模块和 TC-XXX 条目）。
        project_name: 项目或功能名称，用于文档标题。
        requirements_source: 需求来源描述。

    Returns:
        格式化后的完整测试用例 Markdown 文档字符串。
    """
    today = datetime.now().strftime("%Y-%m-%d")

    tc_ids: list[Any] = re.findall(r"TC-\d+", test_cases_markdown)
    total = len(tc_ids)

    p0 = len(re.findall(r"\*\*优先级\*\*:\s*P0", test_cases_markdown))
    p1 = len(re.findall(r"\*\*优先级\*\*:\s*P1", test_cases_markdown))
    p2 = len(re.findall(r"\*\*优先级\*\*:\s*P2", test_cases_markdown))
    p3 = len(re.findall(r"\*\*优先级\*\*:\s*P3", test_cases_markdown))

    modules = re.findall(r"^### 模块[：:]\s*(.+)$", test_cases_markdown, re.MULTILINE)
    if not modules:
        modules = re.findall(r"^## (.+)$", test_cases_markdown, re.MULTILINE)

    module_summary_rows = ""
    for mod in modules:
        mod_cases = re.findall(
            rf"TC-\d+.*?模块[：:]\s*{re.escape(mod)}", test_cases_markdown, re.DOTALL
        )
        module_summary_rows += f"| {mod} | {len(mod_cases)} | - | - | - | - |\n"

    header = f"""# 测试用例文档

## 项目概述

| 项目       | 内容                     |
|------------|--------------------------|
| 项目名称   | {project_name}           |
| 需求来源   | {requirements_source}    |
| 生成日期   | {today}                  |
| 测试用例总数 | {total}                |
| P0（冒烟） | {p0}                     |
| P1（核心） | {p1}                     |
| P2（扩展） | {p2}                     |
| P3（边界） | {p3}                     |

## 功能模块一览

| 模块 | 用例数 | P0 | P1 | P2 | P3 |
|------|--------|----|----|----|----|
{module_summary_rows if module_summary_rows else f"| （见下文详情） | {total} | {p0} | {p1} | {p2} | {p3} |\n"}
---

## 测试用例详情

"""
    return header + test_cases_markdown


@tool
def extract_requirement_sections(requirement_text: str) -> str:
    """从需求文档中提取结构化的功能模块和需求条目。

    Args:
        requirement_text: 原始需求文档文本内容。

    Returns:
        结构化的模块和需求条目字符串，供分析子 Agent 使用。
    """
    lines = requirement_text.strip().split("\n")
    sections: dict[str, list[str]] = {}
    current_section = "通用需求"

    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        if re.match(r"^#{1,2}\s+", stripped):
            current_section = re.sub(r"^#+\s+", "", stripped)
            if current_section not in sections:
                sections[current_section] = []
        elif re.match(r"^#{3,}\s+", stripped):
            sub = re.sub(r"^#+\s+", "", stripped)
            sections.setdefault(current_section, []).append(f"[子功能] {sub}")
        elif re.match(r"^\d+[\.\)]\s+", stripped):
            sections.setdefault(current_section, []).append(stripped)
        elif re.match(r"^[-*+]\s+", stripped):
            sections.setdefault(current_section, []).append(stripped)
        elif re.match(r"^(GET|POST|PUT|PATCH|DELETE)\s+/", stripped):
            sections.setdefault(current_section, []).append(f"[接口] {stripped}")

    if not sections:
        return f"未能从文档中识别出结构化模块，原始内容如下：\n\n{requirement_text}"

    output_parts = ["# 提取的需求模块\n"]
    for section, items in sections.items():
        output_parts.append(f"\n## {section}\n")
        if items:
            for item in items:
                output_parts.append(f"- {item}")
        else:
            output_parts.append("- （本模块无明确条目，请根据标题推断需求）")

    return "\n".join(output_parts)


@tool
def count_coverage(
    requirements_text: str,
    test_cases_text: str,
) -> str:
    """统计测试用例对需求的覆盖情况。

    Args:
        requirements_text: 需求文档文本。
        test_cases_text: 测试用例文档文本。

    Returns:
        覆盖率分析报告字符串。
    """
    req_items = re.findall(r"^\s*[-*+\d][\.\)]\s+.+", requirements_text, re.MULTILINE)
    req_count = max(len(req_items), 1)
    tc_count = len(re.findall(r"TC-\d+", test_cases_text))
    positive = len(re.findall(r"用例类型.*?正向", test_cases_text))
    negative = len(re.findall(r"用例类型.*?负向", test_cases_text))
    boundary = len(re.findall(r"用例类型.*?边界", test_cases_text))
    exception = len(re.findall(r"用例类型.*?异常", test_cases_text))
    coverage_ratio = min(tc_count / req_count, 1.0) * 100

    return f"""# 测试覆盖率报告

| 指标              | 数值          |
|-------------------|---------------|
| 需求条目数（估算） | {req_count}  |
| 测试用例总数       | {tc_count}   |
| 平均覆盖比         | {coverage_ratio:.1f}% |
| 正向测试用例       | {positive}   |
| 负向测试用例       | {negative}   |
| 边界值测试用例     | {boundary}   |
| 异常测试用例       | {exception}  |

## 建议

{"✅ 覆盖率良好，建议重点检查 P0/P1 用例的完整性。" if coverage_ratio >= 80 else "⚠️ 覆盖率不足 80%，建议补充更多负向和边界值测试用例。"}
"""


def _parse_test_cases(test_cases_text: str) -> list[dict[str, str]]:
    """从 Markdown 格式的测试用例文本中解析出结构化数据。"""
    records: list[dict[str, str]] = []
    blocks = re.split(r"(?=####\s+TC-\d+)", test_cases_text)
    for block in blocks:
        block = block.strip()
        if not block or not re.match(r"####\s+TC-\d+", block):
            continue
        record: dict[str, str] = {}
        m = re.match(r"####\s+(TC-\d+)[：:]\s*(.+)", block)
        if m:
            record["用例ID"] = m.group(1)
            record["标题"] = m.group(2).strip()
        else:
            m2 = re.match(r"####\s+(TC-\d+)\s*(.+)?", block)
            if m2:
                record["用例ID"] = m2.group(1)
                record["标题"] = (m2.group(2) or "").strip()
        for field, pattern in [
            ("模块", r"\*\*模块\*\*[：:]\s*(.+)"),
            ("前置条件", r"\*\*前置条件\*\*[：:]\s*(.+)"),
            ("预期结果", r"\*\*预期结果\*\*[：:]\s*(.+)"),
            ("优先级", r"\*\*优先级\*\*[：:]\s*(.+)"),
            ("用例类型", r"\*\*用例类型\*\*[：:]\s*(.+)"),
        ]:
            fm = re.search(pattern, block)
            record[field] = fm.group(1).strip() if fm else ""
        steps_m = re.search(r"\*\*测试步骤\*\*[：:](.+?)(?=\n\s*\*\*|\Z)", block, re.DOTALL)
        if steps_m:
            steps = re.sub(r"\s+", " ", steps_m.group(1)).strip()
            record["测试步骤"] = steps
        else:
            record["测试步骤"] = ""
        if record.get("用例ID"):
            records.append(record)
    return records


@tool
def export_to_csv(
    test_cases_text: str,
    tool_call_id: Annotated[str, InjectedToolCallId],
) -> Command:
    """将 Markdown 格式的测试用例导出为 CSV 字符串，并自动写入 /test_cases.csv 文件状态。

    Args:
        test_cases_text: 标准格式的测试用例 Markdown 文本。

    Returns:
        Command：自动将 CSV 写入 files["/test_cases.csv"]，同时返回 CSV 内容供 Agent 查看。
    """
    from langchain_core.messages import ToolMessage

    records = _parse_test_cases(test_cases_text)
    if not records:
        msg = "未能解析出任何测试用例，请确认格式包含 '#### TC-XXX: 标题' 条目。"
        return Command(update={"messages": [ToolMessage(content=msg, tool_call_id=tool_call_id)]})

    fieldnames = ["用例ID", "标题", "模块", "前置条件", "测试步骤", "预期结果", "优先级", "用例类型"]
    output = io.StringIO()
    output.write("\ufeff")
    writer = csv.DictWriter(
        output,
        fieldnames=fieldnames,
        extrasaction="ignore",
        lineterminator="\n",
    )
    writer.writeheader()
    writer.writerows(records)
    csv_content = output.getvalue()

    result_msg = f"✅ CSV 已自动写入 /test_cases.csv（共 {len(records)} 条用例）"
    return Command(
        update={
            "files": {"/test_cases.csv": _make_file_data(csv_content)},
            "messages": [ToolMessage(content=result_msg, tool_call_id=tool_call_id)],
        }
    )


@tool
def export_to_markdown(
    test_cases_text: str,
    project_name: str = "项目",
    tool_call_id: Annotated[str, InjectedToolCallId] = "",
) -> Command:
    """将测试用例整理为标准 Markdown 文档，并自动写入 /test_cases_export.md 文件状态。

    Args:
        test_cases_text: 标准格式的测试用例 Markdown 文本。
        project_name: 项目名称，用于文档标题。

    Returns:
        Command：自动将 Markdown 写入 files["/test_cases_export.md"]。
    """
    from langchain_core.messages import ToolMessage

    records = _parse_test_cases(test_cases_text)
    today = datetime.now().strftime("%Y-%m-%d")
    total = len(records)
    p_counts: dict[str, int] = {"P0": 0, "P1": 0, "P2": 0, "P3": 0}
    for r in records:
        p = r.get("优先级", "").strip()
        if p in p_counts:
            p_counts[p] += 1

    header = f"""# {project_name} — 测试用例文档

> 生成日期：{today}　|　用例总数：{total}　|　P0: {p_counts['P0']}　P1: {p_counts['P1']}　P2: {p_counts['P2']}　P3: {p_counts['P3']}

---

"""
    md_content = header + test_cases_text
    result_msg = f"✅ Markdown 已自动写入 /test_cases_export.md（共 {total} 条用例）"
    return Command(
        update={
            "files": {"/test_cases_export.md": _make_file_data(md_content)},
            "messages": [ToolMessage(content=result_msg, tool_call_id=tool_call_id)],
        }
    )


@tool
def export_to_mindmap(
    test_cases_text: str,
    project_name: str = "测试用例脑图",
    tool_call_id: Annotated[str, InjectedToolCallId] = "",
) -> Command:
    """将测试用例转换为 markmap 可直接渲染的 Markdown 脑图格式，并自动写入 /mindmap.md 文件状态。

    按「模块 → 用例类型 → TC-XXX 标题 [优先级]」三层结构生成脑图 Markdown。
    自动写入后可在前端 /mindmap 页面用 markmap 渲染，无需 Agent 额外调用 write_file。

    Args:
        test_cases_text: 标准格式的测试用例 Markdown 文本。
        project_name: 脑图根节点名称。

    Returns:
        Command：自动将脑图写入 files["/mindmap.md"]。
    """
    from langchain_core.messages import ToolMessage

    records = _parse_test_cases(test_cases_text)
    if not records:
        msg = "未能解析出任何测试用例，请确认格式包含 '#### TC-XXX: 标题' 条目。"
        return Command(update={"messages": [ToolMessage(content=msg, tool_call_id=tool_call_id)]})

    tree: dict[str, dict[str, list[dict[str, str]]]] = {}
    for r in records:
        mod = r.get("模块") or "未分类"
        typ = r.get("用例类型") or "其他"
        tree.setdefault(mod, {}).setdefault(typ, []).append(r)

    today = datetime.now().strftime("%Y-%m-%d")
    total = len(records)

    lines = [
        f"# {project_name}",
        f"",
        f"## 📊 概览",
        f"",
        f"- 用例总数：{total}",
        f"- 生成日期：{today}",
        f"- P0：{sum(1 for r in records if r.get('优先级') == 'P0')}　"
        f"P1：{sum(1 for r in records if r.get('优先级') == 'P1')}　"
        f"P2：{sum(1 for r in records if r.get('优先级') == 'P2')}　"
        f"P3：{sum(1 for r in records if r.get('优先级') == 'P3')}",
        f"",
    ]

    priority_mark = {"P0": "🔴", "P1": "🟠", "P2": "🟡", "P3": "🟢"}

    for mod, type_map in tree.items():
        lines.append(f"## 📁 {mod}")
        lines.append("")
        for typ, cases in type_map.items():
            lines.append(f"### {typ}")
            lines.append("")
            for r in cases:
                tc_id = r.get("用例ID", "")
                title = r.get("标题", "")
                priority = r.get("优先级", "")
                mark = priority_mark.get(priority, "⚪")
                lines.append(f"- {mark} **{tc_id}** {title} `[{priority}]`")
            lines.append("")

    mindmap_content = "\n".join(lines)
    result_msg = f"✅ 脑图已自动写入 /mindmap.md（共 {total} 条用例，可在 /mindmap 页面查看）"
    return Command(
        update={
            "files": {"/mindmap.md": _make_file_data(mindmap_content)},
            "messages": [ToolMessage(content=result_msg, tool_call_id=tool_call_id)],
        }
    )
