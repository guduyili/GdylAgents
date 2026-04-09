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

    # P0 自动校正：如果没有 P0 用例，将第一条正向（P1）用例自动升级为 P0
    p0_warning = ""
    if p0 == 0 and total > 0:
        # 找第一条 P1 正向用例的位置，将其优先级改为 P0
        upgraded = False
        def _upgrade_first_p1(m: re.Match) -> str:
            nonlocal upgraded
            if not upgraded:
                upgraded = True
                return m.group(0).replace("**优先级**: P1", "**优先级**: P0")
            return m.group(0)

        # 匹配一整条用例块（从 #### TC- 到下一条 #### TC- 或文末）
        test_cases_markdown = re.sub(
            r"(#### TC-\d+:.*?(?=#### TC-|\Z))",
            _upgrade_first_p1,
            test_cases_markdown,
            flags=re.DOTALL,
        )
        if upgraded:
            p0 = 1
            p1 = max(p1 - 1, 0)
            p0_warning = "\n> ⚠️ **P0 自动校正**：原文档无 P0 冒烟用例，已将第一条 P1 正向用例自动升级为 P0。\n"

    modules = re.findall(r"^### 模块[：:]\s*(.+)$", test_cases_markdown, re.MULTILINE)
    if not modules:
        modules = re.findall(r"^## (.+)$", test_cases_markdown, re.MULTILINE)

    module_summary_rows = ""
    for mod in modules:
        mod_cases = re.findall(
            rf"TC-\d+.*?模块[：:]\s*{re.escape(mod)}", test_cases_markdown, re.DOTALL
        )
        module_summary_rows += f"| {mod} | {len(mod_cases)} | - | - | - | - |\n"

    if not module_summary_rows:
        module_summary_rows = f"| （见下文详情） | {total} | {p0} | {p1} | {p2} | {p3} |\n"

    header = f"""# 测试用例文档
{p0_warning}
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
{module_summary_rows}
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


# ============================================================================
# 新增：需求文档格式检测工具
# ============================================================================

@tool
def detect_doc_format(requirement_text: str) -> str:
    """检测需求文档的格式类型，并给出针对性的解析建议。

    支持识别以下格式：
    - Markdown（含 ## / ### 标题层级）
    - Confluence 导出（含 h1. / h2. 或 {code} 块）
    - API 接口文档（以 GET/POST/PUT/DELETE 路径为主）
    - 用户故事格式（含 As a / Given / When / Then）
    - 纯文本/列表格式
    - 结构化表格格式

    Args:
        requirement_text: 原始需求文档文本内容。

    Returns:
        格式检测报告，包含识别到的格式类型、置信度和解析策略建议。
    """
    text = requirement_text.strip()
    scores: dict[str, int] = {
        "markdown": 0,
        "confluence": 0,
        "api_doc": 0,
        "user_story": 0,
        "plain_list": 0,
        "table": 0,
    }

    # Markdown 特征
    if re.search(r"^#{1,4}\s+\S", text, re.MULTILINE):
        scores["markdown"] += 3
    if re.search(r"^\s*[-*+]\s+\S", text, re.MULTILINE):
        scores["markdown"] += 1
    if re.search(r"\*\*\S+\*\*", text):
        scores["markdown"] += 1
    if re.search(r"```", text):
        scores["markdown"] += 2

    # Confluence 特征
    if re.search(r"^h[1-6]\.\s+\S", text, re.MULTILINE):
        scores["confluence"] += 4
    if re.search(r"\{code\}|\{panel\}|\{info\}|\{note\}", text):
        scores["confluence"] += 3
    if re.search(r"AC\d+:|验收标准\d+:|Acceptance Criteria", text, re.IGNORECASE):
        scores["confluence"] += 2

    # API 接口文档特征
    api_matches = re.findall(
        r"^(GET|POST|PUT|PATCH|DELETE)\s+/\S+", text, re.MULTILINE | re.IGNORECASE
    )
    scores["api_doc"] += len(api_matches) * 2
    if re.search(r"请求[参数体]|响应[体结构]|状态码|Content-Type|Authorization", text):
        scores["api_doc"] += 2
    if re.search(r"```json|application/json", text):
        scores["api_doc"] += 1

    # 用户故事格式
    if re.search(r"As an?\s+\w+|作为\s*\w+", text, re.IGNORECASE):
        scores["user_story"] += 3
    if re.search(r"\bGiven\b|\bWhen\b|\bThen\b|假设|当|那么", text):
        scores["user_story"] += 2
    if re.search(r"用户故事|User Story|场景[:：]", text):
        scores["user_story"] += 2

    # 纯列表格式
    numbered = len(re.findall(r"^\s*\d+[\.\)]\s+\S", text, re.MULTILINE))
    scores["plain_list"] += min(numbered, 5)

    # 表格格式
    if re.search(r"\|.+\|.+\|", text):
        scores["table"] += 3
    if re.search(r"^[-|]+$", text, re.MULTILINE):
        scores["table"] += 1

    # 识别主格式
    detected = max(scores, key=lambda k: scores[k])
    max_score = scores[detected]
    total_score = sum(scores.values()) or 1
    confidence = min(int(max_score / total_score * 100 + 0.5), 95)

    # 各格式解析策略
    strategies = {
        "markdown": (
            "**Markdown 格式**\n\n"
            "解析策略：\n"
            "- 以 `##` / `###` 标题作为功能模块边界\n"
            "- 以 `- ` / `* ` 列表项提取具体需求条目\n"
            "- 代码块（` ``` `）中的内容识别为接口示例或数据结构\n"
            "- 建议直接使用 `extract_requirement_sections` 工具处理"
        ),
        "confluence": (
            "**Confluence 导出格式**\n\n"
            "解析策略：\n"
            "- `h1.` / `h2.` 标题对应功能模块\n"
            "- `{code}` 块通常是接口请求/响应示例\n"
            "- `AC1:` / `验收标准` 行是可测试的验收条件，优先提取\n"
            "- 建议先将 Confluence 格式标题转为 Markdown `##` 再处理"
        ),
        "api_doc": (
            "**API 接口文档格式**\n\n"
            "解析策略：\n"
            "- 每个 `METHOD /path` 是一个独立测试单元\n"
            "- 重点生成：正常请求、参数缺失、参数格式错误、鉴权、状态码四类用例\n"
            "- 建议同时委派 `api-test-writer` 子 Agent 专项处理\n"
            f"- 检测到 {len(api_matches)} 个接口端点"
        ),
        "user_story": (
            "**用户故事（BDD）格式**\n\n"
            "解析策略：\n"
            "- `As a / 作为` 确定用户角色\n"
            "- `Given / 假设` → 测试用例前置条件\n"
            "- `When / 当` → 测试步骤\n"
            "- `Then / 那么` → 预期结果\n"
            "- 每个 Scenario 直接对应一条测试用例"
        ),
        "plain_list": (
            "**纯文本/列表格式**\n\n"
            "解析策略：\n"
            "- 编号列表（1. / 2.）视为独立需求条目\n"
            "- 建议先让 `requirement-analyzer` 子 Agent 进行语义分组\n"
            "- 无明确模块时统一归入「通用功能」模块"
        ),
        "table": (
            "**结构化表格格式**\n\n"
            "解析策略：\n"
            "- 表头字段通常对应测试维度（如：功能点、输入、预期输出）\n"
            "- 每行记录是一个潜在测试场景\n"
            "- 建议按表格行逐条生成测试用例"
        ),
    }

    # 检测文档基本统计
    lines_count = len(text.splitlines())
    chars_count = len(text)
    api_count = len(api_matches)

    report = f"""# 需求文档格式检测报告

## 检测结论

| 项目 | 结果 |
|------|------|
| **识别格式** | {strategies[detected].split(chr(10))[0].replace("**", "")} |
| **置信度** | {confidence}% |
| **文档行数** | {lines_count} 行 |
| **文档字符数** | {chars_count} 字符 |
| **API 端点数** | {api_count} 个 |

## 格式评分详情

| 格式类型 | 得分 |
|---------|------|
| Markdown | {scores['markdown']} |
| Confluence | {scores['confluence']} |
| API 接口文档 | {scores['api_doc']} |
| 用户故事/BDD | {scores['user_story']} |
| 纯列表 | {scores['plain_list']} |
| 表格 | {scores['table']} |

## 推荐解析策略

{strategies[detected]}

## 后续建议

{"⚠️ 文档较短（< 50 行），建议请求用户补充更详细的需求描述。" if lines_count < 50 else "✅ 文档长度适中，可直接进入需求提取阶段。"}
{"🔌 检测到 API 接口，建议同时委派 api-test-writer 子 Agent 生成接口测试用例。" if api_count > 0 else ""}
"""
    return report


# ============================================================================
# 新增：测试用例自我审查与增量补充工具
# ============================================================================

@tool
def review_test_cases(
    test_cases_text: str,
    requirements_text: str = "",
    focus_module: str = "",
) -> str:
    """对已生成的测试用例进行自我审查，识别覆盖盲点并给出增量补充建议。

    与 test-case-reviewer 子 Agent 的区别：
    - reviewer 子 Agent：全面质量审查（格式、步骤可执行性、预期结果等）
    - 本工具：快速覆盖率盲点分析 + 增量补充建议，适合迭代优化场景

    Args:
        test_cases_text: 已生成的测试用例 Markdown 文本。
        requirements_text: 原始需求文档文本（可选，用于需求对比分析）。
        focus_module: 指定重点检查的模块名称（可选，为空则检查全部模块）。

    Returns:
        审查报告：包含覆盖盲点列表、优先级分布分析和增量补充建议。
    """
    records = _parse_test_cases(test_cases_text)
    if not records:
        return "⚠️ 未能解析出任何测试用例，请确认格式包含 '#### TC-XXX: 标题' 条目。"

    # 按模块分组
    modules: dict[str, list[dict]] = {}
    for r in records:
        mod = r.get("模块") or "未分类"
        modules.setdefault(mod, []).append(r)

    # 优先级分布
    priority_dist: dict[str, int] = {"P0": 0, "P1": 0, "P2": 0, "P3": 0, "未设置": 0}
    for r in records:
        p = r.get("优先级", "").strip()
        if p in priority_dist:
            priority_dist[p] += 1
        else:
            priority_dist["未设置"] += 1

    # 用例类型分布
    type_dist: dict[str, int] = {"正向": 0, "负向": 0, "边界值": 0, "异常": 0, "其他": 0}
    for r in records:
        t = r.get("用例类型", "").strip()
        if t in type_dist:
            type_dist[t] += 1
        else:
            type_dist["其他"] += 1

    total = len(records)

    # 覆盖盲点检测
    gaps: list[str] = []
    suggestions: list[str] = []

    # 检查 P0 冒烟用例
    if priority_dist["P0"] == 0:
        gaps.append("❌ 缺少 P0 冒烟测试用例（核心功能最基础的验证）")
        suggestions.append("补充至少 1 条 P0 冒烟用例，覆盖最核心的功能路径")

    # 检查负向测试
    if type_dist["负向"] == 0:
        gaps.append("❌ 缺少负向测试用例（无效输入、权限拒绝场景）")
        suggestions.append("补充负向用例：空值输入、格式错误、越权访问等场景")
    elif type_dist["负向"] / total < 0.15:
        gaps.append(f"⚠️ 负向测试用例占比偏低（{type_dist['负向']}/{total}={type_dist['负向']/total*100:.0f}%，建议 ≥15%）")

    # 检查边界值测试
    if type_dist["边界值"] == 0:
        gaps.append("❌ 缺少边界值测试用例（最大值、最小值、临界值）")
        suggestions.append("补充边界值用例：字段长度上限/下限、数值边界、空列表等")

    # 检查异常测试
    if type_dist["异常"] == 0:
        gaps.append("⚠️ 缺少异常测试用例（网络超时、服务不可用等）")
        suggestions.append("可选：补充异常用例覆盖网络超时、第三方服务不可用场景")

    # 检查优先级分布合理性
    if total > 5:
        p1_ratio = priority_dist["P1"] / total
        if p1_ratio < 0.3:
            gaps.append(f"⚠️ P1 核心用例占比偏低（{priority_dist['P1']}/{total}={p1_ratio*100:.0f}%，建议 ≥30%）")
        if priority_dist["P2"] == 0 and total > 10:
            gaps.append("⚠️ 缺少 P2 扩展场景用例")

    # 检查用例完整性（字段缺失）
    incomplete = [
        r["用例ID"] for r in records
        if not r.get("前置条件") or not r.get("测试步骤") or not r.get("预期结果")
    ]
    if incomplete:
        gaps.append(f"❌ 以下用例字段不完整：{', '.join(incomplete[:5])}" + ("..." if len(incomplete) > 5 else ""))
        suggestions.append("补全上述用例的前置条件、测试步骤或预期结果")

    # 模块覆盖对比（如果提供了需求文档）
    req_modules: list[str] = []
    if requirements_text:
        req_modules = re.findall(r"^#{1,2}\s+(.+)$", requirements_text, re.MULTILINE)
        req_modules = [m.strip() for m in req_modules if m.strip()]
        tested_modules = set(modules.keys())
        for rm in req_modules:
            if not any(rm in tm or tm in rm for tm in tested_modules):
                gaps.append(f"⚠️ 需求模块「{rm}」可能未被测试用例覆盖")
                suggestions.append(f"检查并补充「{rm}」模块的测试用例")

    # 聚焦模块分析
    focus_report = ""
    if focus_module and focus_module in modules:
        focus_cases = modules[focus_module]
        focus_types = {t: 0 for t in ["正向", "负向", "边界值", "异常"]}
        for r in focus_cases:
            t = r.get("用例类型", "").strip()
            if t in focus_types:
                focus_types[t] += 1
        missing_types = [t for t, cnt in focus_types.items() if cnt == 0]
        focus_report = f"""
## 聚焦模块分析：{focus_module}

| 用例类型 | 数量 |
|---------|------|
{"".join(f"| {t} | {cnt} |" + chr(10) for t, cnt in focus_types.items())}

{"**缺失类型**: " + "、".join(missing_types) if missing_types else "✅ 该模块四种测试类型均有覆盖"}
"""

    # 生成增量补充建议（具体可操作的指令）
    increment_suggestions = ""
    if suggestions:
        suggestion_lines = "\n".join(f"{i+1}. {s}" for i, s in enumerate(suggestions))
        increment_suggestions = f"""
## 增量补充建议

以下建议可直接传给 `test-case-writer` 子 Agent 执行：

{suggestion_lines}
"""

    # 最终评分
    gap_count = len([g for g in gaps if g.startswith("❌")])
    warn_count = len([g for g in gaps if g.startswith("⚠️")])
    score = max(10 - gap_count * 2 - warn_count, 1)

    gaps_str = "\n".join(f"- {g}" for g in gaps) if gaps else "- ✅ 未发现明显覆盖盲点"

    report = f"""# 测试用例自我审查报告

## 总体评分：{score}/10

## 统计概览

| 指标 | 数值 |
|------|------|
| 用例总数 | {total} |
| 模块数量 | {len(modules)} |
| P0 冒烟 | {priority_dist['P0']} |
| P1 核心 | {priority_dist['P1']} |
| P2 扩展 | {priority_dist['P2']} |
| P3 边界 | {priority_dist['P3']} |
| 正向用例 | {type_dist['正向']} |
| 负向用例 | {type_dist['负向']} |
| 边界值用例 | {type_dist['边界值']} |
| 异常用例 | {type_dist['异常']} |
| 字段不完整用例 | {len(incomplete)} |

## 覆盖盲点

{gaps_str}
{focus_report}
{increment_suggestions}
## 结论

{"✅ 测试用例质量良好，可以进入导出阶段。" if score >= 7 else "⚠️ 建议根据上述增量补充建议完善后再导出。"}
"""
    return report


# ============================================================================
# 新增：增量追加测试用例工具
# ============================================================================

@tool
def append_test_cases(
    existing_test_cases: str,
    new_test_cases: str,
    module_name: str = "",
) -> str:
    """将新生成的测试用例增量追加到已有测试用例文档中，自动重编 TC-ID。

    适用场景：
    - 用户说"补充某个模块的边界测试"
    - 审查后需要新增缺失场景
    - 迭代优化时追加新用例，不重新生成全部

    Args:
        existing_test_cases: 已有的测试用例 Markdown 文本（含 TC-001、TC-002...）。
        new_test_cases: 新生成的测试用例 Markdown 文本（TC-ID 可能与已有的重复）。
        module_name: 新用例所属模块名（可选，用于在追加时添加模块标题）。

    Returns:
        合并后的测试用例 Markdown 文本，TC-ID 已自动重编为连续序号。
    """
    # 找出现有最大 TC-ID
    existing_ids = re.findall(r"TC-(\d+)", existing_test_cases)
    max_id = max((int(n) for n in existing_ids), default=0)

    # 对新用例重编 ID（从 max_id + 1 开始）
    counter = [max_id]

    def _renumber(m: re.Match) -> str:
        counter[0] += 1
        return f"TC-{counter[0]:03d}"

    renumbered = re.sub(r"TC-\d+", _renumber, new_test_cases)

    # 如果指定了模块名且新用例没有模块标题，加一个
    if module_name:
        has_heading = bool(re.search(
            rf"^###\s+.*{re.escape(module_name)}", renumbered, re.MULTILINE
        ))
        if not has_heading:
            renumbered = f"\n### 模块：{module_name}（增量补充）\n\n" + renumbered.lstrip()

    # 合并：在已有用例末尾追加
    combined = existing_test_cases.rstrip() + "\n\n" + renumbered.strip() + "\n"

    # 统计
    total_new = len(re.findall(r"TC-\d+", renumbered))
    total_all = len(re.findall(r"TC-\d+", combined))

    return (
        f"✅ 已追加 {total_new} 条新用例（TC-{max_id+1:03d} ~ TC-{counter[0]:03d}），"
        f"合并后共 {total_all} 条。\n\n"
        f"---\n\n"
        + combined
    )
