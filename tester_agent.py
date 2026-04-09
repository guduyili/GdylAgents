"""
软件测试 Agent — 供 langgraph dev 前端使用
==========================================

将 gdyltester 的功能集成到 gdylresearch 多 Agent 服务中。
前端选择 "tester" assistant 即可使用。

工作流程：
  1. 用户提供需求文档（文本 / Markdown / 接口描述）
  2. 主 Agent 使用 write_todos 拆解测试任务
  3. 委派 requirement-analyzer 子 Agent 提取测试点
  4. 委派 test-case-writer 子 Agent 编写完整测试用例（含 API 接口则委派 api-test-writer）
  5. 委派 test-case-reviewer 子 Agent 审查用例质量
  6. 使用 format_test_cases 工具格式化输出并保存 /test_cases.md
  7. 使用 export_to_csv / export_to_markdown 导出文件
  8. 使用 count_coverage 工具验证覆盖率
"""

from __future__ import annotations

import os
import sys
from datetime import datetime
from pathlib import Path as _Path

sys.path.insert(0, str(_Path(__file__).parent.parent / "shared"))
from custom_model import StreamFixChatOpenAI
from deepagents import create_deep_agent
from dotenv import load_dotenv
from tester_agent import (
    API_TEST_WRITER_INSTRUCTIONS,
    REQUIREMENT_ANALYZER_INSTRUCTIONS,
    TEST_CASE_REVIEWER_INSTRUCTIONS,
    TEST_CASE_WRITER_INSTRUCTIONS,
    TEST_WORKFLOW_INSTRUCTIONS,
    append_test_cases,
    count_coverage,
    detect_doc_format,
    export_to_csv,
    export_to_markdown,
    export_to_mindmap,
    extract_requirement_sections,
    format_test_cases,
    review_test_cases,
)

load_dotenv()

# ============================================================================
# 主 Agent 系统提示
# ============================================================================

current_date = datetime.now().strftime("%Y-%m-%d")
INSTRUCTIONS = TEST_WORKFLOW_INSTRUCTIONS + f"\n\n> 当前日期：{current_date}\n"

# ============================================================================
# 子 Agent 定义
# ============================================================================

requirement_analyzer = {
    "name": "requirement-analyzer",
    "description": (
        "专门从需求文档中提取可测试的功能点和验证条件。"
        "当你需要分析需求文档、识别测试点时，委派给此 Agent。"
        "输入：需求文档原文。输出：按模块分组的测试点列表。"
    ),
    "system_prompt": REQUIREMENT_ANALYZER_INSTRUCTIONS,
}

test_case_writer = {
    "name": "test-case-writer",
    "description": (
        "专门将测试点转化为完整、可执行的功能测试用例。"
        "当你需要编写测试步骤、前置条件、预期结果时，委派给此 Agent。"
        "输入：测试点列表（由 requirement-analyzer 输出）。"
        "输出：标准格式的测试用例 Markdown 文本。"
    ),
    "system_prompt": TEST_CASE_WRITER_INSTRUCTIONS,
}

test_case_reviewer = {
    "name": "test-case-reviewer",
    "description": (
        "专门审查测试用例的质量和完整性，检查格式合规、覆盖完整性、步骤可执行性和预期结果可验证性。"
        "在 test-case-writer 生成用例后，委派给此 Agent 进行审查。"
        "输入：完整的测试用例 Markdown 文本。输出：审查报告和修改建议。"
    ),
    "system_prompt": TEST_CASE_REVIEWER_INSTRUCTIONS,
}

api_test_writer = {
    "name": "api-test-writer",
    "description": (
        "专门编写 REST API 接口测试用例，覆盖正常请求、参数校验、鉴权和错误码。"
        "当需求中包含 API 接口描述（GET/POST/PUT/DELETE）时，委派给此 Agent。"
        "输入：接口文档或接口需求描述。"
        "输出：包含请求体、响应体、状态码验证的完整接口测试用例。"
    ),
    "system_prompt": API_TEST_WRITER_INSTRUCTIONS,
}

# ============================================================================
# 模型 & Agent
# ============================================================================

model = StreamFixChatOpenAI(
    model="gpt-5.4-mini",
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url="https://code.swpumc.cn/v1",
    temperature=0,
    timeout=300,
    max_retries=2,
)

agent = create_deep_agent(
    model=model,
    tools=[
        format_test_cases,
        extract_requirement_sections,
        count_coverage,
        export_to_csv,
        export_to_markdown,
        export_to_mindmap,
        detect_doc_format,
        review_test_cases,
        append_test_cases,
    ],
    system_prompt=INSTRUCTIONS,
    subagents=[requirement_analyzer, test_case_writer, test_case_reviewer, api_test_writer],
)
