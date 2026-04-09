"""
工作流程：
  1. 用户提出研究问题
  2. 主 Agent 使用 write_todos 创建研究计划
  3. 主 Agent 将研究任务委派给 research-agent 子Agent（可并行）
  4. 每个子 Agent 独立进行网络搜索和反思
  5. 主 Agent 收集所有子 Agent 的结果
  6. 主 Agent 综合所有结果，写出最终报告

关键设计决策：
  - 主 Agent 不直接做研究，只做编排和综合
  - 子 Agent 有独立的上下文窗口，可以深入研究而不占用主 Agent 的 token
  - think_tool 让子 Agent 在搜索之间"停下来想一想"，提高搜索质量
"""


import os
from datetime import datetime
import sys
from pathlib import Path as _Path
# sys.path.insert(0, str(_Path(__file__).parent.parent / "shared"))
from custom_model import StreamFixChatOpenAI
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv()

# from langchain_core.language_models import init_chat_model
# from langchain_google_genai import ChatGoogleGenerativeAI
from deepagents import create_deep_agent

from research_agent.prompts import (
    RESEARCHER_INSTRUCTIONS,
    RESEARCH_WORKFLOW_INSTRUCTIONS,
    SUBAGENT_DELEGATION_INSTRUCTIONS,
)
from research_agent.tools import ddg_search, think_tool

# 配置参数
# 最多同时派出几个研究子 Agent（并行度）
max_concurrent_research_units = 3
# 每个子Agent 最多进行几轮搜索 - 反思循环
max_researcher_iterations = 3

# 获取当前日期（注入到prompts中，让Agent知道“今天是几号”）
current_date = datetime.now().strftime("%Y-%m-%d")

# 构建主 Agent 的系统提示
# 将两段指令拼接成主 Agent 的完整系统提示
# 1.RESEARCH_WORKFLOW_INSTRUCTIONS: 定义研究流程（计划-委派-综合-写报告-验证）
# 2. SUBAGENT_DELEGATION_INSTRUCTIONS: 定义如何委派子 Agent（并行度、迭代次数等）
INSTRUCTIONS = (
    RESEARCH_WORKFLOW_INSTRUCTIONS
    + "\n\n"
    + "=" * 80
    + "\n\n"
    + SUBAGENT_DELEGATION_INSTRUCTIONS.format(
        max_concurrent_research_units=max_concurrent_research_units,
        max_researcher_iterations=max_researcher_iterations,
    )
)

# 定义研究子Agent
research_sub_agent = {
    "name": "gdyl-research-agent",
    "description": "一个专门从事深度网络研究的智能代理，能够独立探索和分析复杂问题",
    "system_prompt": RESEARCHER_INSTRUCTIONS.format(date=current_date),
    "tools": [ddg_search, think_tool],
}

# 新增一个中文摘要子agent
summary_agent = {
    "name": "chinese-summarizer",
    "description": "将研究发现翻译并整理为中文摘要。在收集完所有英文资料后委派给此 Agent。",
    "system_prompt": (
        "你是一名专业的中文科技编辑。"
        "你会收到英文研究发现，需要将其整理为结构清晰、语言流畅的中文摘要。"
        "要求：保留关键数据和来源，去除冗余内容，长度控制在 500 字以内。"
    ),
}


# model = ChatOpenAI(
#     model="qwen-plus",
#     api_key=os.getenv("DASHSCOPE_API_KEY"),
#     base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
#     temperature=0,
# )

# model = StreamFixChatOpenAI(
#     model="gpt-5.4",
#     api_key=os.getenv("CODEX_API_KEY"),
#     base_url="https://code.swpumc.cn/v1",
#     temperature=0,
# )


# model = StreamFixChatOpenAI(
#     model="gpt-5.4",
#     api_key=os.getenv("OPENAI_API_KEY"),
#     base_url="https://code.swpumc.cn/v1",
#     temperature=0,
# )
model = StreamFixChatOpenAI(
    model="gpt-5.4-mini",
    api_key=os.getenv("OPENAI_API_KEY1"),
    base_url="https://code.swpumc.cn/v1",
    temperature=0,
)

agent = create_deep_agent(
    model=model,
    tools=[ddg_search, think_tool],
    system_prompt=INSTRUCTIONS,
    subagents=[research_sub_agent,summary_agent],
)
