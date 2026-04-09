"""
Ralph Agent — 供 langgraph dev 前端使用的 Ralph 自主循环 Agent
==============================================================

设计原理：
  Ralph 模式的核心是"每次迭代 = 全新 Agent 实例"。
  在 UI 中，用户每次发送消息就相当于启动一次新的 Ralph 迭代：
    - 每次消息创建全新的 create_deep_agent() 实例（无对话历史）
    - 工作目录按 thread_id 隔离：.langgraph_api/ralph_work/<thread_id>/
    - Agent 通过读取 PROGRESS.md 了解之前的工作进度
    - 文件系统（FilesystemBackend）是唯一的跨迭代"记忆"

与 CLI 版本（gdylralph.py）的区别：
  - CLI：Python while 循环，自动驱动
  - UI：用户每发一条消息 = 一次 Ralph 迭代，用户手动控制节奏

使用方式：
  在 gdylresearch 目录运行 `langgraph dev`，
  前端选择 "ralph" assistant 即可使用。
"""

from __future__ import annotations
import asyncio
import os
from datetime import datetime
from pathlib import Path
from typing import Annotated, Any

import sys
from pathlib import Path as _Path
# sys.path.insert(0, str(_Path(__file__).parent.parent / "shared"))
from custom_model import StreamFixChatOpenAI
from deepagents import create_deep_agent
from deepagents.backends import FilesystemBackend
from dotenv import load_dotenv
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.runnables import RunnableConfig
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict

load_dotenv()

# ============================================================================
# Ralph 系统提示
# ============================================================================

RALPH_SYSTEM_PROMPT = """你是一个专业的软件工程师，工作在 Ralph 自主循环模式下。

## Ralph 循环工作原理
你会被反复调用，每次都是全新的对话。你之前的工作保存在文件系统中。
每次被调用时，你需要：
1. 先检查工作目录中的文件（使用 ls 或 read_file）
2. 阅读 PROGRESS.md 了解当前进度（如果存在）
3. 继续下一个未完成的任务
4. 完成后更新 PROGRESS.md 记录进度

## PROGRESS.md 格式
每次迭代结束前，将 PROGRESS.md 更新为：
```markdown
# 项目进度

## 已完成
- [x] 任务1（完成时间）

## 进行中
- [ ] 当前任务

## 待完成
- [ ] 下一个任务
```

## 工作规范
- 每次迭代专注于完成 1-2 个具体的任务，不要贪多
- 代码必须可以直接运行，不写半成品
- 如果发现之前的代码有问题，先修复再继续
- 保持代码整洁，加必要的注释
- 操作完成后，给用户一个清晰的迭代总结
"""

# ============================================================================
# LangGraph State 定义
# ============================================================================

class RalphState(TypedDict):
    """Ralph Agent 的状态。

    messages 字段使用 add_messages reducer，LangGraph UI 会自动展示完整对话历史。
    """

    messages: Annotated[list[BaseMessage], add_messages]


# ============================================================================
# 模型构建
# ============================================================================

def _build_model() -> Any:
    """构建模型实例。优先使用 OPENAI_API_KEY（code.swpumc.cn），备用 qwen。"""
    openai_key = os.getenv("OPENAI_API_KEY1")
    if openai_key and openai_key.startswith("sk-"):
        return StreamFixChatOpenAI(
            model="gpt-5.4-mini",
            api_key=openai_key,
            base_url="https://code.swpumc.cn/v1",
            temperature=0,
        )
    # 备用：阿里云 qwen-plus
    return ChatOpenAI(
        model="gpt-5.4",
        api_key=os.getenv("OPENAI_API_KEY") or "",
        base_url="https://code.swpumc.cn/v1",
        temperature=0,
    )


# ============================================================================
# Ralph 迭代节点
# ============================================================================

async def ralph_node(state: RalphState, config: RunnableConfig) -> dict[str, list[BaseMessage]]:
    """Ralph 核心节点：每次调用 = 一次 Ralph 迭代。

    从用户消息中提取任务，以全新的 Agent 实例执行一次迭代。
    工作目录按 thread_id 隔离，Agent 可以跨迭代读写文件。

    Args:
        state: 当前图状态，包含消息历史。
        config: LangGraph 配置，包含 thread_id。

    Returns:
        包含 AI 回复消息的字典。
    """
    # 获取 thread_id（用于隔离工作目录）
    thread_id = config.get("configurable", {}).get("thread_id", "default")

    # 工作目录：.langgraph_api/ralph_work/<thread_id>/
    # 这样不同 thread（不同用户/对话）的文件不会相互干扰
    base_dir = Path(__file__).parent / ".langgraph_api" / "ralph_work"
    work_dir = base_dir / str(thread_id)
    # work_dir.mkdir(parents=True, exist_ok=True)
    await asyncio.to_thread(work_dir.mkdir, parents=True, exist_ok=True)

    # 提取当前用户消息（最后一条 HumanMessage）
    user_messages = [m for m in state["messages"] if isinstance(m, HumanMessage)]
    if not user_messages:
        return {"messages": [AIMessage(content="请输入你想让 Ralph 完成的任务。")]}

    task = user_messages[-1].content
    if isinstance(task, list):
        # 多模态消息，提取文本
        task = " ".join(p.get("text", "") for p in task if isinstance(p, dict) and p.get("type") == "text")

    current_date = datetime.now().strftime("%Y-%m-%d")
    iteration_num = len(user_messages)  # 第几次迭代

    # 构建迭代消息：告诉 Agent 这是第几次迭代，去工作目录检查进度
    iteration_message = (
        f"## Ralph 迭代 {iteration_num}（{current_date}）\n\n"
        f"工作目录 `{work_dir}` 中包含你之前的工作（如果有）。\n"
        f"请先检查文件，阅读 PROGRESS.md，然后继续推进任务。\n\n"
        f"**任务：**\n{task}\n\n"
        f"完成后更新 PROGRESS.md，记录本次迭代完成了什么，下次还需要做什么。\n"
        f"最后给用户一个清晰的迭代总结，说明完成了什么、下一步建议做什么。"
    )

    # 每次迭代创建全新的 Agent 实例（Ralph 模式核心：清空对话历史）
    model = _build_model()
    one_shot_agent = create_deep_agent(
        model=model,
        system_prompt=RALPH_SYSTEM_PROMPT,
        backend=FilesystemBackend(root_dir=work_dir, virtual_mode=False),
    )

    # 以全新 thread_id 调用，确保无历史记忆
    result = await one_shot_agent.ainvoke(
        {"messages": [HumanMessage(content=iteration_message)]},
        config={"configurable": {"thread_id": f"ralph-inner-{thread_id}-iter{iteration_num}"}},
    )

    # 提取 Agent 最终回复
    inner_messages = result.get("messages", [])
    last_ai = None
    for m in reversed(inner_messages):
        if isinstance(m, AIMessage) and m.content:
            last_ai = m
            break

    if last_ai is None:
        reply_content = f"迭代 {iteration_num} 完成，但没有生成回复内容。请检查 {work_dir} 目录中的文件。"
    else:
        content = last_ai.content
        if isinstance(content, list):
            content = "\n".join(
                p.get("text", "") for p in content
                if isinstance(p, dict) and p.get("type") == "text"
            )
        # 添加工作目录提示
        reply_content = f"{content}\n\n---\n📁 工作目录：`{work_dir}`"

    return {"messages": [AIMessage(content=reply_content)]}


# ============================================================================
# 构建 LangGraph 图，暴露给 langgraph dev
# ============================================================================

_graph_builder = StateGraph(RalphState)
_graph_builder.add_node("ralph", ralph_node)
_graph_builder.set_entry_point("ralph")
_graph_builder.add_edge("ralph", END)

# 供 langgraph.json 引用的入口变量
agent = _graph_builder.compile()
