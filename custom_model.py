"""自定义 ChatModel：兼容 Content-Type: text/event-stream 的 OpenAI 代理接口。

【问题背景】
https://code.swpumc.cn 返回标准 OpenAI JSON 格式，但 Content-Type 错误地设置为
text/event-stream，导致 LangChain 的 ChatOpenAI 把响应当流式数据处理，最终报错：
  AttributeError: 'str' object has no attribute 'model_dump'

【解决思路】
绕过 LangChain 对 Content-Type 的判断，直接用 OpenAI 原生 SDK（AsyncOpenAI）调用，
然后把结果包装成 LangChain 标准的 AIMessage 返回。

【bind_tools 设计】
deepagents 框架（通过 langchain 的 create_agent）会在构建时调用 model.bind_tools(tools)，
之后 SubAgentMiddleware 等中间件还会在运行时再次调用 bind_tools()。

标准 BaseChatModel.bind_tools() 返回 self.bind(tools=...) → RunnableBinding，
这会导致中间件链断裂（RunnableBinding 没有 bind_tools 方法）。

解决方案：覆写 bind_tools()，将工具存储为实例字段并返回 self（链式调用），
_generate/_agenerate 从实例字段中取工具，传给 OpenAI API。
"""

from __future__ import annotations

import asyncio
import time
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from pydantic import PrivateAttr
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.outputs import ChatGeneration, ChatResult
import httpx
import json as _json

from openai import APIStatusError, AsyncOpenAI, OpenAI

# 服务器繁忙时重试配置
_MAX_RETRIES = 8
_RETRY_BASE_DELAY = 8.0  # 初始等待秒数，每次翻倍
_RETRY_MAX_DELAY = 120.0  # 最长等待秒数
_BUSY_KEYWORDS = (
    "系统繁忙",
    "当前系统繁忙",
    "请稍后再试",
    "稍后重试",
    "server busy",
    "rate limit",
    "too many requests",
    "overloaded",
    "concurrency limit",
    "service unavailable",
    "temporarily unavailable",
)

# 全局并发信号量：最多允许 2 个请求同时发往 API
# 避免多个 thread 并发运行时加剧服务器繁忙
_ASYNC_SEMAPHORE: asyncio.Semaphore | None = None


def _get_semaphore() -> asyncio.Semaphore:
    """获取（懒加载）全局并发信号量。"""
    global _ASYNC_SEMAPHORE
    if _ASYNC_SEMAPHORE is None:
        _ASYNC_SEMAPHORE = asyncio.Semaphore(2)
    return _ASYNC_SEMAPHORE


def _to_openai_messages(messages: list[BaseMessage]) -> list[dict]:
    """将 LangChain 消息转为 OpenAI API 格式。"""
    result = []
    for msg in messages:
        if isinstance(msg, HumanMessage):
            role = "user"
        elif isinstance(msg, AIMessage):
            role = "assistant"
        elif isinstance(msg, SystemMessage):
            role = "system"
        elif isinstance(msg, ToolMessage):
            # 工具调用结果消息
            result.append(
                {
                    "role": "tool",
                    "content": msg.content
                    if isinstance(msg.content, str)
                    else str(msg.content),
                    "tool_call_id": msg.tool_call_id,
                }
            )
            continue
        else:
            role = "user"

        # 处理 AIMessage 中的 tool_calls（需要转换格式）
        if isinstance(msg, AIMessage) and msg.tool_calls:
            ai_msg: dict[str, Any] = {
                "role": "assistant",
                "content": msg.content
                if isinstance(msg.content, str)
                else (msg.content or None),
                "tool_calls": [
                    {
                        "id": tc["id"],
                        "type": "function",
                        "function": {
                            "name": tc["name"],
                            "arguments": __import__("json").dumps(tc["args"]),
                        },
                    }
                    for tc in msg.tool_calls
                ],
            }
            result.append(ai_msg)
            continue

        content = msg.content if isinstance(msg.content, str) else str(msg.content)
        result.append({"role": role, "content": content})
    return result


class StreamFixChatOpenAI(BaseChatModel):
    """兼容 text/event-stream Content-Type 错误的自定义 ChatOpenAI 包装器。

    直接使用 OpenAI 原生 SDK，绕过 LangChain 的 httpx 流式解析。
    bind_tools() 返回 self 而非 RunnableBinding，以兼容 deepagents 中间件链。

    Args:
        model: 模型名称。
        api_key: API 密钥。
        base_url: 自定义代理接口地址。
        temperature: 温度参数。
    """

    model: str = "gpt-5.4-mini"
    api_key: str = ""
    base_url: str = "https://code.swpumc.cn/v1"
    temperature: float = 0.0
    timeout: float = 300.0
    max_retries: int = 2
    # 存储通过 bind_tools() 绑定的 OpenAI 格式工具（PrivateAttr 避免 Pydantic 序列化问题）
    _bound_tools: list[dict] = PrivateAttr(default_factory=list)

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)

    @property
    def _llm_type(self) -> str:
        return "stream-fix-chat-openai"

    def _get_sync_client(self) -> OpenAI:
        return OpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
        )

    def _get_async_client(self) -> AsyncOpenAI:
        return AsyncOpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
        )

    def bind_tools(self, tools: list, **kwargs: Any) -> "StreamFixChatOpenAI":
        """绑定工具，返回 self 以兼容 deepagents 中间件链。"""
        from langchain_core.utils.function_calling import convert_to_openai_tool

        self._bound_tools = [convert_to_openai_tool(t) for t in tools]
        return self

    def _build_create_kwargs(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None,
        kwargs: dict[str, Any],
    ) -> dict[str, Any]:
        """构建 OpenAI API 调用参数。"""
        oai_messages = _to_openai_messages(messages)

        # kwargs 中的 tools 优先（来自 model.bind(tools=...)），其次用实例存储的工具
        tools = kwargs.get("tools") or self._bound_tools or None
        tool_choice = kwargs.get("tool_choice")

        create_kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": oai_messages,
            "temperature": self.temperature,
            "stream": True,  # 强制流式：该代理 stream=False 时不返回 content
        }
        if stop:
            create_kwargs["stop"] = stop
        if tools:
            create_kwargs["tools"] = tools
        if tool_choice:
            create_kwargs["tool_choice"] = tool_choice
        return create_kwargs

    @staticmethod
    def _aggregate_stream_chunks(chunks: list[Any]) -> AIMessage:
        """聚合流式 chunks 为 AIMessage（同时支持 tool_calls）。"""
        content_parts: list[str] = []
        # tool_calls 聚合: index -> {id, name, arguments_parts}
        tc_map: dict[int, dict[str, Any]] = {}

        for chunk in chunks:
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta
            if delta.content:
                content_parts.append(delta.content)
            if delta.tool_calls:
                for tc_delta in delta.tool_calls:
                    idx = tc_delta.index
                    if idx not in tc_map:
                        tc_map[idx] = {"id": "", "name": "", "arguments": ""}
                    if tc_delta.id:
                        tc_map[idx]["id"] = tc_delta.id
                    if tc_delta.function and tc_delta.function.name:
                        tc_map[idx]["name"] += tc_delta.function.name
                    if tc_delta.function and tc_delta.function.arguments:
                        tc_map[idx]["arguments"] += tc_delta.function.arguments

        tool_calls = [
            {
                "name": tc["name"],
                "args": _json.loads(tc["arguments"] or "{}"),
                "id": tc["id"],
                "type": "tool_call",
            }
            for tc in tc_map.values()
            if tc["name"]
        ]
        return AIMessage(content="".join(content_parts), tool_calls=tool_calls)

    @staticmethod
    def _parse_response(msg: Any) -> AIMessage:
        """将 OpenAI 响应消息转换为 LangChain AIMessage。

        兼容两种响应格式：
        - 标准 ChatCompletion message 对象
        - 原始字符串（text/event-stream 代理直接返回文本）
        """
        # 原始字符串响应
        if isinstance(msg, str):
            return AIMessage(content=msg, tool_calls=[])

        tool_calls = None
        if msg.tool_calls:
            tool_calls = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments,
                    },
                }
                for tc in msg.tool_calls
            ]

        return AIMessage(
            content=msg.content or "",
            tool_calls=[
                {
                    "name": tc["function"]["name"],
                    "args": __import__("json").loads(tc["function"]["arguments"]),
                    "id": tc["id"],
                    "type": "tool_call",
                }
                for tc in (tool_calls or [])
            ],
        )

    @staticmethod
    def _is_busy_error(exc: Exception) -> bool:
        """判断是否是服务器繁忙/限流/超时错误，需要重试。"""
        # httpx.HTTPStatusError: 502/503/504 等网关超时
        if isinstance(exc, httpx.HTTPStatusError):
            return exc.response.status_code in (429, 500, 502, 503, 504)
        msg = str(exc).lower()
        # 504/502 文本形式
        if "504" in msg or "502" in msg or "503" in msg or "gateway" in msg or "timeout" in msg:
            return True
        return any(k.lower() in msg for k in _BUSY_KEYWORDS)

    @staticmethod
    def _check_busy_response(response: Any) -> None:
        """检查 200 响应中是否包含"系统繁忙"文字（部分代理会这样返回）。

        部分代理（如 text/event-stream 接口）会将响应直接返回为字符串，
        需要先判断类型再访问 .choices 属性，避免 AttributeError。

        Args:
            response: OpenAI SDK 的 ChatCompletion 对象，或原始字符串响应。

        Raises:
            RuntimeError: 如果响应内容包含繁忙关键词。
        """
        # 原始字符串响应（text/event-stream 代理返回格式）
        if isinstance(response, str):
            if any(k in response for k in _BUSY_KEYWORDS):
                raise RuntimeError(
                    f"系统繁忙（字符串响应含错误文字）: {response[:100]}"
                )
            return
        # 标准 ChatCompletion 对象
        if not response.choices:
            return
        content = response.choices[0].message.content or ""
        if any(k in content for k in _BUSY_KEYWORDS):
            raise RuntimeError(f"系统繁忙（200 响应含错误文字）: {content[:100]}")

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        client = self._get_sync_client()
        create_kwargs = self._build_create_kwargs(messages, stop, kwargs)
        delay = _RETRY_BASE_DELAY
        for attempt in range(_MAX_RETRIES):
            try:
                stream = client.chat.completions.create(**create_kwargs)
                chunks = list(stream)
                if chunks:
                    content = "".join(
                        c.choices[0].delta.content or ""
                        for c in chunks if c.choices
                    )
                    if any(k in content for k in _BUSY_KEYWORDS):
                        raise RuntimeError(f"系统繁忙（流式响应）: {content[:100]}")
                ai_message = self._aggregate_stream_chunks(chunks)
                return ChatResult(generations=[ChatGeneration(message=ai_message)])
            except (APIStatusError, RuntimeError, httpx.HTTPStatusError) as e:
                if self._is_busy_error(e) and attempt < _MAX_RETRIES - 1:
                    print(f"[StreamFixChatOpenAI] 服务器繁忙，{delay:.0f}s 后重试（第 {attempt + 1}/{_MAX_RETRIES - 1} 次）...")
                    time.sleep(delay)
                    delay = min(delay * 2, _RETRY_MAX_DELAY)
                else:
                    raise
        raise RuntimeError("重试次数耗尽")

    async def _agenerate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        # 信号量限制全局并发：最多 2 个请求同时访问 API，避免加剧服务器繁忙
        async with _get_semaphore():
            client = self._get_async_client()
            create_kwargs = self._build_create_kwargs(messages, stop, kwargs)
            delay = _RETRY_BASE_DELAY
            for attempt in range(_MAX_RETRIES):
                try:
                    stream = await client.chat.completions.create(**create_kwargs)
                    chunks = [chunk async for chunk in stream]
                    if chunks:
                        content = "".join(
                            c.choices[0].delta.content or ""
                            for c in chunks if c.choices
                        )
                        if any(k in content for k in _BUSY_KEYWORDS):
                            raise RuntimeError(f"系统繁忙（流式响应）: {content[:100]}")
                    ai_message = self._aggregate_stream_chunks(chunks)
                    return ChatResult(generations=[ChatGeneration(message=ai_message)])
                except (APIStatusError, RuntimeError, httpx.HTTPStatusError) as e:
                    if self._is_busy_error(e) and attempt < _MAX_RETRIES - 1:
                        print(f"[StreamFixChatOpenAI] 服务器繁忙，{delay:.0f}s 后重试（第 {attempt + 1}/{_MAX_RETRIES - 1} 次）...")
                        await asyncio.sleep(delay)
                        delay = min(delay * 2, _RETRY_MAX_DELAY)
                    else:
                        raise
            raise RuntimeError("重试次数耗尽")
