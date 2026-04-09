"""研究工具。

本模块为研究 Agent 提供搜索和内容处理实用工具，
使用 DuckDuckGo 进行免费网络搜索（无需 API Key）。
"""

import httpx
from duckduckgo_search import DDGS
from langchain_core.tools import InjectedToolArg, tool
from markdownify import markdownify
from typing_extensions import Annotated, Literal


def fetch_webpage_content(url: str, timeout: float = 10.0) -> str:
    """获取并将网页内容转换为 Markdown。"""
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }
    try:
        response = httpx.get(url, headers=headers, timeout=timeout, follow_redirects=True)
        response.raise_for_status()
        return markdownify(response.text)[:3000]
    except Exception as e:
        return f"从 {url} 获取内容出错: {str(e)}"


# 使用duckduckgo_search
@tool
def ddg_search(
    query: str,
    max_results: Annotated[int, InjectedToolArg] = 3,
    topic: Annotated[
        Literal["general", "news", "finance"], InjectedToolArg
    ] = "general",
) -> str:
    """在网络上搜索有关给定查询的信息（使用 DuckDuckGo，免费无需 API Key）。

    Args:
        query: 要执行的搜索查询
        max_results: 返回的最大结果数
        topic: 主题过滤器（当前 DuckDuckGo 实现忽略此参数）

    Returns:
        包含搜索摘要和网页内容的格式化结果
    """
    result_texts = []
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=max_results))

        for result in results:
            url = result.get("href", "")
            title = result.get("title", "无标题")
            snippet = result.get("body", "")

            content = fetch_webpage_content(url) if url else snippet

            result_text = f"""## {title}
**URL:** {url}

{content}

---
"""
            result_texts.append(result_text)

    except Exception as e:
        return f"搜索出错: {str(e)}"

    response = f"""🔍 为 '{query}' 找到 {len(result_texts)} 个结果:

{chr(10).join(result_texts)}"""

    return response


@tool
def think_tool(reflection: str) -> str:
    """用于对研究进度和决策进行战略性反思的工具。

    在每次搜索后使用此工具来分析结果并系统地规划后续步骤。

    Args:
        reflection: 你对研究进度、发现、空白和后续步骤的详细反思

    Returns:
        已记录反思以供决策的确认信息
    """
    return f"反思已记录: {reflection}"
