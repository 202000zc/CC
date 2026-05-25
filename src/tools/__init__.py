"""工具模块 - Python 版本 (Pydantic)"""

from typing import Any, Callable, Awaitable
from pydantic import BaseModel, Field


class AgentToolParameter(BaseModel):
    type: str
    description: str | None = None
    properties: dict[str, "AgentToolParameter"] | None = None
    required: list[str] | None = None


class ToolParameters(BaseModel):
    type: str = "object"
    properties: dict[str, AgentToolParameter] = Field(default_factory=dict)
    required: list[str] | None = None


class ToolContent(BaseModel):
    type: str
    text: str


class ToolResult(BaseModel):
    content: list[ToolContent]
    details: dict[str, Any] | None = None
    terminate: bool = False


ToolExecutor = Callable[
    [str, dict[str, Any], Any, Callable | None],
    Awaitable[ToolResult]
]


class AgentTool(BaseModel):
    name: str
    description: str
    parameters: ToolParameters
    execute: ToolExecutor

    def to_openai_tool(self) -> dict[str, Any]:
        """转换为 OpenAI API 兼容的工具定义格式"""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters.model_dump(exclude_none=True),
            },
        }


from .common.add import add

file_tools: list[AgentTool] = []
web_tools: list[AgentTool] = []
bash_tools: list[AgentTool] = []
common_tools: list[AgentTool] = [add]

all_tools: list[AgentTool] = [
    *file_tools,
    *web_tools,
    *bash_tools,
    *common_tools,
]

tools_by_name: dict[str, AgentTool] = {tool.name: tool for tool in all_tools}


def register_tool(tool: AgentTool) -> None:
    tools_by_name[tool.name] = tool
    all_tools.append(tool)


def get_tool(name: str) -> AgentTool | None:
    return tools_by_name.get(name)


def clear_tools() -> None:
    all_tools.clear()
    tools_by_name.clear()


def get_openai_tools() -> list[dict[str, Any]]:
    """获取所有工具的 OpenAI API 兼容格式"""
    return [tool.to_openai_tool() for tool in all_tools]
