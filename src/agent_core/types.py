"""类型定义 - 核心数据结构"""

from dataclasses import dataclass, field
from typing import Any, Callable, Awaitable
from .enums import StopReason


@dataclass
class AgentMessage:
    role: str
    content: str | None = None
    tool_calls: list[dict] | None = None
    tool_call_id: str | None = None
    timestamp: float | None = None
    reasoning_content: str | None = None


@dataclass
class ToolCallInfo:
    id: str
    name: str
    arguments: dict[str, Any]


@dataclass
class ToolResult:
    tool_call_id: str
    content: str
    success: bool = True
    error: str | None = None
    details: dict[str, Any] | None = None
    terminate: bool = False


@dataclass
class LLMResponse:
    content: str | None
    finish_reason: StopReason
    tool_calls: list[ToolCallInfo] = field(default_factory=list)
    reasoning_content: str | None = None
    usage: dict[str, int] | None = None


@dataclass
class AgentResult:
    final_response: str
    messages: list[AgentMessage]
    total_turns: int
    stop_reason: StopReason
    tool_results: list[ToolResult] = field(default_factory=list)


ToolExecutor = Callable[
    [str, dict[str, Any], Any],
    Awaitable[ToolResult]
]
