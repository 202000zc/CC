"""枚举定义 - 统一状态和停止原因"""

from enum import Enum


class StopReason(Enum):
    STOP = "stop"
    TOOL_CALLS = "tool_calls"
    LENGTH = "length"
    CONTENT_FILTER = "content_filter"
    ERROR = "error"


class AgentState(Enum):
    IDLE = "idle"
    RUNNING = "running"
    EXECUTING_TOOLS = "executing_tools"
    STREAMING = "streaming"
    PAUSED = "paused"
    STOPPED = "stopped"
    ERROR = "error"


class QueueMode(Enum):
    ONE_AT_A_TIME = "one_at_a_time"
    ALL_AT_ONCE = "all"


class ToolExecutionMode(Enum):
    SEQUENTIAL = "sequential"
    PARALLEL = "parallel"


FINISH_REASON_MAP: dict[str, StopReason] = {
    "stop": StopReason.STOP,
    "tool_calls": StopReason.TOOL_CALLS,
    "length": StopReason.LENGTH,
    "content_filter": StopReason.CONTENT_FILTER,
    "end_turn": StopReason.STOP,
    "tool_use": StopReason.TOOL_CALLS,
    "max_tokens": StopReason.LENGTH,
    "STOP": StopReason.STOP,
    "MAX_TOKENS": StopReason.LENGTH,
    "SAFETY": StopReason.CONTENT_FILTER,
}


def normalize_finish_reason(raw: str) -> StopReason:
    return FINISH_REASON_MAP.get(raw, StopReason.ERROR)
