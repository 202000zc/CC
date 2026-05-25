"""配置管理 - Agent 配置项"""

from dataclasses import dataclass, field
from .enums import QueueMode, ToolExecutionMode


@dataclass
class ThinkingBudgets:
    minimal: int = 128
    low: int = 512
    medium: int = 1024
    high: int = 2048
    xhigh: int = 4096


@dataclass
class AgentConfig:
    max_turns: int = 20
    tool_timeout: float = 30.0
    max_retries: int = 3
    retry_backoff: float = 1.0
    thinking_level: str = "off"
    thinking_budgets: ThinkingBudgets = field(default_factory=ThinkingBudgets)
    steering_mode: QueueMode = QueueMode.ONE_AT_A_TIME
    follow_up_mode: QueueMode = QueueMode.ONE_AT_A_TIME
    tool_execution: ToolExecutionMode = ToolExecutionMode.PARALLEL
