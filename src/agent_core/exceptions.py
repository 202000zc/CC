"""自定义异常 - 分层错误处理"""


class AgentError(Exception):
    pass


class ToolExecutionError(AgentError):
    def __init__(self, tool_name: str, message: str):
        self.tool_name = tool_name
        self.message = message
        super().__init__(f"工具 [{tool_name}] 执行错误: {message}")


class ToolTimeoutError(AgentError):
    def __init__(self, tool_name: str, timeout: float):
        self.tool_name = tool_name
        self.timeout = timeout
        super().__init__(f"工具 [{tool_name}] 超时: {timeout}s")


class ToolValidationError(AgentError):
    def __init__(self, tool_name: str, errors: list[str]):
        self.tool_name = tool_name
        self.errors = errors
        super().__init__(f"工具 [{tool_name}] 参数验证失败: {'; '.join(errors)}")


class MaxTurnsExceededError(AgentError):
    def __init__(self, max_turns: int):
        self.max_turns = max_turns
        super().__init__(f"达到最大轮次限制: {max_turns}")


class APIError(AgentError):
    def __init__(self, message: str, status_code: int | None = None):
        self.status_code = status_code
        super().__init__(message)


class ContentFilterError(AgentError):
    def __init__(self, message: str = "内容被安全过滤"):
        super().__init__(message)


class AgentAbortedError(AgentError):
    pass
