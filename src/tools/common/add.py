"""加法工具"""

from .. import AgentTool, ToolParameters, AgentToolParameter, ToolContent, ToolResult
from typing import Any, Callable

async def execute_add(
    tool_call_id: str,
    args: dict[str, Any],
    signal: Any,
    on_update: Callable | None = None
) -> ToolResult:
    """执行加法"""
    a: float = args.get("a", 0)
    b: float = args.get("b", 0)
    result = a + b
    return ToolResult(
        content=[ToolContent(type="text", text=f"{a} + {b} = {result}")],
        details={"a": a, "b": b, "result": result}
    )


add = AgentTool(
    name="add",
    description="计算两个数字的和",
    parameters=ToolParameters(
        type="object",
        properties={
            "a": AgentToolParameter(type="number", description="第一个数字"),
            "b": AgentToolParameter(type="number", description="第二个数字"),
        },
        required=["a", "b"],
    ),
    execute=execute_add,
)
