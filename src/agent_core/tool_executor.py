"""工具执行器 - 验证、执行、超时处理"""

import asyncio
import logging
from typing import Any

from .types import ToolResult, ToolCallInfo
from .exceptions import ToolTimeoutError, ToolExecutionError, ToolValidationError
from .events import EventEmitter

logger = logging.getLogger(__name__)


class ToolExecutor:
    def __init__(self, tools: dict[str, Any], timeout: float = 30.0):
        self.tools = tools
        self.timeout = timeout
        self.events = EventEmitter()

    async def execute_tool(self, tool_call: ToolCallInfo) -> ToolResult:
        tool = self.tools.get(tool_call.name)
        if not tool:
            raise ToolExecutionError(tool_call.name, f"未找到工具: {tool_call.name}")

        await self.events.emit("tool_execution_start", {
            "tool_name": tool_call.name,
            "args": tool_call.arguments,
            "tool_call_id": tool_call.id,
        })

        try:
            result = await asyncio.wait_for(
                self._run_with_validation(tool, tool_call),
                timeout=self.timeout,
            )
            logger.info(f"工具 [{tool_call.name}] 执行成功")
            return result
        except asyncio.TimeoutError:
            raise ToolTimeoutError(tool_call.name, self.timeout)

    async def _run_with_validation(self, tool: Any, tool_call: ToolCallInfo) -> ToolResult:
        try:
            args = tool_call.arguments
            if hasattr(tool, "parameters") and tool.parameters:
                validated = self._validate_args(args, tool.parameters)
                args = validated

            result = await tool.execute(
                tool_call.id,
                args,
                None,
            )

            if isinstance(result, dict):
                content = result.get("content", [{}])[0].get("text", "")
                details = result.get("details")
                terminate = result.get("terminate", False)
            else:
                content = str(result.content[0].text) if result.content else ""
                details = getattr(result, "details", None)
                terminate = getattr(result, "terminate", False)

            tool_result = ToolResult(
                tool_call_id=tool_call.id,
                content=content,
                success=True,
                details=details,
                terminate=terminate,
            )

            await self.events.emit("tool_execution_end", {
                "tool_name": tool_call.name,
                "result": tool_result,
            })

            return tool_result

        except Exception as e:
            error_msg = str(e)
            logger.error(f"工具 [{tool_call.name}] 执行失败: {error_msg}")
            return ToolResult(
                tool_call_id=tool_call.id,
                content=f"错误: {error_msg}",
                success=False,
                error=error_msg,
            )

    def _validate_args(self, args: dict, parameters: Any) -> dict:
        errors = []
        properties = parameters.properties if hasattr(parameters, "properties") else {}
        required = parameters.required if hasattr(parameters, "required") else []

        for req_field in required:
            if req_field not in args or args[req_field] is None:
                errors.append(f"缺少必填参数: {req_field}")

        for key, value in args.items():
            if key in properties:
                param_def = properties[key]
                expected_type = param_def.type if hasattr(param_def, "type") else None
                if expected_type == "number" and not isinstance(value, (int, float)):
                    errors.append(f"参数 [{key}] 应为数字类型")

        if errors:
            raise ToolValidationError(args.get("_tool_name", "unknown"), errors)

        return args

    async def execute_tools_parallel(self, tool_calls: list[ToolCallInfo]) -> list[ToolResult]:
        tasks = [self.execute_tool(tc) for tc in tool_calls]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        final_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                final_results.append(ToolResult(
                    tool_call_id=tool_calls[i].id,
                    content=f"错误: {str(result)}",
                    success=False,
                    error=str(result),
                ))
            else:
                final_results.append(result)

        return final_results
