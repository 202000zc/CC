"""Agent Loop - 核心循环引擎"""

import asyncio
import json
import logging
import time
from typing import Any, Callable, Awaitable

from .enums import StopReason, AgentState
from .config import AgentConfig
from .types import (
    AgentMessage,
    ToolCallInfo,
    LLMResponse,
    AgentResult,
    ToolResult,
)
from .exceptions import (
    MaxTurnsExceededError,
    AgentAbortedError,
    APIError,
    ContentFilterError,
)
from .events import EventEmitter
from .tool_executor import ToolExecutor

logger = logging.getLogger(__name__)

LLMClient = Callable[..., Awaitable[LLMResponse]]


class AgentLoop:
    def __init__(
        self,
        llm_client: LLMClient,
        tools: dict[str, Any],
        config: AgentConfig | None = None,
    ):
        self.llm_client = llm_client
        self.tools = tools
        self.config = config or AgentConfig()
        self.events = EventEmitter()
        self.tool_executor = ToolExecutor(tools, self.config.tool_timeout)

        self._state = AgentState.IDLE
        self._abort_signal = asyncio.Event()
        self._messages: list[AgentMessage] = []
        self._tool_results: list[ToolResult] = []

    @property
    def state(self) -> AgentState:
        return self._state

    @property
    def messages(self) -> list[AgentMessage]:
        return list(self._messages)

    async def run(
        self,
        user_message: str | AgentMessage,
        system_prompt: str | None = None,
        model: str = "default",
        **kwargs,
    ) -> AgentResult:
        if isinstance(user_message, str):
            initial_message = AgentMessage(role="user", content=user_message)
        else:
            initial_message = user_message

        self._messages = []
        self._tool_results = []
        self._abort_signal.clear()

        if system_prompt:
            self._messages.append(AgentMessage(role="system", content=system_prompt))

        self._messages.append(initial_message)

        await self.events.emit("agent_start", {
            "message": initial_message.content,
            "model": model,
        })

        try:
            result = await self._run_loop(model, **kwargs)
            await self.events.emit("agent_end", {"result": result})
            return result
        except Exception as e:
            logger.error(f"Agent 循环异常: {e}")
            raise

    async def _run_loop(self, model: str, **kwargs) -> AgentResult:
        turn = 0
        retry_count = 0
        final_content = ""

        while turn < self.config.max_turns:
            if self._abort_signal.is_set():
                raise AgentAbortedError("用户取消操作")

            turn += 1
            self._state = AgentState.RUNNING

            await self.events.emit("turn_start", {"turn": turn})

            try:
                response = await self.llm_client(
                    model=model,
                    messages=self._format_messages(),
                    tools=self._get_tool_definitions(),
                    **kwargs,
                )

                finish_reason = response.finish_reason
                logger.debug(f"Turn {turn}: finish_reason={finish_reason.value}")

                match finish_reason:
                    case StopReason.STOP:
                        final_content = response.content or ""
                        self._add_assistant_message(response.content)
                        break

                    case StopReason.TOOL_CALLS:
                        tool_results = await self._handle_tool_calls(
                            response.tool_calls,
                            response.reasoning_content
                        )
                        self._tool_results.extend(tool_results)

                        all_terminate = all(r.terminate for r in tool_results if r.success)
                        if all_terminate and tool_results:
                            final_content = response.content or ""
                            break

                    case StopReason.LENGTH:
                        partial = response.content or ""
                        self._add_assistant_message(partial)
                        self._messages.append(AgentMessage(role="user", content="请继续"))
                        logger.warning(f"Turn {turn}: 响应被截断，继续生成")

                    case StopReason.CONTENT_FILTER:
                        await self.events.emit("error", {
                            "type": "content_filter",
                            "message": "内容被过滤",
                        })
                        self._messages.append(
                            AgentMessage(role="user", content="你的回答包含不当内容，请重新组织语言。")
                        )
                        logger.warning(f"Turn {turn}: 内容被过滤")

                    case StopReason.ERROR:
                        retry_count += 1
                        if retry_count < self.config.max_retries:
                            backoff = self.config.retry_backoff * (2 ** (retry_count - 1))
                            logger.warning(f"Turn {turn}: 错误，{backoff}s 后重试 ({retry_count}/{self.config.max_retries})")
                            await asyncio.sleep(backoff)
                            continue
                        else:
                            raise APIError(f"API 连续错误 {self.config.max_retries} 次")

                await self.events.emit("turn_end", {
                    "turn": turn,
                    "finish_reason": finish_reason.value,
                })

            except MaxTurnsExceededError:
                raise
            except AgentAbortedError:
                raise
            except Exception as e:
                logger.error(f"Turn {turn} 异常: {e}", exc_info=True)
                retry_count += 1
                if retry_count < self.config.max_retries:
                    await asyncio.sleep(self.config.retry_backoff * retry_count)
                    continue
                raise

        if turn >= self.config.max_turns:
            raise MaxTurnsExceededError(self.config.max_turns)

        self._state = AgentState.STOPPED

        return AgentResult(
            final_response=final_content or "（无内容）",
            messages=list(self._messages),
            total_turns=turn,
            stop_reason=StopReason.STOP,
            tool_results=list(self._tool_results),
        )

    async def _handle_tool_calls(
        self,
        tool_calls: list[ToolCallInfo],
        reasoning_content: str | None = None
    ) -> list[ToolResult]:
        self._state = AgentState.EXECUTING_TOOLS

        assistant_msg = AgentMessage(
            role="assistant",
            content=None,
            tool_calls=[{
                "id": tc.id,
                "type": "function",
                "function": {
                    "name": tc.name,
                    "arguments": json.dumps(tc.arguments) if isinstance(tc.arguments, dict) else tc.arguments
                }
            } for tc in tool_calls],
            reasoning_content=reasoning_content,
        )
        self._messages.append(assistant_msg)

        if self.config.tool_execution.value == "parallel":
            results = await self.tool_executor.execute_tools_parallel(tool_calls)
        else:
            results = []
            for tc in tool_calls:
                result = await self.tool_executor.execute_tool(tc)
                results.append(result)

        for result in results:
            self._messages.append(AgentMessage(
                role="tool",
                content=result.content,
                tool_call_id=result.tool_call_id,
            ))

        return results

    def _format_messages(self) -> list[dict]:
        formatted = []
        for msg in self._messages:
            m: dict[str, Any] = {"role": msg.role}
            if msg.content:
                m["content"] = msg.content
            if msg.tool_calls:
                formatted_tc = []
                for tc in msg.tool_calls:
                    func = tc.get("function", {})
                    args = func.get("arguments")
                    if isinstance(args, dict):
                        args = json.dumps(args)
                    formatted_tc.append({
                        "id": tc.get("id"),
                        "type": "function",
                        "function": {
                            "name": func.get("name"),
                            "arguments": args
                        }
                    })
                m["tool_calls"] = formatted_tc
            if msg.tool_call_id:
                m["tool_call_id"] = msg.tool_call_id
            if msg.reasoning_content:
                m["reasoning_content"] = msg.reasoning_content
            formatted.append(m)
        return formatted

    def _get_tool_definitions(self) -> list[dict]:
        definitions = []
        for tool in self.tools.values():
            if hasattr(tool, "to_openai_tool"):
                definitions.append(tool.to_openai_tool())
        return definitions

    def _add_assistant_message(self, content: str | None):
        self._messages.append(AgentMessage(role="assistant", content=content))

    def abort(self):
        self._abort_signal.set()
        self._state = AgentState.STOPPED
        logger.info("Agent 已中止")

    def on(self, event_type: str, handler: Callable):
        return self.events.on(event_type, handler)

    def subscribe(self, handler: Callable):
        return self.events.on("*", handler)
