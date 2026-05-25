"""极简 Agent 类"""

import json
from dataclasses import dataclass, field
from typing import Callable, Any


@dataclass
class Agent:
    """极简 Agent：包含 ReAct 循环"""

    llm_client: Callable
    tools: dict[str, Any] = field(default_factory=dict)
    max_turns: int = 20
    messages: list[dict] = field(default_factory=list)
    _subscribers: dict[str, list[Callable]] = field(default_factory=dict, init=False)

    def on(self, event: str, handler: Callable):
        """订阅事件: agent_start, turn_start, tool_call, tool_result, agent_end"""
        if event not in self._subscribers:
            self._subscribers[event] = []
        self._subscribers[event].append(handler)

        def unsubscribe():
            self._subscribers[event].remove(handler)

        return unsubscribe

    def _emit(self, event: str, data: Any = None):
        for handler in self._subscribers.get(event, []):
            handler(data)
        for handler in self._subscribers.get("*", []):
            handler(event, data)

    async def run(self, user_message: str, system_prompt: str | None = None) -> str:
        """运行 ReAct 循环"""
        self.messages = []
        if system_prompt:
            self.messages.append({"role": "system", "content": system_prompt})

        self.messages.append({"role": "user", "content": user_message})
        self._emit("agent_start", {"message": user_message})

        for turn in range(self.max_turns):
            self._emit("turn_start", {"turn": turn + 1})

            tool_defs = [tool.to_openai_tool() for tool in self.tools.values()]
            response = self.llm_client(model="default", messages=self.messages, tools=tool_defs)

            choice = response.choices[0]
            content = choice.message.content
            tool_calls = choice.message.tool_calls

            if tool_calls:
                self._emit("tool_call", {"calls": tool_calls, "content": content})

                assistant_msg: dict[str, Any] = {
                    "role": "assistant",
                    "content": content,
                    "tool_calls": [
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {
                                "name": tc.function.name,
                                "arguments": tc.function.arguments,
                            }
                        }
                        for tc in tool_calls
                    ],
                }

                reasoning = getattr(choice.message, "reasoning_content", None)
                if reasoning:
                    assistant_msg["reasoning_content"] = reasoning

                self.messages.append(assistant_msg)

                for tc in tool_calls:
                    tool_name = tc.function.name
                    args = json.loads(tc.function.arguments) if isinstance(tc.function.arguments, str) else tc.function.arguments

                    self._emit("tool_result", {"tool": tool_name, "args": args})

                    tool = self.tools.get(tool_name)
                    if tool:
                        result = await tool.execute(tc.id, args, None)
                        result_text = result.content[0].text if hasattr(result, "content") else result.get("content", [""])[0].get("text", "")
                    else:
                        result_text = f"错误：未找到工具 {tool_name}"

                    self.messages.append({
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": result_text,
                    })

                self._emit("turn_end", {"turn": turn + 1, "type": "tool_calls"})
            else:
                self.messages.append({"role": "assistant", "content": content})
                self._emit("turn_end", {"turn": turn + 1, "type": "stop"})
                self._emit("agent_end", {"response": content})
                return content

        self._emit("agent_end", {"response": "达到最大轮次限制"})
        return "达到最大轮次限制"
