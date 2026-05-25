import asyncio
import os
import sys
from dotenv import load_dotenv

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from unified_llm import UnifiedLLMClient
from src.agent_core import AgentLoop, AgentConfig, StopReason, LLMResponse, ToolCallInfo
from src.tools import all_tools, tools_by_name

load_dotenv()

api_key = os.getenv("KIMI_API_KEY")
base_url = os.getenv("KIMI_BASE_URL")
model = os.getenv("KIMI_MODEL")

print(f"API_KEY: {api_key}")
print(f"BASE_URL: {base_url}")
print(f"MODEL: {model}")

client = UnifiedLLMClient(provider="openai", api_key=api_key, base_url=base_url)


async def llm_client_fn(model: str, messages: list[dict], tools: list[dict] | None = None, **kwargs) -> LLMResponse:
    """LLM 客户端适配函数"""
    call_kwargs = {"model": model, "messages": messages}
    if tools:
        call_kwargs["tools"] = tools

    response = client.chat(**call_kwargs)

    choice = response.choices[0]
    finish_raw = choice.finish_reason or "stop"
    finish_reason = StopReason.STOP if finish_raw == "stop" else (
        StopReason.TOOL_CALLS if choice.message.tool_calls else
        StopReason.ERROR
    )

    tool_calls = []
    if choice.message.tool_calls:
        for tc in choice.message.tool_calls:
            import json
            args = json.loads(tc.function.arguments) if isinstance(tc.function.arguments, str) else tc.function.arguments
            tool_calls.append(ToolCallInfo(
                id=tc.id,
                name=tc.function.name,
                arguments=args,
            ))

    return LLMResponse(
        content=choice.message.content,
        finish_reason=finish_reason,
        tool_calls=tool_calls,
        reasoning_content=getattr(choice.message, "reasoning_content", None),
    )


async def main():
    config = AgentConfig(max_turns=10, tool_timeout=30.0)
    
    agent = AgentLoop(
        llm_client=llm_client_fn,
        tools=tools_by_name,
        config=config,
    )

    unsub_agent_start = agent.on("agent_start", lambda d: print("\n🚀 Agent 开始..."))
    unsub_turn_start = agent.on("turn_start", lambda d: print(f"\n📋 第 {d['turn']} 轮"))
    unsub_tool_start = agent.on("tool_execution_start", lambda d: print(f"  🔧 执行工具: {d['tool_name']}({d['args']})"))
    unsub_tool_end = agent.on("tool_execution_end", lambda d: print(f"  ✅ 工具完成: {d['result'].content}"))
    unsub_turn_end = agent.on("turn_end", lambda d: print(f"  📝 第 {d['turn']} 轮结束 (原因: {d['finish_reason']})"))
    unsub_agent_end = agent.on("agent_end", lambda d: print("\n🏁 Agent 结束"))

    try:
        result = await agent.run(
            user_message="帮我计算 123 + 456 等于多少？",
            system_prompt="你是一个有帮助的助手。当需要计算时，使用 add 工具。",
            model=model,
        )

        print("\n" + "=" * 50)
        print("最终回答:")
        print(result.final_response)
        print("=" * 50)
        print(f"\n总轮次: {result.total_turns}")
        print(f"工具调用次数: {len(result.tool_results)}")

    except Exception as e:
        print(f"\n❌ 错误: {e}")

    unsub_agent_start()
    unsub_turn_start()
    unsub_tool_start()
    unsub_tool_end()
    unsub_turn_end()
    unsub_agent_end()


if __name__ == "__main__":
    asyncio.run(main())
