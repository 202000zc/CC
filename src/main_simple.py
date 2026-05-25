import asyncio
import sys
import os
from dotenv import load_dotenv

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

load_dotenv()

from unified_llm import UnifiedLLMClient
from src.agent_simple import Agent
from src.tools import tools_by_name


def create_llm_client():
    client = UnifiedLLMClient(
        provider="openai",
        api_key=os.getenv("KIMI_API_KEY"),
        base_url=os.getenv("KIMI_BASE_URL"),
    )

    def llm_client(model: str, messages: list[dict], tools: list[dict] | None = None):
        return client.chat(model=os.getenv("KIMI_MODEL"), messages=messages, tools=tools or [])

    return llm_client


async def main():
    agent = Agent(
        llm_client=create_llm_client(),
        tools=tools_by_name,
        max_turns=10,
    )

    agent.on("agent_start", lambda d: print("\n🚀 Agent 开始"))
    agent.on("turn_start", lambda d: print(f"\n📋 第 {d['turn']} 轮"))
    agent.on("tool_call", lambda d: print(f"  🤖 LLM 要调用工具: {[tc.function.name for tc in d['calls']]}"))
    agent.on("tool_result", lambda d: print(f"  ✅ {d['tool']}({d['args']})"))
    agent.on("turn_end", lambda d: print(f"  📝 第 {d['turn']} 轮结束 ({d['type']})"))
    agent.on("agent_end", lambda d: print(f"\n🏁 Agent 结束: {d['response'][:50]}..."))

    result = await agent.run(
        user_message="帮我计算 123 + 456 等于多少？",
        system_prompt="你是一个有帮助的助手。当需要计算时，使用 add 工具。",
    )

    print("\n" + "=" * 50)
    print("最终回答:", result)


if __name__ == "__main__":
    asyncio.run(main())
