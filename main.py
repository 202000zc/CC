from openai import OpenAI
from unified_llm import UnifiedLLMClient
from dotenv import load_dotenv
import os
import sys
import asyncio
from unified_llm.base import StopReason, normalize_finish_reason



sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.tools import all_tools, get_tool, get_openai_tools

load_dotenv()

api_key = os.getenv("KIMI_API_KEY")
base_url = os.getenv("KIMI_BASE_URL")
model = os.getenv("KIMI_MODEL")
print(f"API_KEY: {api_key}")
print(f"BASE_URL: {base_url}")
print(f"MODEL: {model}")

client = UnifiedLLMClient(provider="openai", api_key=api_key, base_url=base_url)

messages = [
    {"role": "system", "content": "你是一个有帮助的助手。使用工具来解决问题。"},
    {"role": "user", "content": "1+1等于多少"}
]

tools = get_openai_tools()



finish_reason = None


# while finish_reason != "stop":
#     #有四种状态:
#     #1. "tool_calls": 检测到工具调用，等待工具执行
#     #2. "stop": 用户请求停止，结束对话
#     #3. "length": 最大token数超过限制，结束对话
#     #4. "CONTENT_FILTER": 内容过滤，结束对话
#     #5. "ERROR": 发生错误，结束对话
#     #6. 其他: 其他状态，等待用户输入或工具调用

def tool_call_handler(response):
    """处理工具调用"""
    choice = response.choices[0]
    print("\n" + "=" * 50)
    print(f"检测到工具调用: {len(choice.message.tool_calls)} 个")
    # messages.append(choice.message)
    assistant_message = choice.message
    assistant_msg_dict = {
        "role": "assistant",
        "content": assistant_message.content or "",
        "tool_calls": [
            {
                "id": tool_call.id,
                "type": "function",
                "function": {
                    "name": tool_call.function.name,
                    "arguments": tool_call.function.arguments
                }
            }
            for tool_call in choice.message.tool_calls
        ]
    }
    if hasattr(assistant_message, "reasoning_content") and assistant_message.reasoning_content:
        assistant_msg_dict["reasoning_content"] = assistant_message.reasoning_content
    messages.append(assistant_msg_dict)

    for tool_call in choice.message.tool_calls:
        print(f"工具调用: {tool_call}")
        tool_call_name = tool_call.function.name
        tool_call_args = tool_call.function.arguments
        tool_function = get_tool(tool_call_name)
        tool_result = asyncio.run(tool_function.execute(
            tool_call.id,
            eval(tool_call_args),
            None
        ))
        print(f"工具执行结果: {tool_result.content[0].text}")
        messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "name": tool_call_name,
                "content": tool_result.content[0].text
        })
try:
    step_number = 1
    max_steps = 4

    while step_number < max_steps:
        response = client.chat(model=model, messages=messages, tools=tools)
        match response.choices[0].finish_reason:
            case StopReason.TOOL_CALLS.value:
                print("检测到工具调用")
                tool_call_handler(response)
            case "stop":
                print(response.choices[0].message.content)
                print("用户请求停止")
                break
            case _:
                print("其他状态")

        step_number += 1

except Exception as e:
    execution.final_result = f"Agent execution failed: {str(e)}"






# while finish_reason is None or finish_reason != "stop":
#     response = client.chat(model=model, messages=messages, tools=tools)
#     print("=" * 50)
#     print("LLM 原始响应:")
#     print(response.choices[0].model_dump_json(indent=4))
#     choice = response.choices[0]
#     finish_reason = normalize_finish_reason(choice.finish_reason)

#     if finish_reason == "tool_calls":
#         print("\n" + "=" * 50)
#         print(f"检测到工具调用: {len(choice.message.tool_calls)} 个")
#         # messages.append(choice.message)
#         assistant_message = choice.message
#         assistant_msg_dict = {
#             "role": "assistant",
#             "content": assistant_message.content or "",
#             "tool_calls": [
#                 {
#                     "id": tool_call.id,
#                     "type": "function",
#                     "function": {
#                         "name": tool_call.function.name,
#                         "arguments": tool_call.function.arguments
#                     }
#                 }
#                 for tool_call in choice.message.tool_calls
#             ]
#         }
#         if hasattr(assistant_message, "reasoning_content") and assistant_message.reasoning_content:
#             assistant_msg_dict["reasoning_content"] = assistant_message.reasoning_content
#         messages.append(assistant_msg_dict)

#         for tool_call in choice.message.tool_calls:
#             print(f"工具调用: {tool_call}")
#             tool_call_name = tool_call.function.name
#             tool_call_args = tool_call.function.arguments
#             tool_function = get_tool(tool_call_name)
#             tool_result = asyncio.run(tool_function.execute(
#                 tool_call.id,
#                 eval(tool_call_args),
#                 None
#             ))
#             print(f"工具执行结果: {tool_result.content[0].text}")
#             messages.append({
#                     "role": "tool",
#                     "tool_call_id": tool_call.id,
#                     "name": tool_call_name,
#                     "content": tool_result.content[0].text
#             })
# print(choice.message.content)












# tool_calls = response.choices[0].message.tool_calls
# if tool_calls:
#     print("\n" + "=" * 50)
#     print(f"检测到工具调用: {len(tool_calls)} 个")

#     assistant_message = response.choices[0].message
#     assistant_msg_dict = {
#         "role": "assistant",
#         "content": assistant_message.content or "",
#         "tool_calls": [
#             {
#                 "id": tc.id,
#                 "type": "function",
#                 "function": {
#                     "name": tc.function.name,
#                     "arguments": tc.function.arguments
#                 }
#             }
#             for tc in tool_calls
#         ]
#     }

#     if hasattr(assistant_message, "reasoning_content") and assistant_message.reasoning_content:
#         assistant_msg_dict["reasoning_content"] = assistant_message.reasoning_content

#     messages.append(assistant_msg_dict)

#     for tool_call in tool_calls:
#         tool_name = tool_call.function.name
#         args = eval(tool_call.function.arguments)

#         print(f"\n工具: {tool_name}")
#         print(f"参数: {args}")

#         tool = get_tool(tool_name)
#         if tool:
#             result = asyncio.run(tool.execute(
#                 tool_call.id,
#                 args,
#                 None
#             ))
#             result_text = result.content[0].text
#             print(f"执行结果: {result_text}")

#             messages.append({
#                 "role": "tool",
#                 "tool_call_id": tool_call.id,
#                 "content": result_text
#             })

#     print("\n" + "=" * 50)
#     print("继续对话，获取最终回答...")

#     response = client.chat(model=model, messages=messages, tools=tools)
#     print("\n最终回答:")
#     print(response.choices[0].message.content)
# else:
#     print("\n没有工具调用，直接回答:")
#     print(response.choices[0].message.content)
