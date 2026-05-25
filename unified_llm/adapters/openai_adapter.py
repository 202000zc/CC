from ast import mod
import os
from turtle import mode
from typing import List, Dict, Any, Iterator
from openai import OpenAI
from ..base import BaseAdapter

class OpenAIAdapter(BaseAdapter):
    def __init__(self, api_key: str, base_url: str):
        self.client = OpenAI(api_key=api_key, base_url=base_url)

    @property
    def provider_name(self) -> str:
        return "openai"
    

    def convert_messages(self, messages: List[Dict]) -> Dict[str, Any]:
        return {"messages": messages}

    def chat(self, model: str, messages: List[Dict], **kwargs) -> Dict[str, Any]:
        payload = self.convert_messages(messages)
        payload["model"] = model
        payload.update(kwargs)
        return self.client.chat.completions.create(**payload)
    

    def chat_stream(self, model: str, messages: List[Dict], **kwargs) -> Iterator[str]:
       print(f"[DEBUG] 开始流式请求: model={model}")
       payload = self.convert_messages(messages)
       payload["model"] = model
       payload["stream"] = True
       payload.update(kwargs)
    #    print(f"[DEBUG] payload: {payload}")

       stream = self.client.chat.completions.create(**payload)
       print(f"[DEBUG] stream 对象创建成功，开始迭代")

       for chunk in stream:
        #    print(f"[DEBUG] 收到 chunk: {chunk}", flush=True)
           if chunk.choices[0].delta.content:
               yield chunk.choices[0].delta.content
       print(f"[DEBUG] 流式结束")
    
    def parse_response(self, response) -> str:
       return response.choices[0].message.content