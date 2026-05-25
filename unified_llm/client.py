from typing import List, Dict, Any, Iterator
from .adapters.openai_adapter import OpenAIAdapter
from .base import BaseAdapter


class UnifiedLLMClient:

    _adapters = {
        "openai": OpenAIAdapter
    }


    def __init__(self, provider: str, api_key: str = None, base_url: str = None) -> None:
        if provider not in self._adapters:
            raise ValueError(f"不支持的供应商: {provider}，支持: {list(self._adapters.keys())}")

        if not api_key:
            raise ValueError(f"供应商 {provider} 的 api_key 不能为空")
        if not base_url:
            raise ValueError(f"供应商 {provider} 的 base_url 不能为空")

        self.adapter: BaseAdapter = self._adapters[provider](api_key, base_url)
        self.provider = provider

    def chat(self, model: str, messages: List[Dict], **kwargs) -> str:
        response = self.adapter.chat(model, messages, **kwargs)
        # return self.adapter.parse_response(response)
        return response

    def chat_stream(self, model: str, messages: List[Dict], **kwargs) -> Iterator[str]:
        yield from self.adapter.chat_stream(model, messages, **kwargs)

    @classmethod
    def register_adapter(cls, name: str, adapter_class):
        """注册新的适配器（扩展用）"""
        cls._adapters[name] = adapter_class