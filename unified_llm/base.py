from abc import ABC, abstractmethod
from enum import Enum
from typing import List, Dict, Any, Iterator

class BaseAdapter(ABC):
    @abstractmethod
    def chat(self, model: str, messages: List[Dict], **kwargs) -> Dict[str, Any]:
        pass
    
    @abstractmethod
    def chat_stream(self, model: str, messages: List[Dict], **kwargs) -> Iterator[str]:
        """流式对话请求"""
        pass
    
    @abstractmethod
    def parse_response(self, response) -> str:
        """解析响应，提取文本内容"""
        pass    

class StopReason(Enum):
    STOP = "stop"
    TOOL_CALLS = "tool_calls"
    LENGTH = "length"
    CONTENT_FILTER = "content_filter"
    ERROR = "error"

FINISH_REASON_MAP = {
    #OpenAI
    "stop": StopReason.STOP,
    "tool_calls": StopReason.TOOL_CALLS,
    "length": StopReason.LENGTH,
    "content_filter": StopReason.CONTENT_FILTER,
    # Anthropic (示例)
    "end_turn": StopReason.STOP,
    "tool_use": StopReason.TOOL_CALLS,
    "max_tokens": StopReason.LENGTH,
}

def normalize_finish_reason(reason: str) -> StopReason:
    return FINISH_REASON_MAP.get(reason, StopReason.ERROR).value
    
