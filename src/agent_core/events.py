"""事件系统 - 发布订阅模式"""

from typing import Any, Callable, Awaitable
import logging
import asyncio

logger = logging.getLogger(__name__)

EventHandler = Callable[[Any], Awaitable[None] | None]


class EventEmitter:
    def __init__(self):
        self._handlers: dict[str, list[EventHandler]] = {}

    def on(self, event_type: str, handler: EventHandler) -> Callable[[], None]:
        if event_type not in self._handlers:
            self._handlers[event_type] = []
        self._handlers[event_type].append(handler)

        def unsubscribe():
            if event_type in self._handlers:
                try:
                    self._handlers[event_type].remove(handler)
                except ValueError:
                    pass
                if not self._handlers[event_type]:
                    del self._handlers[event_type]

        return unsubscribe

    async def emit(self, event_type: str, data: Any = None) -> None:
        handlers = self._handlers.get(event_type, [])
        for handler in handlers:
            try:
                result = handler(data)
                if asyncio.iscoroutine(result):
                    await result
            except Exception as e:
                logger.error(f"事件处理器错误 [{event_type}]: {e}")

    def emit_sync(self, event_type: str, data: Any = None) -> None:
        handlers = self._handlers.get(event_type, [])
        for handler in handlers:
            try:
                handler(data)
            except Exception as e:
                logger.error(f"事件处理器错误 [{event_type}]: {e}")

    def off(self, event_type: str = None) -> None:
        if event_type:
            self._handlers.pop(event_type, None)
        else:
            self._handlers.clear()

    def listener_count(self, event_type: str) -> int:
        return len(self._handlers.get(event_type, []))
