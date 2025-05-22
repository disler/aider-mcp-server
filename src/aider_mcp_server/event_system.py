"""
Simple Event System for the Aider MCP Server.

This module provides a basic callback-based event subscription and broadcasting mechanism.
This implementation corresponds to Task 2 requirements.
"""

import asyncio
import logging
import typing
from typing import Any, Dict, List, Callable, Awaitable

from aider_mcp_server.mcp_types import LoggerFactory, LoggerProtocol

# Initialize the logger factory
get_logger_func: LoggerFactory

try:
    from aider_mcp_server.atoms.logging import get_logger as custom_get_logger

    get_logger_func = typing.cast(LoggerFactory, custom_get_logger)
except ImportError:

    def fallback_get_logger(name: str, *args: typing.Any, **kwargs: typing.Any) -> LoggerProtocol:
        logger_instance = logging.getLogger(name)
        if not logger_instance.hasHandlers():
            handler = logging.StreamHandler()
            formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
            handler.setFormatter(formatter)
            logger_instance.addHandler(handler)
            if logger_instance.level == logging.NOTSET:
                logger_instance.setLevel(logging.INFO)

        class CustomLogger(LoggerProtocol):
            def debug(self, message: str, **kwargs: typing.Any) -> None:
                logger_instance.debug(message, **kwargs)

            def info(self, message: str, **kwargs: typing.Any) -> None:
                logger_instance.info(message, **kwargs)

            def warning(self, message: str, **kwargs: typing.Any) -> None:
                logger_instance.warning(message, **kwargs)

            def error(self, message: str, **kwargs: typing.Any) -> None:
                logger_instance.error(message, **kwargs)

            def critical(self, message: str, **kwargs: typing.Any) -> None:
                logger_instance.critical(message, **kwargs)

            def exception(self, message: str, **kwargs: typing.Any) -> None:
                logger_instance.exception(message, **kwargs)

            def verbose(self, message: str, **kwargs: typing.Any) -> None:
                # Map verbose to debug for standard logger
                logger_instance.debug(message, **kwargs)

        return CustomLogger()

    get_logger_func = fallback_get_logger

logger = get_logger_func(__name__)

EventCallback = Callable[[Dict[str, Any]], Awaitable[None]]


class EventSystem:
    """
    Manages event subscriptions and broadcasting using a simple callback mechanism.
    """

    def __init__(self) -> None:
        """
        Initialize the EventSystem.
        """
        self._subscribers: Dict[str, List[EventCallback]] = {}
        self._lock = asyncio.Lock()
        logger.info("Simple EventSystem initialized")

    async def subscribe(self, event_type: str, callback: EventCallback) -> None:
        """
        Subscribe a callback to a specific event type.

        Args:
            event_type: The type of event to subscribe to (e.g., "user_created").
            callback: The asynchronous callback function to execute when the event occurs.
        """
        async with self._lock:
            if event_type not in self._subscribers:
                self._subscribers[event_type] = []

            # Avoid adding the same callback multiple times for the same event type
            if callback not in self._subscribers[event_type]:
                self._subscribers[event_type].append(callback)
                logger.debug(f"Callback {getattr(callback, '__name__', str(callback))} subscribed to event type '{event_type}'")
            else:
                logger.debug(f"Callback {getattr(callback, '__name__', str(callback))} already subscribed to event type '{event_type}'")

    async def unsubscribe(self, event_type: str, callback: EventCallback) -> None:
        """
        Unsubscribe a callback from a specific event type.

        Args:
            event_type: The type of event to unsubscribe from.
            callback: The callback function to remove.
        """
        async with self._lock:
            if event_type in self._subscribers and callback in self._subscribers[event_type]:
                self._subscribers[event_type].remove(callback)
                logger.debug(f"Callback {getattr(callback, '__name__', str(callback))} unsubscribed from event type '{event_type}'")
                # If no more subscribers for this event type, remove the key
                if not self._subscribers[event_type]:
                    del self._subscribers[event_type]
                    logger.debug(f"Event type '{event_type}' removed as it has no more subscribers.")
            else:
                logger.debug(
                    f"Callback {getattr(callback, '__name__', str(callback))} not found for event type '{event_type}', or event type not registered."
                )

    async def broadcast(self, event_type: str, event_data: Dict[str, Any]) -> None:
        """
        Broadcast an event to all subscribed callbacks for that event type.

        Args:
            event_type: The type of event being broadcast.
            event_data: The data associated with the event.
        """
        callbacks_to_invoke: List[EventCallback] = []
        async with self._lock:
            if event_type in self._subscribers:
                # Create a copy to iterate over, allowing callbacks to unsubscribe during iteration
                # without affecting the current broadcast.
                callbacks_to_invoke = self._subscribers[event_type][:]

        if not callbacks_to_invoke:
            logger.debug(f"No subscribers for event type '{event_type}'. Event data: {event_data}")
            return

        logger.debug(f"Broadcasting event '{event_type}' to {len(callbacks_to_invoke)} subscribers. Data: {event_data}")

        for callback in callbacks_to_invoke:
            try:
                await callback(event_data)
            except Exception as e:
                # Log error but continue with other callbacks
                logger.error(
                    f"Error in event callback {getattr(callback, '__name__', str(callback))} for event type '{event_type}': {e}",
                    exc_info=True  # Includes stack trace
                )
