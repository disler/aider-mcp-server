"""
A simple event system for managing event subscriptions and broadcasting
based on callbacks. This aligns with Task 2 specification.
"""

import asyncio
import logging
from typing import Any, Awaitable, Callable, Dict, List

# Define EventCallback type as per Task 2, ensuring it's Awaitable
EventCallback = Callable[[Dict[str, Any]], Awaitable[None]]

# Basic logger for this module
logger = logging.getLogger(__name__)
# Configure basic logging if no handlers are present (e.g., when run in isolation or tests)
if not logger.hasHandlers():
    _handler = logging.StreamHandler()
    _formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    _handler.setFormatter(_formatter)
    logger.addHandler(_handler)
    # Set a default level; can be overridden by application's logging config
    if logger.level == logging.NOTSET:
        logger.setLevel(logging.INFO)


class EventSystem:
    """
    A simple event system for managing event subscriptions and broadcasting.
    This system allows components to subscribe to specific event types (by string name)
    and receive notifications when those events are broadcast.

    It uses asynchronous callbacks and ensures that one callback's failure
    does not prevent others from executing.
    """

    def __init__(self) -> None:
        """
        Initializes the EventSystem with an empty subscriber list and an asyncio Lock
        to ensure thread-safe modifications to subscribers.
        """
        self._subscribers: Dict[str, List[EventCallback]] = {}
        self._lock = asyncio.Lock()
        logger.info("Simple EventSystem initialized.")

    async def subscribe(self, event_type: str, callback: EventCallback) -> None:
        """
        Subscribes an asynchronous callback to a specific event type.

        If the callback is already subscribed to the event type, this operation
        is a no-op.

        Args:
            event_type: The string identifier of the event type to subscribe to
                        (e.g., "user_created", "item_updated").
            callback: The asynchronous callback function to execute when the event occurs.
                      The callback must accept a single argument: event_data (Dict[str, Any]).
        """
        async with self._lock:
            if event_type not in self._subscribers:
                self._subscribers[event_type] = []
            if callback not in self._subscribers[event_type]:
                self._subscribers[event_type].append(callback)
                logger.debug(
                    f"Callback '{getattr(callback, '__name__', repr(callback))}' subscribed to event type '{event_type}'."
                )
            else:
                logger.debug(
                    f"Callback '{getattr(callback, '__name__', repr(callback))}' already subscribed to event type '{event_type}'."
                )

    async def unsubscribe(self, event_type: str, callback: EventCallback) -> None:
        """
        Unsubscribes an asynchronous callback from a specific event type.

        If the event type does not exist or the callback is not found for that
        event type, this operation is a no-op.

        Args:
            event_type: The string identifier of the event type to unsubscribe from.
            callback: The asynchronous callback function to remove.
        """
        async with self._lock:
            if event_type in self._subscribers and callback in self._subscribers[event_type]:
                self._subscribers[event_type].remove(callback)
                logger.debug(
                    f"Callback '{getattr(callback, '__name__', repr(callback))}' unsubscribed from event type '{event_type}'."
                )
                # If no subscribers are left for an event type, remove the event type key
                if not self._subscribers[event_type]:
                    del self._subscribers[event_type]
                    logger.debug(f"Event type '{event_type}' removed as no subscribers are left.")
            else:
                logger.debug(
                    f"Callback '{getattr(callback, '__name__', repr(callback))}' not found for event type '{event_type}' or event type does not exist."
                )

    async def broadcast(self, event_type: str, event_data: Dict[str, Any]) -> None:
        """
        Broadcasts an event to all subscribed asynchronous callbacks for the given event type.

        Callbacks are executed concurrently. If a callback raises an exception,
        it is logged, and other callbacks for the same event type are still executed.

        Args:
            event_type: The string identifier of the event to broadcast.
            event_data: The data payload associated with the event.
        """
        callbacks_to_execute = []
        async with self._lock:
            if event_type in self._subscribers:
                # Create a copy of the list to iterate over, preventing issues if
                # a callback modifies the subscription list during execution.
                callbacks_to_execute = list(self._subscribers[event_type])

        if not callbacks_to_execute:
            logger.debug(f"No subscribers for event type '{event_type}'. Event data: {event_data}")
            return

        logger.debug(
            f"Broadcasting event type '{event_type}' to {len(callbacks_to_execute)} subscriber(s). Data: {event_data}"
        )

        # Prepare tasks for all callbacks to run them concurrently
        tasks = [self._execute_callback(callback, event_data, event_type) for callback in callbacks_to_execute]

        # asyncio.gather will run all tasks. Errors within _execute_callback are logged there.
        # We don't need return_exceptions=True if _execute_callback handles its own exceptions
        # and doesn't re-raise them in a way that should stop gather.
        await asyncio.gather(*tasks)

    async def _execute_callback(self, callback: EventCallback, event_data: Dict[str, Any], event_type: str) -> None:
        """
        Helper method to execute a single callback and handle/log potential errors.
        This ensures that one faulty callback does not affect others.
        """
        try:
            callback_name = getattr(callback, "__name__", repr(callback))
            logger.debug(f"Executing callback '{callback_name}' for event type '{event_type}'.")
            await callback(event_data)
        except Exception as e:
            # Log the error but do not re-raise, allowing other callbacks to proceed.
            logger.error(
                f"Error in event callback '{getattr(callback, '__name__', repr(callback))}' for event type '{event_type}': {e}",
                exc_info=True,  # Include stack trace in the log
            )
