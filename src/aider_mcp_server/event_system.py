"""
Event System for the Aider MCP Server.

This module handles event broadcasting and routing to appropriate transports.
Extracted from ApplicationCoordinator to improve modularity and maintainability.
"""

import asyncio
import logging
import typing
import uuid
from typing import Any, Dict, List, Optional, Set, Tuple

from aider_mcp_server.atoms.event_types import EventTypes
from aider_mcp_server.interfaces.transport_adapter import ITransportAdapter
from aider_mcp_server.interfaces.transport_registry import TransportAdapterRegistry
from aider_mcp_server.mcp_types import LoggerFactory, LoggerProtocol

# Initialize the logger factory
get_logger_func: LoggerFactory

try:
    from aider_mcp_server.atoms.logging import get_logger as custom_get_logger

    get_logger_func = typing.cast(LoggerFactory, custom_get_logger)
except ImportError:

    def fallback_get_logger(
        name: str, *args: typing.Any, **kwargs: typing.Any
    ) -> LoggerProtocol:
        logger = logging.getLogger(name)
        if not logger.hasHandlers():
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            if logger.level == logging.NOTSET:
                logger.setLevel(logging.INFO)
        return logger

    get_logger_func = fallback_get_logger

logger = get_logger_func(__name__)


class EventSystem:
    """
    Manages event broadcasting and routing to appropriate transports.

    This class is responsible for:
    1. Broadcasting events to subscribed transports
    2. Filtering events based on transport capabilities
    3. Sending directed events to specific transports
    """

    def __init__(self, transport_registry: TransportAdapterRegistry) -> None:
        """
        Initialize the EventSystem.

        Args:
            transport_registry: The TransportAdapterRegistry instance to use for transport management
        """
        self._transport_registry = transport_registry
        self._loop = asyncio.get_event_loop()
        self._is_shutting_down = False

        logger.info("EventSystem initialized")

    def start_shutdown(self) -> None:
        """Signal that the system is shutting down, preventing further event distribution."""
        self._is_shutting_down = True
        logger.info("EventSystem marked as shutting down")

    def is_shutting_down(self) -> bool:
        """Check if the system is in the process of shutting down."""
        return self._is_shutting_down

    async def broadcast_event(
        self,
        event_type: EventTypes,
        data: Dict[str, Any],
        exclude_transport_id: Optional[str] = None,
    ) -> None:
        """
        Broadcasts an event to all subscribed transports, optionally excluding one.

        Args:
            event_type: The type of event to broadcast
            data: The event data payload
            exclude_transport_id: Optional transport ID to exclude from the broadcast
        """
        if self._is_shutting_down:
            return

        logger.debug(
            f"Broadcasting event {event_type.value} (excluding {exclude_transport_id}): {data}"
        )

        # Extract potential request details (parameters) if available in data for filtering
        request_params = data.get("details", {}).get("parameters")
        await self._send_event_to_transports(
            event_type,
            data,
            exclude_transport_id=exclude_transport_id,
            request_details=request_params,  # Pass params if available
        )

    async def send_event_to_transport(
        self,
        transport_id: str,
        event_type: EventTypes,
        data: Dict[str, Any],
    ) -> None:
        """
        Sends a single event directly to a specific transport, if it exists.
        Does not check subscriptions. Primarily used for direct responses like errors.

        Args:
            transport_id: The ID of the transport to send the event to
            event_type: The type of event to send
            data: The event data payload
        """
        if self._is_shutting_down:
            return

        transport = await self._transport_registry.get_transport(transport_id)
        if transport:
            logger.debug(
                f"Sending direct event {event_type.value} to transport {transport_id} (Request: {data.get('request_id', 'N/A')})"
            )
            try:
                # Run send_event, but don't block if it takes time
                self._loop.create_task(
                    transport.send_event(event_type, data),
                    name=f"direct-send-{event_type.value}-{transport_id}-{data.get('request_id', uuid.uuid4())}",
                )
            except Exception as e:
                # This catch block might not be effective if send_event is async and raises later
                logger.error(
                    f"Error creating task to send direct event {event_type.value} to transport {transport_id}: {e}",
                    exc_info=True,
                )
        else:
            logger.warning(
                f"Attempted to send direct event {event_type.value} to non-existent transport {transport_id}"
            )

    async def _should_send_to_transport(
        self,
        transport_id: str,
        transport: ITransportAdapter,
        event_type: EventTypes,
        data: Dict[str, Any],
        originating_transport_id: Optional[str],
        subscriptions: Dict[str, Set[EventTypes]],
        request_details: Optional[Dict[str, Any]],
    ) -> bool:
        """
        Determine if an event should be sent to a specific transport based on
        subscription status, origin rules, and transport-specific filtering.

        Args:
            transport_id: The ID of the transport
            transport: The transport adapter instance
            event_type: The type of event
            data: The event data payload
            originating_transport_id: Optional ID of the transport that originated the request
            subscriptions: Dictionary mapping transport IDs to subscribed event types
            request_details: Optional original request parameters for context

        Returns:
            True if the event should be sent to the transport, False otherwise
        """
        # Check subscription
        is_subscribed = event_type in subscriptions.get(transport_id, set())
        if not is_subscribed:
            return False

        # Special handling for STATUS events
        if event_type == EventTypes.STATUS and originating_transport_id is not None:
            if transport_id != originating_transport_id:
                return False

        # Check transport-specific filtering
        try:
            # Pass original request parameters if available
            if not transport.should_receive_event(event_type, data, request_details):
                logger.debug(
                    f"Transport {transport_id} filtered out event {event_type.value} for request {data.get('request_id', 'N/A')}"
                )
                return False
        except Exception as e:
            logger.error(
                f"Error calling should_receive_event for transport {transport_id}: {e}",
                exc_info=True,
            )
            return False

        return True

    def _create_send_event_task(
        self,
        transport: ITransportAdapter,
        transport_id: str,
        event_type: EventTypes,
        data: Dict[str, Any],
    ) -> asyncio.Task:
        """
        Create a task to send an event to a transport.

        Args:
            transport: The transport adapter instance
            transport_id: The ID of the transport
            event_type: The type of event
            data: The event data payload

        Returns:
            An asyncio task that will send the event
        """
        request_id = data.get("request_id", uuid.uuid4())
        logger.debug(
            f"Queueing event {event_type.value} for transport {transport_id} (Request: {request_id})"
        )
        return self._loop.create_task(
            transport.send_event(event_type, data),
            name=f"send-{event_type.value}-{transport_id}-{request_id}",
        )

    async def _log_originating_transport_status(
        self,
        event_type: EventTypes,
        data: Dict[str, Any],
        originating_transport_id: str,
        sent_to: Set[str],
        transports_to_notify: List[Tuple[str, ITransportAdapter]],
        subscriptions: Dict[str, Set[EventTypes]],
    ) -> None:
        """
        Log warnings if the originating transport didn't receive an event it might have expected.

        Args:
            event_type: The type of event
            data: The event data payload
            originating_transport_id: The ID of the transport that originated the request
            sent_to: Set of transport IDs that received the event
            transports_to_notify: List of (transport_id, transport) tuples
            subscriptions: Dictionary mapping transport IDs to subscribed event types
        """
        request_id = data.get("request_id", "N/A")
        origin_subscribed = event_type in subscriptions.get(
            originating_transport_id, set()
        )
        origin_exists = any(
            t_id == originating_transport_id for t_id, _ in transports_to_notify
        )

        if origin_exists and not origin_subscribed:
            logger.warning(
                f"Event {event_type.value} for request {request_id} was not sent to originating transport {originating_transport_id} because it was not subscribed."
            )
        elif not origin_exists:
            logger.warning(
                f"Event {event_type.value} for request {request_id} could not be sent to originating transport {originating_transport_id} because it was not found (likely unregistered)."
            )

    async def _handle_task_results(
        self,
        tasks: List[asyncio.Task],
        results: List[Any],
    ) -> None:
        """
        Process the results of event sending tasks and log any errors.

        Args:
            tasks: List of asyncio tasks
            results: List of task results
        """
        for i, result in enumerate(results):
            if not isinstance(result, Exception):
                continue

            task_name = tasks[i].get_name()
            log_transport_id = "unknown"
            try:
                parts = task_name.split("-")
                if len(parts) >= 4 and parts[0] == "send":
                    log_transport_id = parts[2]
            except Exception:
                logger.warning(
                    f"Failed to parse transport ID from task name {task_name}"
                )

            # Avoid logging CancelledError stack traces unless debugging needed
            log_exc_info = (
                result if not isinstance(result, asyncio.CancelledError) else None
            )
            logger.error(
                f"Error sending event via task {task_name} (Transport: {log_transport_id}): {result}",
                exc_info=log_exc_info,
            )

    async def _send_event_to_transports(
        self,
        event_type: EventTypes,
        data: Dict[str, Any],
        originating_transport_id: Optional[str] = None,
        exclude_transport_id: Optional[str] = None,
        request_details: Optional[
            Dict[str, Any]
        ] = None,  # Original request params for filtering
    ) -> None:
        """
        Internal helper to send an event to relevant transports based on subscriptions.
        Includes logic to check transport's should_receive_event method if available.

        Args:
            event_type: The type of event
            data: The event data payload
            originating_transport_id: Optional ID of the transport that originated the request
            exclude_transport_id: Optional ID of the transport to exclude
            request_details: Optional original request parameters for context
        """
        if self._is_shutting_down:
            return

        tasks = []
        sent_to = set()

        # Get transports and subscriptions
        transports = await self._transport_registry.get_all_transports()
        subscriptions = await self._transport_registry.get_all_subscriptions()

        # Process transports
        for transport_id, transport in transports.items():
            if transport_id == exclude_transport_id:
                continue

            # Check if we should send to this transport
            should_send = await self._should_send_to_transport(
                transport_id,
                transport,
                event_type,
                data,
                originating_transport_id,
                subscriptions,
                request_details,
            )

            if should_send:
                tasks.append(
                    self._create_send_event_task(
                        transport, transport_id, event_type, data
                    )
                )
                sent_to.add(transport_id)

        # Check if the originating transport should have received it but didn't
        if (
            originating_transport_id
            and originating_transport_id not in sent_to
            and originating_transport_id != exclude_transport_id
        ):
            await self._log_originating_transport_status(
                event_type,
                data,
                originating_transport_id,
                sent_to,
                list(transports.items()),
                subscriptions,
            )

        # Log if no transports received the event
        if not sent_to:
            # Avoid logging warning if it was a STATUS event only meant for origin and origin wasn't subscribed/found
            is_status_for_origin_only = (
                event_type == EventTypes.STATUS and originating_transport_id is not None
            )
            if not is_status_for_origin_only:
                logger.debug(
                    f"No transports subscribed or eligible to receive event {event_type.value} (Request: {data.get('request_id', 'N/A')})"
                )

        # Wait for all send tasks to complete
        if tasks:
            results = await asyncio.gather(*tasks, return_exceptions=True)
            await self._handle_task_results(tasks, results)
