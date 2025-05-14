"""
Transport Registry for the Aider MCP Server.

This module handles registration, capabilities, and subscriptions for transport adapters.
Extracted from ApplicationCoordinator to improve modularity and maintainability.
"""

import asyncio
import logging
import typing
from typing import Dict, Optional, Set

from aider_mcp_server.atoms.event_types import EventTypes
from aider_mcp_server.interfaces.transport_adapter import ITransportAdapter
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


class TransportRegistry:
    """
    Manages transport adapter registration, capabilities, and subscriptions.

    This class is responsible for:
    1. Registering and unregistering transports
    2. Tracking transport capabilities and event subscriptions
    3. Retrieving transports and checking if they exist
    """

    def __init__(self) -> None:
        """Initialize the TransportRegistry."""
        self._transports: Dict[str, ITransportAdapter] = {}
        self._transport_capabilities: Dict[str, Set[EventTypes]] = {}
        self._transport_subscriptions: Dict[str, Set[EventTypes]] = {}

        # Locks for async safety
        self._transports_lock = asyncio.Lock()
        self._transport_capabilities_lock = asyncio.Lock()
        self._transport_subscriptions_lock = asyncio.Lock()

        logger.info("TransportRegistry initialized")

    async def register_transport(
        self, transport_id: str, transport: ITransportAdapter
    ) -> None:
        """
        Registers a new transport adapter.

        Args:
            transport_id: The unique identifier for the transport
            transport: The transport adapter instance
        """
        async with self._transports_lock:
            if transport_id in self._transports:
                logger.warning(
                    f"Transport {transport_id} already registered. Overwriting."
                )
            self._transports[transport_id] = transport
            logger.info(
                f"Transport registered: {transport_id} ({transport.get_transport_type()})"
            )

        # Update capabilities and default subscriptions (outside transports_lock)
        await self.update_transport_capabilities(
            transport_id, transport.get_capabilities()
        )

    async def unregister_transport(self, transport_id: str) -> None:
        """
        Unregisters a transport adapter.

        Args:
            transport_id: The unique identifier for the transport
        """
        transport_exists = False
        async with self._transports_lock:
            if transport_id in self._transports:
                del self._transports[transport_id]
                transport_exists = True
                logger.info(f"Transport unregistered: {transport_id}")
            else:
                logger.warning(
                    f"Attempted to unregister non-existent transport: {transport_id}"
                )

        if transport_exists:
            # Clean up capabilities and subscriptions (outside transports_lock)
            async with self._transport_capabilities_lock:
                if transport_id in self._transport_capabilities:
                    del self._transport_capabilities[transport_id]
            async with self._transport_subscriptions_lock:
                if transport_id in self._transport_subscriptions:
                    del self._transport_subscriptions[transport_id]

    async def update_transport_capabilities(
        self, transport_id: str, capabilities: Set[EventTypes]
    ) -> None:
        """
        Updates the capabilities of a registered transport.

        Args:
            transport_id: The unique identifier for the transport
            capabilities: Set of event types the transport can handle
        """
        async with self._transport_capabilities_lock:
            self._transport_capabilities[transport_id] = capabilities
            logger.debug(f"Updated capabilities for {transport_id}: {capabilities}")

        # By default, subscribe to all capabilities when capabilities are updated
        await self.update_transport_subscriptions(transport_id, capabilities)

    async def update_transport_subscriptions(
        self, transport_id: str, subscriptions: Set[EventTypes]
    ) -> None:
        """
        Updates the event types a transport is subscribed to (replaces existing).

        Args:
            transport_id: The unique identifier for the transport
            subscriptions: Set of event types the transport should be subscribed to
        """
        # Check if transport exists first (read lock)
        transport_exists = await self.transport_exists(transport_id)
        if not transport_exists:
            logger.warning(
                f"Attempted to update subscriptions for non-existent transport: {transport_id}"
            )
            return

        async with self._transport_subscriptions_lock:
            # Validate that subscriptions are a subset of capabilities? Optional.
            self._transport_subscriptions[transport_id] = subscriptions
            logger.debug(f"Updated subscriptions for {transport_id}: {subscriptions}")

    async def subscribe_to_event_type(
        self, transport_id: str, event_type: EventTypes
    ) -> None:
        """
        Subscribes a transport to a specific event type.

        Args:
            transport_id: The unique identifier for the transport
            event_type: The event type to subscribe to
        """
        transport_exists = await self.transport_exists(transport_id)
        if not transport_exists:
            logger.warning(
                f"Attempted to subscribe non-existent transport {transport_id} to {event_type.value}"
            )
            return

        async with self._transport_subscriptions_lock:
            if transport_id not in self._transport_subscriptions:
                self._transport_subscriptions[transport_id] = set()
            self._transport_subscriptions[transport_id].add(event_type)
            logger.debug(f"Transport {transport_id} subscribed to {event_type.value}")

    async def unsubscribe_from_event_type(
        self, transport_id: str, event_type: EventTypes
    ) -> None:
        """
        Unsubscribes a transport from a specific event type.

        Args:
            transport_id: The unique identifier for the transport
            event_type: The event type to unsubscribe from
        """
        async with self._transport_subscriptions_lock:
            if transport_id in self._transport_subscriptions:
                self._transport_subscriptions[transport_id].discard(event_type)
                logger.debug(
                    f"Transport {transport_id} unsubscribed from {event_type.value}"
                )
            # else: No warning needed if transport exists but wasn't subscribed

    async def is_subscribed(self, transport_id: str, event_type: EventTypes) -> bool:
        """
        Checks if a transport is subscribed to a specific event type.

        Args:
            transport_id: The unique identifier for the transport
            event_type: The event type to check

        Returns:
            True if the transport is subscribed to the event type, False otherwise
        """
        async with self._transport_subscriptions_lock:
            subscriptions = self._transport_subscriptions.get(transport_id, set())
            return event_type in subscriptions

    async def get_transport(self, transport_id: str) -> Optional[ITransportAdapter]:
        """
        Gets a transport adapter by ID.

        Args:
            transport_id: The unique identifier for the transport

        Returns:
            The transport adapter instance or None if it doesn't exist
        """
        async with self._transports_lock:
            return self._transports.get(transport_id)

    async def transport_exists(self, transport_id: str) -> bool:
        """
        Checks if a transport ID exists.

        Args:
            transport_id: The unique identifier for the transport

        Returns:
            True if the transport exists, False otherwise
        """
        async with self._transports_lock:
            return transport_id in self._transports

    async def get_transport_capabilities(self, transport_id: str) -> Set[EventTypes]:
        """
        Gets the capabilities of a transport.

        Args:
            transport_id: The unique identifier for the transport

        Returns:
            Set of event types the transport is capable of handling,
            empty set if transport doesn't exist or has no capabilities
        """
        async with self._transport_capabilities_lock:
            return self._transport_capabilities.get(transport_id, set())

    async def get_transport_subscriptions(self, transport_id: str) -> Set[EventTypes]:
        """
        Gets the event types a transport is subscribed to.

        Args:
            transport_id: The unique identifier for the transport

        Returns:
            Set of event types the transport is subscribed to,
            empty set if transport doesn't exist or has no subscriptions
        """
        async with self._transport_subscriptions_lock:
            return self._transport_subscriptions.get(transport_id, set())

    async def get_all_transports(self) -> Dict[str, ITransportAdapter]:
        """
        Gets all registered transports.

        Returns:
            Dictionary mapping transport IDs to transport adapter instances
        """
        async with self._transports_lock:
            return self._transports.copy()

    async def get_all_subscriptions(self) -> Dict[str, Set[EventTypes]]:
        """
        Gets all transport subscriptions.

        Returns:
            Dictionary mapping transport IDs to sets of subscribed event types
        """
        async with self._transport_subscriptions_lock:
            return self._transport_subscriptions.copy()

    async def clear(self) -> None:
        """Clears all registered transports, capabilities, and subscriptions."""
        async with self._transports_lock:
            self._transports.clear()
        async with self._transport_capabilities_lock:
            self._transport_capabilities.clear()
        async with self._transport_subscriptions_lock:
            self._transport_subscriptions.clear()
        logger.info("TransportRegistry cleared")
