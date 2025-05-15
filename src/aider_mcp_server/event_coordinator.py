"""
Event Coordinator for handling event publishing and subscription functionality.
"""

from typing import Any, Dict, Optional, Set

from aider_mcp_server.atoms.event_types import EventTypes
from aider_mcp_server.event_system import EventSystem
from aider_mcp_server.interfaces.transport_registry import TransportAdapterRegistry
from aider_mcp_server.mcp_types import LoggerFactory


class EventCoordinator:
    def __init__(
        self,
        transport_registry: TransportAdapterRegistry,
        logger_factory: LoggerFactory,
    ) -> None:
        self._transport_registry = transport_registry
        self._logger = logger_factory(__name__)
        self._event_system = EventSystem(transport_registry)

    async def subscribe_to_event_type(
        self, transport_id: str, event_type: EventTypes
    ) -> None:
        await self._event_system.subscribe_to_event_type(transport_id, event_type)

    async def unsubscribe_from_event_type(
        self, transport_id: str, event_type: EventTypes
    ) -> None:
        await self._event_system.unsubscribe_from_event_type(transport_id, event_type)

    async def update_transport_capabilities(
        self, transport_id: str, capabilities: Set[EventTypes]
    ) -> None:
        await self._event_system.update_transport_capabilities(
            transport_id, capabilities
        )

    async def update_transport_subscriptions(
        self, transport_id: str, subscriptions: Set[EventTypes]
    ) -> None:
        await self._event_system.update_transport_subscriptions(
            transport_id, subscriptions
        )

    async def broadcast_event(
        self,
        event_type: EventTypes,
        data: Dict[str, Any],
        exclude_transport_id: Optional[str] = None,
        test_mode: bool = False,  # Added for testing
    ) -> None:
        await self._event_system.broadcast_event(
            event_type, data, exclude_transport_id, test_mode=test_mode
        )

    async def send_event_to_transport(
        self,
        transport_id: str,
        event_type: EventTypes,
        data: Dict[str, Any],
        test_mode: bool = False,
    ) -> None:
        await self._event_system.send_event_to_transport(
            transport_id, event_type, data, test_mode=test_mode
        )

    async def is_subscribed(self, transport_id: str, event_type: EventTypes) -> bool:
        return await self._event_system.is_subscribed(transport_id, event_type)
