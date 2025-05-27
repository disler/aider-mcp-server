from typing import Dict, Optional, Set

from aider_mcp_server.atoms.logging.fallback_logger import get_fallback_logger_factory, get_logger_with_fallback
from aider_mcp_server.atoms.types.event_types import EventTypes
from aider_mcp_server.interfaces.transport_adapter import ITransportAdapter

# Use shared logger factory to eliminate duplication
get_logger_func = get_fallback_logger_factory()
logger = get_logger_with_fallback(__name__)


class TransportRegistry:
    def __init__(self) -> None:
        self.transports: Dict[str, ITransportAdapter] = {}
        self.capabilities: Dict[str, Set[EventTypes]] = {}
        self.subscriptions_by_event: Dict[EventTypes, Set[str]] = {}
        self.subscriptions_by_transport: Dict[str, Set[EventTypes]] = {}

    def register_transport(self, transport_id: str, transport: ITransportAdapter) -> None:
        self.transports[transport_id] = transport
        self.capabilities[transport_id] = set()
        self.subscriptions_by_transport[transport_id] = set()

    def unregister_transport(self, transport_id: str) -> None:
        if transport_id in self.transports:
            del self.transports[transport_id]
            del self.capabilities[transport_id]
            del self.subscriptions_by_transport[transport_id]

    def update_transport_capabilities(self, transport_id: str, capabilities: Set[EventTypes]) -> None:
        if transport_id in self.capabilities:
            self.capabilities[transport_id] = capabilities

    def update_transport_subscriptions(self, transport_id: str, subscriptions: Set[EventTypes]) -> None:
        if transport_id in self.subscriptions_by_transport:
            self.subscriptions_by_transport[transport_id] = subscriptions

    def subscribe_to_event_type(self, transport_id: str, event_type: EventTypes) -> None:
        if event_type not in self.subscriptions_by_event:
            self.subscriptions_by_event[event_type] = set()
        self.subscriptions_by_event[event_type].add(transport_id)

    def unsubscribe_from_event_type(self, transport_id: str, event_type: EventTypes) -> None:
        if event_type in self.subscriptions_by_event and transport_id in self.subscriptions_by_event[event_type]:
            self.subscriptions_by_event[event_type].remove(transport_id)

    def is_subscribed(self, transport_id: str, event_type: EventTypes) -> bool:
        return event_type in self.subscriptions_by_event and transport_id in self.subscriptions_by_event[event_type]

    def get_transport(self, transport_id: str) -> Optional[ITransportAdapter]:
        return self.transports.get(transport_id)

    def transport_exists(self, transport_id: str) -> bool:
        return transport_id in self.transports

    def get_transport_capabilities(self, transport_id: str) -> Set[EventTypes]:
        if transport_id in self.capabilities:
            return self.capabilities[transport_id]
        return set()

    def get_transport_subscriptions(self, transport_id: str) -> Set[EventTypes]:
        if transport_id in self.subscriptions_by_transport:
            return self.subscriptions_by_transport[transport_id]
        return set()

    def get_all_transports(self) -> Dict[str, ITransportAdapter]:
        return self.transports

    def get_all_subscriptions(self) -> Dict[EventTypes, Set[str]]:
        return self.subscriptions_by_event

    def clear(self) -> None:
        self.transports.clear()
        self.capabilities.clear()
        self.subscriptions_by_event.clear()
        self.subscriptions_by_transport.clear()
