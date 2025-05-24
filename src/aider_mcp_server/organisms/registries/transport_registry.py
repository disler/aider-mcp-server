import logging
import typing
from typing import Dict, Optional, Set

from aider_mcp_server.atoms.types.event_types import EventTypes
from aider_mcp_server.interfaces.transport_adapter import ITransportAdapter
from aider_mcp_server.atoms.types.mcp_types import LoggerFactory, LoggerProtocol

# Initialize the logger factory
get_logger_func: LoggerFactory

try:
    from aider_mcp_server.atoms.logging.logger import get_logger as custom_get_logger

    get_logger_func = typing.cast(LoggerFactory, custom_get_logger)
except ImportError:

    def fallback_get_logger(name: str, *args: typing.Any, **kwargs: typing.Any) -> LoggerProtocol:
        logger = logging.getLogger(name)
        if not logger.hasHandlers():
            handler = logging.StreamHandler()
            formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            if logger.level == logging.NOTSET:
                logger.setLevel(logging.INFO)

        class CustomLogger(LoggerProtocol):
            def debug(self, message: str, **kwargs: typing.Any) -> None:
                logger.debug(message, **kwargs)

            def info(self, message: str, **kwargs: typing.Any) -> None:
                logger.info(message, **kwargs)

            def warning(self, message: str, **kwargs: typing.Any) -> None:
                logger.warning(message, **kwargs)

            def error(self, message: str, **kwargs: typing.Any) -> None:
                logger.error(message, **kwargs)

            def critical(self, message: str, **kwargs: typing.Any) -> None:
                logger.critical(message, **kwargs)

            def exception(self, message: str, **kwargs: typing.Any) -> None:
                logger.exception(message, **kwargs)

            def verbose(self, message: str, **kwargs: typing.Any) -> None:
                logger.debug(message, **kwargs)

        return CustomLogger()

    get_logger_func = fallback_get_logger

logger = get_logger_func(__name__)


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
